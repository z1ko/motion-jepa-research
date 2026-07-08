"""Loading and labeling windows from the held-out eval datasets.

The held-out eval datasets (SOMA, HumanEva, DanceDB) each encode a downstream
label in the trial filename rather than as a metadata column, so labels are
parsed out of the filename and joined onto the window table here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch as t
from torch.utils.data import Dataset

from motion_jepa.dataset import (
    MotionZarrStore,
    _path_of_samples_index,
    _path_of_windows_index,
    load_normalization_stats,
)
from motion_jepa.utils import center_root_channels, signed_log1p_tau

# ================================================================================================================
# EVAL LABEL PARSING
# ================================================================================================================

def parse_eval_label(dataset: str, trial: str) -> tuple[str | None, str | None, str | None, str]:
    stem = Path(trial).stem.removesuffix("_stageii")
    parts = stem.split("_")

    if dataset == "SOMA":
        action = parts[0] if parts else None
        return action, None, action, "action" if action else "unknown"

    if dataset == "HumanEva":
        if parts and parts[-1].isdigit():
            parts = parts[:-1]
        action = "_".join(parts) if parts else None
        return action, None, action, "action" if action else "unknown"

    if dataset == "DanceDB":
        emotion = parts[1] if len(parts) >= 2 else None
        return None, emotion, emotion, "emotion" if emotion else "unknown"

    return None, None, None, "unknown"


def add_eval_labels(rows: pl.DataFrame) -> pl.DataFrame:
    if "dataset" not in rows.columns or "trial" not in rows.columns:
        raise ValueError("Rows must contain 'dataset' and 'trial' to parse eval labels.")

    actions: list[str | None] = []
    emotions: list[str | None] = []
    labels: list[str | None] = []
    kinds: list[str] = []

    for row in rows.select(["dataset", "trial"]).iter_rows(named=True):
        action, emotion, label, kind = parse_eval_label(
            dataset=str(row["dataset"]),
            trial=str(row["trial"]),
        )
        actions.append(action)
        emotions.append(emotion)
        labels.append(label)
        kinds.append(kind)

    return rows.with_columns(
        pl.Series("action", actions, dtype=pl.Utf8),
        pl.Series("emotion", emotions, dtype=pl.Utf8),
        pl.Series("eval_label", labels, dtype=pl.Utf8),
        pl.Series("eval_label_kind", kinds, dtype=pl.Utf8),
    )


def load_window_table(
    *,
    root: Path | str,
    split: str,
    dataset: str | None = None,
    max_windows: int | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Join windows.parquet with samples.parquet and attach eval labels."""
    root = Path(root)
    windows = pl.read_parquet(_path_of_windows_index(root))
    samples = pl.read_parquet(_path_of_samples_index(root))

    windows = windows.filter(pl.col("split") == split)
    if windows.is_empty():
        raise ValueError(f"No windows found for split={split!r}")

    sample_cols = [col for col in samples.columns if col != "split"]
    rows = windows.join(samples.select(sample_cols), on="suid", how="left")
    if rows.select(pl.col("dataset").is_null().any()).item():
        raise ValueError("Some windows have no matching sample metadata.")

    rows = add_eval_labels(rows)

    if dataset is not None:
        rows = rows.filter(pl.col("dataset") == dataset)
        if rows.is_empty():
            raise ValueError(f"No windows found for split={split!r}, dataset={dataset!r}")

    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("--max-windows must be positive.")
        if rows.height > max_windows:
            rows = rows.sample(n=max_windows, seed=seed, shuffle=True)

    return rows

# ================================================================================================================
# LABEL FILTERING
# ================================================================================================================

def filter_labeled_rows(rows: pl.DataFrame, *, min_label_subjects: int) -> pl.DataFrame:
    """Drop unlabeled rows and any label seen in too few distinct subjects.

    A label with fewer distinct subjects than `min_label_subjects` can never
    appear in both the train and held-out side of a leave-subject-out fold,
    so it's unevaluable noise (e.g. DanceDB's "Haniotikos"/"Mix" -- filename
    parsing artifacts from trials that don't follow the usual naming
    convention, each contributed by a single one-off subject).
    """
    rows = rows.filter(pl.col("eval_label").is_not_null())

    label_subject_counts = (
        rows.group_by("eval_label")
        .agg(pl.col("subject").n_unique().alias("n_subjects"))
    )
    valid_labels = label_subject_counts.filter(
        pl.col("n_subjects") >= min_label_subjects
    )["eval_label"].to_list()

    return rows.filter(pl.col("eval_label").is_in(valid_labels))

# ================================================================================================================
# WINDOW LOADING
# ================================================================================================================

class WindowRowsDataset(Dataset):
    """Fetch + center + normalize + pad an arbitrary list of window rows.

    Mirrors `MotionWindowDataset.__getitem__` (motion_jepa/loader.py), but is
    driven by a caller-supplied row list (e.g. joined with eval labels and
    filtered to specific datasets/subjects) rather than a whole split.
    """

    def __init__(
        self,
        *,
        root: Path | str,
        rows: list[dict],
        window_size: int,
        segment_size: int,
        clip_value: float | None = 10.0,
    ) -> None:
        self.root = Path(root)
        self.rows = rows
        self.window_size = window_size
        self.segment_size = segment_size
        self.segment_count = window_size // segment_size
        self.clip_value = clip_value
        self._store: MotionZarrStore | None = None

        mean, std = load_normalization_stats(self.root)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-8)

    @property
    def store(self) -> MotionZarrStore:
        if self._store is None:
            self._store = MotionZarrStore(self.root)
        return self._store

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_store"] = None
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, t.Tensor]:
        row = self.rows[index]
        x = self.store.get_kinematics_window(
            suid=str(row["suid"]),
            start=int(row["start"]),
            end=int(row["end"]),
        )

        x = np.asarray(x, dtype=np.float32)
        x = center_root_channels(x)
        x = signed_log1p_tau(x)
        x = (x - self.mean) / self.std
        if self.clip_value is not None:
            x = np.clip(x, -self.clip_value, self.clip_value)

        valid_frames = x.shape[0]
        if valid_frames < self.window_size:
            x_padded = np.zeros((self.window_size, *x.shape[1:]), dtype=np.float32)
            x_padded[:valid_frames] = x
            x = x_padded

        valid_segments = min(valid_frames // self.segment_size, self.segment_count)

        return {
            "x": t.as_tensor(np.ascontiguousarray(x), dtype=t.float32),
            "valid_segments": t.tensor(valid_segments, dtype=t.long),
        }
