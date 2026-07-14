"""Loading and labeling windows from the held-out eval datasets.

The held-out eval datasets (SOMA, HumanEva, DanceDB) each encode a downstream
label in the trial filename rather than as a metadata column, so labels are
parsed out of the filename and joined onto the window table here.
"""

from __future__ import annotations

import pickle
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
from motion_jepa.evaluation.babel import attach_babel_labels
from motion_jepa.utils import prepare_window

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


def load_care_pd_labels(path: Path | str) -> dict[str, int]:
    """CARE-PD's flat `filename;score` MDS-UPDRS-gait severity table.

    Keyed by the exact original CSV filename stem -- ingestion
    (scripts/prep_care_pd.py) keeps `trial` as the raw filename verbatim
    (see preprocess.py's load_sample_from_csv), so the join is an exact
    match on `trial` minus its extension, no parsing needed.
    """
    table = pl.read_csv(path, separator=";")
    filename_col, score_col = table.columns
    return dict(zip(table[filename_col].to_list(), table[score_col].to_list()))


def parse_care_pd_label(trial: str, care_pd_labels: dict[str, int]) -> tuple[str | None, str]:
    """Score 3 (severe) is dropped here (returned as unlabeled) -- only 10 of
    871 samples, too few to fold-evaluate meaningfully; matches the CARE-PD
    paper's own F1_0-2 metric. Filtering here (not a separate pass) lets the
    existing filter_labeled_rows (drops null eval_label) handle it for free.
    """
    score = care_pd_labels.get(Path(trial).stem)
    if score is None or score == 3:
        return None, "unknown"
    return str(score), "updrs_gait"


def add_eval_labels(rows: pl.DataFrame, *, care_pd_labels: dict[str, int] | None = None) -> pl.DataFrame:
    if "dataset" not in rows.columns or "trial" not in rows.columns:
        raise ValueError("Rows must contain 'dataset' and 'trial' to parse eval labels.")

    actions: list[str | None] = []
    emotions: list[str | None] = []
    labels: list[str | None] = []
    kinds: list[str] = []

    for row in rows.select(["dataset", "trial"]).iter_rows(named=True):
        dataset = str(row["dataset"])
        trial = str(row["trial"])

        if dataset.startswith("CARE-PD-"):
            if care_pd_labels is None:
                raise ValueError(f"dataset={dataset!r} needs care_pd_labels (see load_care_pd_labels).")
            action, emotion = None, None
            label, kind = parse_care_pd_label(trial, care_pd_labels)
        else:
            action, emotion, label, kind = parse_eval_label(dataset=dataset, trial=trial)

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
    care_pd_labels: dict[str, int] | None = None,
    babel_raw_root: Path | str | None = None,
) -> pl.DataFrame:
    """Join windows.parquet with samples.parquet and attach eval labels.

    `babel_raw_root`: if set, attach BABEL frame-level action labels (read
    from raw AMASS CSVs under this root, see evaluation.babel) instead of
    filename/CARE-PD label parsing -- for datasets like ACCAD/MoSh/SFU that
    have no filename-encoded label of their own. Mutually exclusive with
    `care_pd_labels` in practice (one eval run targets one label source).
    """
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

    if babel_raw_root is not None:
        rows = attach_babel_labels(rows, raw_root=babel_raw_root)
    else:
        rows = add_eval_labels(rows, care_pd_labels=care_pd_labels)

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
# FOLD SOURCES
# ================================================================================================================

def load_fixed_folds(fold_path: Path | str, subject_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load a CARE-PD fold definition (`{fold_id: {"train": [...], "eval":
    [...]}}`, e.g. `<cohort>_43fold_participants.pkl` for true per-subject
    LOSO, or `<cohort>_fixed.pkl` for the paper's single held-out split) and
    map its subject-id lists to row indices against `subject_ids` --
    reproduces the CARE-PD paper's own fold assignment exactly, instead of
    recomputing our own via LeaveOneGroupOut (see run_linear_probe's `folds`).

    A fold whose train/eval subjects don't overlap `subject_ids` at all
    (e.g. that subject's trials were too short to produce a valid window
    here, or got dropped by min_label_subjects) has nothing to evaluate and
    is skipped -- our ingested subject set can be a strict subset of the
    paper's, since it's driven by what's actually usable end to end.
    """
    with open(fold_path, "rb") as f:
        raw_folds = pickle.load(f)

    folds = []
    for fold in raw_folds.values():
        train_idx = np.flatnonzero(np.isin(subject_ids, fold["train"]))
        eval_idx = np.flatnonzero(np.isin(subject_ids, fold["eval"]))
        if len(train_idx) == 0 or len(eval_idx) == 0:
            continue
        folds.append((train_idx, eval_idx))
    return folds

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

        return prepare_window(
            x,
            window_size=self.window_size,
            segment_size=self.segment_size,
            mean=self.mean,
            std=self.std,
            clip_value=self.clip_value,
        )
