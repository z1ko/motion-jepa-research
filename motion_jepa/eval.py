
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch as t
from torch.utils.data import DataLoader, Dataset

from motion_jepa.dataset import (
    MotionZarrStore,
    _path_of_samples_index,
    _path_of_windows_index,
    load_normalization_stats,
)
from motion_jepa.jepa import MotionJEPAModule
from motion_jepa.utils import center_root_channels, signed_log1p_tau

# ================================================================================================================
# EVAL LABEL PARSING
#
# The held-out eval datasets (SOMA, HumanEva, DanceDB) each encode a downstream
# label in the trial filename rather than as a metadata column.
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
# ENCODER LOADING
# ================================================================================================================

def load_encoder(
    *,
    checkpoint: Path | str,
    device: t.device,
) -> tuple[t.nn.Module, object]:
    """Load a trained teacher encoder, reconstructed from ITS OWN saved config.

    Deliberately does not accept a `config_path` to override the architecture:
    `config/experiment.yaml`/`config.py` can (and will) change over the life
    of the project, but a checkpoint's weights only fit the exact
    window_size/segment_size/groups/depth/etc. it was trained with. Passing
    an independently-loaded config here would silently reconstruct the wrong
    architecture the moment those two drift apart (as they did the moment
    `load_config` was fixed to actually read the yaml -- see git history).
    Lightning already stored the real config at train time via
    `save_hyperparameters`; using that is the only way this stays correct.
    """
    module = MotionJEPAModule.load_from_checkpoint(
        str(checkpoint),
        map_location=device,
        # Lightning checkpoints contain OmegaConf hyperparameters. Use only
        # checkpoints you created or otherwise trust.
        weights_only=False,
    )
    module.eval()
    module.to(device)
    return module.model.teacher_encoder, module.config

# ================================================================================================================
# WINDOW LOADING + EMBEDDING EXTRACTION
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


@t.inference_mode()
def compute_embeddings(
    *,
    encoder: t.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: t.device,
    segment_count: int,
    group_count: int,
) -> np.ndarray:
    """Run the encoder over every window and mean-pool its tokens into one embedding.

    Padded (short-trial) tokens are excluded from both attention, via a
    key_padding_mask, and pooling, via a masked mean -- a plain `.mean(dim=1)`
    would silently dilute embeddings for any window shorter than window_size.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    chunks: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        valid_segments = batch["valid_segments"].to(device, non_blocking=True)

        segment_valid = t.arange(segment_count, device=device).unsqueeze(0) < valid_segments.unsqueeze(1)
        token_valid = segment_valid.unsqueeze(-1).expand(-1, -1, group_count)
        key_padding_mask = ~token_valid.flatten(1, 2)

        tokens = encoder(x, idx=None, key_padding_mask=key_padding_mask)

        valid_flat = token_valid.flatten(1, 2).float().unsqueeze(-1)
        pooled = (tokens * valid_flat).sum(dim=1) / valid_flat.sum(dim=1).clamp(min=1.0)

        chunks.append(pooled.cpu().numpy())

    return np.concatenate(chunks, axis=0)
