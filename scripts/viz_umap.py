
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch as t
from torch.utils.data import DataLoader, Dataset

from motion_jepa.config import load_config
from motion_jepa.dataset import (
    MotionZarrStore,
    _path_of_samples_index,
    _path_of_windows_index,
    load_normalization_stats,
)
from motion_jepa.jepa import MotionJEPAModule
from motion_jepa.utils import center_root_channels, signed_log1p_tau


class WindowRowsDataset(Dataset):
    def __init__(
        self,
        *,
        root: Path,
        rows: list[dict],
        clip_value: float | None = 10.0,
    ) -> None:
        self.root = root
        self.rows = rows
        self.clip_value = clip_value
        self.store = MotionZarrStore(root)

        mean, std = load_normalization_stats(root)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-8)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> t.Tensor:
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
        return t.as_tensor(np.ascontiguousarray(x), dtype=t.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 2D UMAP of Motion-JEPA window embeddings."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", default=Path("data/processed/motion"), type=Path)
    parser.add_argument("--config", default=Path("config/experiment.yaml"), type=Path)
    parser.add_argument("--split", default="eval", choices=["train", "val", "eval"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--color-by", default="dataset")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-windows", default=None, type=int)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n-neighbors", default=15, type=int)
    parser.add_argument("--min-dist", default=0.1, type=float)
    return parser.parse_args()


def resolve_device(device: str) -> t.device:
    if device == "auto":
        return t.device("cuda" if t.cuda.is_available() else "cpu")
    if device == "cuda" and not t.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return t.device(device)


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
    root: Path,
    split: str,
    dataset: str | None,
    max_windows: int | None,
    seed: int,
) -> pl.DataFrame:
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


def load_encoder(
    *,
    checkpoint: Path,
    config_path: Path,
    device: t.device,
) -> t.nn.Module:
    config = load_config(config_path)
    module = MotionJEPAModule.load_from_checkpoint(
        str(checkpoint),
        config=config,
        map_location=device,
        # Lightning checkpoints contain OmegaConf hyperparameters. Use only
        # checkpoints you created or otherwise trust.
        weights_only=False,
    )
    module.eval()
    module.to(device)
    return module.model.teacher_encoder


@t.inference_mode()
def compute_embeddings(
    *,
    encoder: t.nn.Module,
    dataset: WindowRowsDataset,
    batch_size: int,
    device: t.device,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    chunks: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        tokens = encoder(batch)
        pooled = tokens.mean(dim=1)
        chunks.append(pooled.cpu().numpy())

    return np.concatenate(chunks, axis=0)


def run_umap(
    *,
    embeddings: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    import umap

    if embeddings.shape[0] < 3:
        raise ValueError("Need at least 3 windows for UMAP.")

    n_neighbors = min(n_neighbors, embeddings.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    return reducer.fit_transform(embeddings)


def is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }


def plot_umap(
    *,
    coords: np.ndarray,
    rows: pl.DataFrame,
    color_by: str,
    split: str,
    dataset: str | None,
    out: Path,
) -> None:
    if color_by not in rows.columns:
        raise ValueError(
            f"Unknown --color-by={color_by!r}. Available columns: {rows.columns}"
        )

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    series = rows.get_column(color_by)

    if is_numeric_dtype(series.dtype):
        values = series.cast(pl.Float64).to_numpy()
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            s=8,
            alpha=0.75,
            cmap="viridis",
            linewidths=0,
        )
        fig.colorbar(scatter, ax=ax, label=color_by)
    else:
        labels = series.fill_null("<null>").cast(pl.Utf8).to_list()
        unique = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique)}
        codes = np.asarray([label_to_idx[label] for label in labels])

        cmap = plt.get_cmap("tab20", max(len(unique), 1))
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=codes,
            s=8,
            alpha=0.75,
            cmap=cmap,
            linewidths=0,
        )

        if len(unique) <= 20:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=label,
                    markerfacecolor=cmap(label_to_idx[label]),
                    markersize=6,
                )
                for label in unique
            ]
            ax.legend(
                handles=handles,
                title=color_by,
                loc="best",
                frameon=False,
                fontsize="small",
            )
        elif len(unique) <= 40:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(color_by)

    dataset_text = "" if dataset is None else f" dataset={dataset}"
    ax.set_title(f"Motion-JEPA UMAP split={split}{dataset_text} color={color_by}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def print_counts(rows: pl.DataFrame, color_by: str) -> None:
    counts = (
        rows.group_by(color_by)
        .len()
        .sort("len", descending=True)
    )
    print(counts)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out = args.out
    if out is None:
        dataset_name = "" if args.dataset is None else f"_{args.dataset}"
        out = Path("outputs") / f"umap_{args.split}{dataset_name}_{args.color_by}.png"

    rows = load_window_table(
        root=args.root,
        split=args.split,
        dataset=args.dataset,
        max_windows=args.max_windows,
        seed=args.seed,
    )
    if args.color_by not in rows.columns:
        raise ValueError(
            f"Unknown --color-by={args.color_by!r}. Available columns: {rows.columns}"
        )

    dataset = WindowRowsDataset(root=args.root, rows=rows.to_dicts())
    encoder = load_encoder(
        checkpoint=args.checkpoint,
        config_path=args.config,
        device=device,
    )

    embeddings = compute_embeddings(
        encoder=encoder,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
    )
    coords = run_umap(
        embeddings=embeddings,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    plot_umap(
        coords=coords,
        rows=rows,
        color_by=args.color_by,
        split=args.split,
        dataset=args.dataset,
        out=out,
    )

    print(f"windows: {rows.height}")
    print(f"embeddings: {embeddings.shape}")
    print_counts(rows, args.color_by)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
