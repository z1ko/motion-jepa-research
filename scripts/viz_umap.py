
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch as t

from motion_jepa.evaluation.data import WindowRowsDataset, load_window_table
from motion_jepa.evaluation.encoder import compute_embeddings, load_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 2D UMAP of Motion-JEPA window embeddings."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", default=Path("data/processed/motion"), type=Path)
    parser.add_argument("--split", default="eval", choices=["train", "val", "eval"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--color-by", default="dataset")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-windows", default=None, type=int)
    parser.add_argument("--eval-window-size", default=None, type=int)
    parser.add_argument("--eval-stride", default=None, type=int)
    parser.add_argument("--eval-min-valid-frames", default=None, type=int)
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

    encoder, config = load_encoder(checkpoint=args.checkpoint, device=device)

    rows = load_window_table(
        root=args.root,
        split=args.split,
        dataset=args.dataset,
        max_windows=args.max_windows,
        seed=args.seed,
        window_size=args.eval_window_size or config.data.window_size,
        stride=args.eval_stride or config.data.stride,
        min_valid_frames=args.eval_min_valid_frames or config.data.min_valid_frames,
    )
    if args.color_by not in rows.columns:
        raise ValueError(
            f"Unknown --color-by={args.color_by!r}. Available columns: {rows.columns}"
        )
    dataset = WindowRowsDataset(
        root=args.root,
        rows=rows.to_dicts(),
        window_size=config.data.window_size,
        segment_size=config.architecture.segment_size,
    )

    embeddings = compute_embeddings(
        encoder=encoder,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        segment_count=config.data.window_size // config.architecture.segment_size,
        group_count=len(config.training.groups),
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
