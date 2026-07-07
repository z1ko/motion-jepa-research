
import argparse
from pathlib import Path
from typing import Any

import matplotlib.pylab as plt
import polars as pl
import numpy as np

from motion_jepa.dataset import MotionZarrStore, _path_of_normalization_stats, _path_of_windows_index
from motion_jepa.preprocess import CHANNELS, JOINTS
from motion_jepa.utils import center_root_channels

def signed_log1p_tau(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    tau_idx = CHANNELS.index("tau")
    x[:, :, tau_idx] = np.sign(x[:, :, tau_idx]) * np.log1p(
        np.abs(x[:, :, tau_idx])
    )
    return x

def sample_windows(
    root: Path,
    *,
    n: int,
    split: str | None,
    seed: int,
) -> pl.DataFrame:
    windows = pl.read_parquet(_path_of_windows_index(root))

    if split is not None:
        windows = windows.filter(pl.col("split") == split)

    if windows.height == 0:
        raise ValueError(f"No windows found for split={split!r}")

    n = min(n, windows.height)

    return windows.sample(
        n=n,
        with_replacement=False,
        seed=seed,
    )

def load_normalization_stats(root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = _path_of_normalization_stats(root)

    if not path.exists():
        raise FileNotFoundError(f"Missing normalization stats: {path}")

    data = np.load(path)
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)

    expected = (len(JOINTS), len(CHANNELS))
    if mean.shape != expected:
        raise ValueError(f"Expected mean shape {expected}, got {mean.shape}")
    if std.shape != expected:
        raise ValueError(f"Expected std shape {expected}, got {std.shape}")

    return mean, std

def clipped_fraction(x: np.ndarray, clip_value: float = 10.0) -> dict[str, float]:
    out = {}

    for c, channel in enumerate(CHANNELS):
        values = x[:, :, :, c]
        out[channel] = float(np.mean(np.abs(values) >= clip_value))

    out["all"] = float(np.mean(np.abs(x) >= clip_value))
    return out

def load_normalized_batch(
    root: Path,
    *,
    n: int,
    split: str | None,
    seed: int,
) -> np.ndarray:
    store = MotionZarrStore(root)
    mean, std = load_normalization_stats(root)
    rows = sample_windows(
        root,
        n=n,
        split=split,
        seed=seed,
    )

    print(len(rows))
    xs: list[np.ndarray] = []

    for row in rows.iter_rows(named=True):
        x = store.get_kinematics_window(
            suid=str(row["suid"]),
            start=int(row["start"]),
            end=int(row["end"]),
        )

        x = center_root_channels(x)
        x = signed_log1p_tau(x)
        x = (x - mean) / std


        xs.append(x.astype(np.float32))

    result = np.stack(xs, axis=0)
    print(clipped_fraction(result, clip_value=8.0))

    # DO THIS EVERYTIME
    result = result.clip(-8.0, 8.0)

    return result  # [N, T, D, C]


def print_channel_report(x: np.ndarray) -> None:
    # x: [N, T, D, C]
    channel_mean = x.mean(axis=(0, 1, 2))
    channel_std = x.std(axis=(0, 1, 2))
    channel_abs_mean = np.abs(x).mean(axis=(0, 1, 2))
    channel_rms = np.sqrt(np.square(x).mean(axis=(0, 1, 2)))

    print("\nChannel report after normalization")
    print("----------------------------------")
    for i, channel in enumerate(CHANNELS):
        print(
            f"{channel:>4s}: "
            f"mean={channel_mean[i]: .4f} "
            f"std={channel_std[i]: .4f} "
            f"abs_mean={channel_abs_mean[i]: .4f} "
            f"rms={channel_rms[i]: .4f}"
        )


def print_suspicious_dimensions_report(
    x: np.ndarray,
    *,
    top_k: int = 20,
) -> None:
    # x: [N, T, D, C]
    mean = x.mean(axis=(0, 1))
    std = x.std(axis=(0, 1))
    abs_mean = np.abs(mean)

    rows: list[dict[str, Any]] = []

    for j, joint in enumerate(JOINTS):
        for c, channel in enumerate(CHANNELS):
            values = x[:, :, j, c].reshape(-1)

            rows.append(
                {
                    "joint": joint,
                    "channel": channel,
                    "mean": float(mean[j, c]),
                    "std": float(std[j, c]),
                    "abs_mean": float(abs_mean[j, c]),
                    "p0.1": float(np.percentile(values, 0.1)),
                    "p1": float(np.percentile(values, 1.0)),
                    "p99": float(np.percentile(values, 99.0)),
                    "p99.9": float(np.percentile(values, 99.9)),
                    "max_abs": float(np.max(np.abs(values))),
                }
            )

    by_abs_mean = sorted(rows, key=lambda r: r["abs_mean"], reverse=True)
    by_std_far = sorted(rows, key=lambda r: abs(r["std"] - 1.0), reverse=True)
    by_max_abs = sorted(rows, key=lambda r: r["max_abs"], reverse=True)

    print("\nLargest absolute normalized means")
    print("---------------------------------")
    for row in by_abs_mean[:top_k]:
        print(
            f"{row['joint']:28s} {row['channel']:>4s} "
            f"mean={row['mean']: .4f} std={row['std']: .4f} "
            f"p1={row['p1']: .2f} p99={row['p99']: .2f}"
        )

    print("\nStd furthest from 1")
    print("-------------------")
    for row in by_std_far[:top_k]:
        print(
            f"{row['joint']:28s} {row['channel']:>4s} "
            f"mean={row['mean']: .4f} std={row['std']: .4f} "
            f"p1={row['p1']: .2f} p99={row['p99']: .2f}"
        )

    print("\nLargest absolute normalized values")
    print("----------------------------------")
    for row in by_max_abs[:top_k]:
        print(
            f"{row['joint']:28s} {row['channel']:>4s} "
            f"max_abs={row['max_abs']: .2f} "
            f"p0.1={row['p0.1']: .2f} p99.9={row['p99.9']: .2f}"
        )

def save_heatmap(
    values: np.ndarray,
    *,
    title: str,
    path: Path,
    colorbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 14))

    im = ax.imshow(values, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Joint")

    ax.set_xticks(np.arange(len(CHANNELS)))
    ax.set_xticklabels(CHANNELS)

    ax.set_yticks(np.arange(len(JOINTS)))
    ax.set_yticklabels(JOINTS)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_distribution_plots(x: np.ndarray, out_dir: Path) -> None:
    flat = x.reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(flat, bins=300)
    ax.set_title("All normalized kinematics values")
    ax.set_xlabel("normalized value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "hist_all_normalized_values.png", dpi=200)
    plt.close(fig)

    clipped = flat[np.abs(flat) <= 8.0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(clipped, bins=300)
    ax.set_title("All normalized kinematics values, clipped to [-8, 8]")
    ax.set_xlabel("normalized value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "hist_all_normalized_values_clipped.png", dpi=200)
    plt.close(fig)

    for c, channel in enumerate(CHANNELS):
        values = x[:, :, :, c].reshape(-1)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(values[np.abs(values) <= 8.0], bins=300)
        ax.set_title(f"Normalized {channel} distribution, clipped to [-8, 8]")
        ax.set_xlabel("normalized value")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{channel}_normalized_clipped.png", dpi=200)
        plt.close(fig)


def save_channel_boxplot(x: np.ndarray, out_dir: Path) -> None:
    values = [
        x[:, :, :, c].reshape(-1)
        for c in range(len(CHANNELS))
    ]

    # Subsample for faster plotting if needed.
    rng = np.random.default_rng(13)
    values_sampled = []
    for v in values:
        if v.size > 200_000:
            idx = rng.choice(v.size, size=200_000, replace=False)
            v = v[idx]
        values_sampled.append(v)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(values_sampled, label=CHANNELS, showfliers=False)
    ax.set_title("Normalized channel distributions, outliers hidden")
    ax.set_ylabel("normalized value")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_channels_normalized.png", dpi=200)
    plt.close(fig)

def create_visual_report(
    x: np.ndarray,
    *,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mean/std over sampled normalized windows.
    mean = x.mean(axis=(0, 1))
    std = x.std(axis=(0, 1))
    abs_mean = np.abs(mean)

    save_heatmap(
        mean,
        title="Sampled normalized kinematics: mean",
        path=out_dir / "heatmap_sampled_normalized_mean.png",
        colorbar_label="mean",
    )

    save_heatmap(
        std,
        title="Sampled normalized kinematics: std",
        path=out_dir / "heatmap_sampled_normalized_std.png",
        colorbar_label="std",
    )

    save_heatmap(
        abs_mean,
        title="Sampled normalized kinematics: abs(mean)",
        path=out_dir / "heatmap_sampled_normalized_abs_mean.png",
        colorbar_label="abs(mean)",
    )

    save_distribution_plots(x, out_dir)
    save_channel_boxplot(x, out_dir)
    

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing arrays.zarr, windows.parquet, normalization_stats.npz",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=512,
        help="Number of windows to sample",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to sample from. Use 'none' to sample all splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    split = None if args.split.lower() == "none" else args.split
    out_dir = args.out_dir or (args.root / "normalization_check")

    x = load_normalized_batch(
        args.root,
        n=args.n,
        split=split,
        seed=args.seed,
    )

    print(f"Loaded normalized batch shape: {x.shape}")
    print(f"Output directory: {out_dir}")

    print_channel_report(x)
    print_suspicious_dimensions_report(x)

    create_visual_report(x, out_dir=out_dir)
    #save_numeric_report(x, out_dir=out_dir)

    print("\nSaved report files:")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path}")

if __name__ == "__main__":
    main()