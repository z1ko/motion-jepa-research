
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import argparse

from pathlib import Path

from motion_jepa.loader import MotionWindowDataset
from motion_jepa.preprocess import CHANNELS, JOINTS

def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing arrays.zarr, windows.parquet, normalization_stats.npz",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Target channel",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Target sequence",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )

    return parser.parse_args()

def visualize(args: argparse.Namespace):

    dataset = MotionWindowDataset(root=args.root, normalize=True, clip_value=10.0)
    seq: np.ndarray = np.asarray(dataset[args.index], dtype=np.float32)

    fig, axes = plt.subplots(49, 1, figsize=(10, 80), sharex=True, sharey=True)
    for i in range(49):
        ax = axes[i]

        j = CHANNELS.index(args.channel)
        ax.plot(seq[:, i, j])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_ylabel(f"{JOINTS[i]}", rotation=0, labelpad=15)
        if i == 0:
            ax.set_title(f"{args.channel}")

        if i != 48:
            ax.tick_params(labelbottom=False)

    fig.tight_layout()
    fig.savefig(args.out_dir, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    visualize(cli())