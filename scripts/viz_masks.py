
import argparse
from pathlib import Path

import numpy as np
import torch as t
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from omegaconf import OmegaConf

from motion_jepa.masking import (
    MaskIndices,
    mask_mixed,
    mask_random,
    mask_spatial_blocks,
    mask_temporal_blocks,
    mask_tube_blocks,
)

# Cell values: 0 context, 1 target, 2 padding, 3 BUG (padding picked as target
# -- should never happen, _rank_to_mask masks padding to -inf before ranking).
_COLORS = ["#dddddd", "#d62728", "#444444", "#ff00ff"]
_LABELS = ["context", "target", "padding", "BUG: padding as target"]
_PADDING_COLOR = "#444444"  # matches _COLORS[2], reused for the frequency panel's NaN cells


def build_token_valid(
    *, batch_size: int, segment_count: int, group_count: int, valid_frac: float,
) -> t.Tensor | None:
    """Synthetic padding mask: the last (1 - valid_frac) fraction of segments
    are marked invalid, same as a short trial padded to window_size. None
    (no padding demo) if valid_frac >= 1.0.
    """
    if valid_frac >= 1.0:
        return None

    valid_segments = max(1, round(segment_count * valid_frac))
    segment_valid = t.arange(segment_count) < valid_segments  # (S,)
    token_valid = segment_valid.unsqueeze(-1).expand(segment_count, group_count)  # (S, G)
    return token_valid.unsqueeze(0).expand(batch_size, -1, -1)  # (B, S, G)


def grid_from_mask(
    mask: MaskIndices, *, segment_count: int, group_count: int, token_valid: t.Tensor | None,
) -> np.ndarray:
    """MaskIndices -> (B, segment_count, group_count) int grid, see _COLORS/_LABELS."""
    batch_size = mask.context.shape[0]
    numel = segment_count * group_count

    is_target = np.zeros((batch_size, numel), dtype=bool)
    np.put_along_axis(is_target, mask.targets.cpu().numpy(), True, axis=1)

    if token_valid is not None:
        is_padding = (~token_valid.cpu().numpy()).reshape(batch_size, numel)
    else:
        is_padding = np.zeros_like(is_target)

    grid = np.zeros((batch_size, numel), dtype=np.int64)
    grid[is_padding & ~is_target] = 2
    grid[is_target & ~is_padding] = 1
    grid[is_target & is_padding] = 3  # invariant violation, should never appear

    return grid.reshape(batch_size, segment_count, group_count)


def target_frequency(
    mask: MaskIndices, *, segment_count: int, group_count: int, token_valid: t.Tensor | None,
) -> np.ndarray:
    """Fraction of draws (over the mask's batch dim) each cell was a target.

    Averaging many draws reveals a strategy's true bias even when any single
    draw looks noisy (e.g. mask_tube_blocks' block is small relative to
    n_targets, so one draw mostly looks like random noise -- the average
    over many draws doesn't). Padding cells are NaN (never targets by
    construction; NaN'd rather than left at their true 0.0 so they render
    as a fixed color instead of looking like "cold" valid cells).
    """
    grid = grid_from_mask(mask, segment_count=segment_count, group_count=group_count, token_valid=token_valid)
    freq = (grid == 1).mean(axis=0).astype(np.float64)  # (S, G)

    if token_valid is not None:
        invalid = ~token_valid[0].cpu().numpy()  # every sample shares the same synthetic padding pattern
        freq = np.where(invalid, np.nan, freq)

    return freq


def draw_block_outline(
    ax, *, kind: str | None, sample_idx: int, meta: dict | None, segment_count: int, group_count: int,
) -> None:
    """Outline the block/groups a strategy actually biased towards for this
    sample, so it's clear what was *intended* even when noise-ranking picked
    targets outside it too (see mask_tube_blocks in particular).
    """
    if kind is None or kind == "random" or meta is None:
        return

    edge = dict(edgecolor="blue", fill=False, lw=1.3)
    if kind in ("temporal_blocks", "tube_blocks"):
        start = int(meta["starts"][sample_idx])
        block = meta["block"]
    if kind in ("spatial_blocks", "tube_blocks"):
        groups = meta["groups"][sample_idx].tolist()

    if kind == "temporal_blocks":
        ax.add_patch(Rectangle((start - 0.5, -0.5), block, group_count, **edge))
    elif kind == "spatial_blocks":
        for g in groups:
            ax.add_patch(Rectangle((-0.5, g - 0.5), segment_count, 1, **edge))
    elif kind == "tube_blocks":
        for g in groups:
            ax.add_patch(Rectangle((start - 0.5, g - 0.5), block, 1, **edge))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize motion_jepa.masking's masking strategies as context/target/padding grids."
    )
    parser.add_argument("--config", type=Path, default=Path("config/experiment.yaml"))
    parser.add_argument("--n-samples", type=int, default=3, help="Random draws shown per strategy.")
    parser.add_argument(
        "--freq-draws", type=int, default=200,
        help="Draws averaged into the target-frequency heatmap column.",
    )
    parser.add_argument(
        "--valid-frac", type=float, default=0.7,
        help="Fraction of segments marked valid (rest simulate short-trial padding). 1.0 disables.",
    )
    parser.add_argument("--targets-p", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("statistics/masks"))
    args = parser.parse_args()

    t.manual_seed(args.seed)

    # Not motion_jepa.config.load_config: it reads sys.argv as OmegaConf
    # dotlist overrides, which would collide with this script's own argparse
    # flags. This script only needs the yaml's own values, no CLI overrides.
    config = OmegaConf.load(args.config)
    segment_count = config.data.window_size // config.architecture.segment_size
    group_names = list(config.training.groups.keys())
    group_count = len(group_names)
    device = t.device("cpu")

    token_valid_ex = build_token_valid(
        batch_size=args.n_samples, segment_count=segment_count, group_count=group_count, valid_frac=args.valid_frac,
    )
    token_valid_freq = build_token_valid(
        batch_size=args.freq_draws, segment_count=segment_count, group_count=group_count, valid_frac=args.valid_frac,
    )
    if token_valid_ex is not None:
        # Same formula as masking._num_targets_from_valid: ratio against the
        # valid count, not the full grid. Every sample shares the same
        # synthetic valid_frac here, so batch-min == any sample's count,
        # and is independent of which batch_size (n_samples/freq_draws) built it.
        valid_count = int(token_valid_ex[0].sum().item())
        n_targets = min(max(1, round(args.targets_p * valid_count)), max(valid_count - 1, 1))
    else:
        n_targets = max(1, round(args.targets_p * segment_count * group_count))

    # Each row: example draws (MaskIndices + optional per-sample meta/labels
    # for the outline/title) plus a many-draw target-frequency grid.
    rows = []

    ex, freq_mask = (
        mask_random(args.n_samples, segment_count, group_count, device, n_targets, token_valid_ex),
        mask_random(args.freq_draws, segment_count, group_count, device, n_targets, token_valid_freq),
    )
    rows.append({"name": "random", "mask": ex, "meta": None, "labels": None, "freq": target_frequency(
        freq_mask, segment_count=segment_count, group_count=group_count, token_valid=token_valid_freq)})

    for name, fn in (
        ("temporal_blocks", mask_temporal_blocks),
        ("spatial_blocks", mask_spatial_blocks),
        ("tube_blocks", mask_tube_blocks),
    ):
        ex, ex_meta = fn(args.n_samples, segment_count, group_count, device, n_targets, token_valid_ex, return_meta=True)
        freq_mask = fn(args.freq_draws, segment_count, group_count, device, n_targets, token_valid_freq)
        rows.append({"name": name, "mask": ex, "meta": ex_meta, "labels": None, "freq": target_frequency(
            freq_mask, segment_count=segment_count, group_count=group_count, token_valid=token_valid_freq)})

    ex, ex_meta = mask_mixed(args.n_samples, segment_count, group_count, device, args.targets_p, token_valid_ex, return_meta=True)
    freq_mask = mask_mixed(args.freq_draws, segment_count, group_count, device, args.targets_p, token_valid_freq)
    rows.append({"name": "mixed", "mask": ex, "meta": ex_meta, "labels": ex_meta["strategy"], "freq": target_frequency(
        freq_mask, segment_count=segment_count, group_count=group_count, token_valid=token_valid_freq)})

    n_cols = args.n_samples + 1  # + frequency column
    disc_cmap = ListedColormap(_COLORS)
    freq_cmap = plt.get_cmap("viridis").copy()
    freq_cmap.set_bad(_PADDING_COLOR)

    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.6 * n_cols, 2.1 * len(rows)), squeeze=False)

    freq_im = None
    for row_idx, row in enumerate(rows):
        grid = grid_from_mask(row["mask"], segment_count=segment_count, group_count=group_count, token_valid=token_valid_ex)

        for col in range(args.n_samples):
            ax = axes[row_idx][col]
            ax.imshow(grid[col].T, cmap=disc_cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")

            if row["labels"] is not None:
                kind = row["labels"][col]
                meta_for_draw = row["meta"].get(kind) if kind != "random" else None
                ax.set_title(f"draw {col}\n({kind})", fontsize=7)
            else:
                kind = row["name"]
                meta_for_draw = row["meta"]
                if row_idx == 0:
                    ax.set_title(f"draw {col}", fontsize=9)

            draw_block_outline(
                ax, kind=kind, sample_idx=col, meta=meta_for_draw,
                segment_count=segment_count, group_count=group_count,
            )

            if col == 0:
                ax.set_ylabel(row["name"], fontsize=9)
                ax.set_yticks(range(group_count))
                ax.set_yticklabels(group_names, fontsize=5)
            else:
                ax.set_yticks([])
            if row_idx == len(rows) - 1:
                ax.set_xlabel("segment (time)", fontsize=8)

        freq_ax = axes[row_idx][args.n_samples]
        freq_im = freq_ax.imshow(row["freq"].T, cmap=freq_cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        freq_ax.set_yticks([])
        if row_idx == 0:
            freq_ax.set_title(f"freq (n={args.freq_draws})", fontsize=9)
        if row_idx == len(rows) - 1:
            freq_ax.set_xlabel("segment (time)", fontsize=8)

    fig.colorbar(freq_im, ax=axes[:, args.n_samples].tolist(), fraction=0.03, pad=0.02, label="target frequency")
    fig.legend(
        handles=[Patch(color=c, label=l) for c, l in zip(_COLORS, _LABELS)],
        loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.45, 0.0),
    )
    fig.tight_layout(rect=(0, 0.04, 0.93, 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "masks.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"segment_count={segment_count} group_count={group_count} n_targets={n_targets}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
