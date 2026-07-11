
import argparse
from pathlib import Path

import numpy as np
import torch as t
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from omegaconf import OmegaConf

from motion_jepa.masking import (
    mask_mixed,
    mask_random,
    mask_spatial_blocks,
    mask_temporal_blocks,
    mask_tube_blocks,
)

# Cell values: 0 context, 1 target, 2 padding, 3 BUG (padding picked as target
# -- should never happen, blocks are clipped to min_valid before construction).
_COLORS = ["#dddddd", "#d62728", "#444444", "#ff00ff"]
_LABELS = ["context", "target", "padding", "BUG: padding as target"]
_PADDING_COLOR = "#444444"  # matches _COLORS[2], reused for the frequency panel's NaN cells


def build_token_valid(
    *, valid_fracs: list[float], batch_size: int, segment_count: int, group_count: int,
) -> t.Tensor | None:
    """Synthetic per-sample padding mask: `valid_fracs` (cycled to fill
    batch_size) gives EACH sample its own valid-segment cutoff, simulating a
    real training batch's mix of trial lengths. None (no padding demo) only
    if every requested fraction is >= 1.0.
    """
    if all(f >= 1.0 for f in valid_fracs):
        return None

    fracs = [valid_fracs[i % len(valid_fracs)] for i in range(batch_size)]
    valid_segments = t.tensor([max(1, round(segment_count * f)) for f in fracs])  # (B,)
    segment_valid = t.arange(segment_count).unsqueeze(0) < valid_segments.unsqueeze(1)  # (B, S)
    return segment_valid.unsqueeze(-1).expand(batch_size, segment_count, group_count)  # (B, S, G)


def draw_many(
    fn, n_draws: int, *args, token_valid_ref: t.Tensor | None, **kwargs,
) -> tuple[list[t.Tensor], list[dict]]:
    """Call fn n_draws times, each treating its OWN single sample (cycled
    from token_valid_ref) as if it were the whole batch for that step -- so
    a fully-valid draw's min_valid is its own full segment_count, not
    dragged down by some other draw's shorter one. Real training's "one
    shared mask per batch" applies WITHIN a step's batch, not across these
    independent per-draw calls (each represents a different step). Returns
    per-draw flat target-index tensors, deliberately NOT stacked into one
    tensor -- mask_mixed's draws can pick strategies with different target
    counts (e.g. tube vs. temporal), so the result is ragged in general.
    """
    ref_batch = token_valid_ref.shape[0] if token_valid_ref is not None else 1
    targets_list, metas = [], []
    for i in range(n_draws):
        tv = token_valid_ref[i % ref_batch : i % ref_batch + 1] if token_valid_ref is not None else None
        mask, meta = fn(1, *args, token_valid=tv, return_meta=True, **kwargs)
        targets_list.append(mask.targets[0])
        metas.append(meta[0])  # return_meta now returns a list of B per-sample dicts
    return targets_list, metas


def grid_from_targets(
    targets_list: list[t.Tensor], *, segment_count: int, group_count: int, token_valid: t.Tensor | None,
) -> np.ndarray:
    """Per-draw flat target indices (ragged-safe) -> (n_draws, segment_count,
    group_count) int grid, see _COLORS/_LABELS."""
    n_draws = len(targets_list)
    numel = segment_count * group_count

    is_target = np.zeros((n_draws, numel), dtype=bool)
    for i, targets in enumerate(targets_list):
        is_target[i, targets.cpu().numpy()] = True

    if token_valid is not None:
        is_padding = (~token_valid.cpu().numpy()).reshape(n_draws, numel)
    else:
        is_padding = np.zeros_like(is_target)

    grid = np.zeros((n_draws, numel), dtype=np.int64)
    grid[is_padding & ~is_target] = 2
    grid[is_target & ~is_padding] = 1
    grid[is_target & is_padding] = 3  # invariant violation, should never appear

    return grid.reshape(n_draws, segment_count, group_count)


def target_frequency(
    targets_list: list[t.Tensor], *, segment_count: int, group_count: int, token_valid: t.Tensor | None,
) -> np.ndarray:
    """Fraction of ELIGIBLE draws each cell was a target -- eligible meaning
    valid for that draw, since with varied per-sample valid_fracs a cell can
    be padding for some draws and real for others. Averaged only over draws
    where the cell was valid.

    A cell that was padding in every draw is NaN (rather than 0.0) so it
    renders as a fixed color instead of looking like a "cold" valid cell.
    """
    grid = grid_from_targets(targets_list, segment_count=segment_count, group_count=group_count, token_valid=token_valid)
    is_target = (grid == 1)  # (B, S, G); already excludes padding, see grid_from_targets

    if token_valid is None:
        return is_target.mean(axis=0).astype(np.float64)

    valid = token_valid.cpu().numpy()  # (B, S, G), can vary per draw
    valid_count = valid.sum(axis=0)  # (S, G) -- how many draws had this cell eligible
    target_count = is_target.sum(axis=0)  # (S, G)
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = target_count / valid_count
    return np.where(valid_count == 0, np.nan, freq)


def draw_block_outline(
    ax, *, kind: str | None, meta: dict | None, segment_count: int, group_count: int,
) -> None:
    """Outline the block this draw actually masked."""
    if kind is None or kind == "random" or not meta:
        return

    edge = dict(edgecolor="blue", fill=False, lw=1.3)
    if kind in ("temporal_blocks", "tube_blocks"):
        start = meta["start"]
        block = meta["block_segments"]
    if kind in ("spatial_blocks", "tube_blocks"):
        groups = meta["groups"].tolist()

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
    parser.add_argument("--n-samples", type=int, default=3, help="Independent draws shown per strategy.")
    parser.add_argument(
        "--freq-draws", type=int, default=200,
        help="Independent draws averaged into the target-frequency heatmap column.",
    )
    parser.add_argument(
        "--valid-fracs", type=float, nargs="+", default=[1.0, 0.7, 0.5],
        help=(
            "Per-draw valid-segment fractions, cycled to fill n_samples/freq_draws -- "
            "simulates a mix of trial lengths across steps. All >= 1.0 disables padding."
        ),
    )
    parser.add_argument("--temporal-block-segments", type=int, default=3)
    parser.add_argument("--spatial-block-groups", type=int, default=2, help="Number of whole kinematic chains (of 5).")
    parser.add_argument("--tube-block-segments", type=int, default=6)
    parser.add_argument("--tube-block-groups", type=int, default=2, help="Number of whole kinematic chains (of 5).")
    parser.add_argument("--random-n-targets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for reproducible draws. Default: random.")
    parser.add_argument("--out-dir", type=Path, default=Path("statistics/masks"))
    args = parser.parse_args()

    if args.seed is not None:
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
        valid_fracs=args.valid_fracs, batch_size=args.n_samples, segment_count=segment_count, group_count=group_count,
    )
    token_valid_freq = build_token_valid(
        valid_fracs=args.valid_fracs, batch_size=args.freq_draws, segment_count=segment_count, group_count=group_count,
    )

    if token_valid_ex is not None:
        valid_segments_ex = token_valid_ex[:, :, 0].sum(dim=1).tolist()
    else:
        valid_segments_ex = [segment_count] * args.n_samples

    # Each row: independent example draws (MaskIndices + per-draw meta for
    # the outline/title) plus a many-draw target-frequency grid.
    rows = []

    strategies = (
        ("random", mask_random, {"n_targets": args.random_n_targets}),
        ("temporal_blocks", mask_temporal_blocks, {"block_segments": args.temporal_block_segments}),
        ("spatial_blocks", mask_spatial_blocks, {"block_groups": args.spatial_block_groups}),
        ("tube_blocks", mask_tube_blocks, {"block_segments": args.tube_block_segments, "block_groups": args.tube_block_groups}),
    )
    for name, fn, kwargs in strategies:
        ex, ex_metas = draw_many(fn, args.n_samples, segment_count, group_count, device, *kwargs.values(), token_valid_ref=token_valid_ex)
        freq_targets, _ = draw_many(fn, args.freq_draws, segment_count, group_count, device, *kwargs.values(), token_valid_ref=token_valid_ex)
        rows.append({
            "name": name, "targets": ex, "metas": ex_metas, "strategies": [name] * args.n_samples,
            "freq": target_frequency(freq_targets, segment_count=segment_count, group_count=group_count, token_valid=token_valid_freq),
        })

    mixed_kwargs = dict(
        temporal_block_segments=args.temporal_block_segments,
        spatial_block_groups=args.spatial_block_groups,
        tube_block_segments=args.tube_block_segments,
        tube_block_groups=args.tube_block_groups,
        random_n_targets=args.random_n_targets,
    )
    ex, ex_metas = draw_many(mask_mixed, args.n_samples, segment_count, group_count, device, token_valid_ref=token_valid_ex, **mixed_kwargs)
    freq_targets, _ = draw_many(mask_mixed, args.freq_draws, segment_count, group_count, device, token_valid_ref=token_valid_ex, **mixed_kwargs)
    rows.append({
        "name": "mixed", "targets": ex, "metas": ex_metas, "strategies": [m["strategy"] for m in ex_metas],
        "freq": target_frequency(freq_targets, segment_count=segment_count, group_count=group_count, token_valid=token_valid_freq),
    })

    n_cols = args.n_samples + 1  # + frequency column
    disc_cmap = ListedColormap(_COLORS)
    freq_cmap = plt.get_cmap("viridis").copy()
    freq_cmap.set_bad(_PADDING_COLOR)

    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.6 * n_cols, 2.1 * len(rows)), squeeze=False)

    freq_im = None
    for row_idx, row in enumerate(rows):
        grid = grid_from_targets(row["targets"], segment_count=segment_count, group_count=group_count, token_valid=token_valid_ex)

        for col in range(args.n_samples):
            ax = axes[row_idx][col]
            ax.imshow(grid[col].T, cmap=disc_cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")

            kind = row["strategies"][col]
            title = f"draw {col}\nvalid={valid_segments_ex[col]}/{segment_count}"
            if row["name"] == "mixed":
                title = f"draw {col}\n({kind})"
            if row_idx == 0 or row["name"] == "mixed":
                ax.set_title(title, fontsize=7 if row["name"] == "mixed" else 8)

            draw_block_outline(
                ax, kind=kind, meta=row["metas"][col],
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

    print(f"segment_count={segment_count} group_count={group_count}")
    print(f"example draws valid_segments per column: {valid_segments_ex}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
