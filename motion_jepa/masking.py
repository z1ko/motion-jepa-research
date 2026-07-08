
import random
from dataclasses import dataclass

import torch as t

@dataclass(frozen=True)
class MaskIndices:
    context: t.Tensor
    targets: t.Tensor


def _min_valid_segments(token_valid: t.Tensor | None, segment_count: int) -> int:
    """Batch's shortest valid-segment extent, for sizing/positioning a
    region so it can never geometrically spill into padding for any sample.
    """
    if token_valid is None:
        return segment_count
    return int(token_valid[:, :, 0].sum(dim=1).min().item())  # any group column: expand() made them identical


def _sample_block_targets(
    segment_count: int,
    group_count: int,
    device: t.device,
    token_valid: t.Tensor | None,
    *,
    block_segments: int,
    block_groups: int,
) -> tuple[t.Tensor, dict]:
    """One randomly-positioned block's flat token indices, shared by the
    whole batch. Clipped to the batch's min_valid segments so it's
    guaranteed valid for every sample -- no per-sample check needed.
    """
    min_valid = _min_valid_segments(token_valid, segment_count)
    block_segments = min(block_segments, min_valid)
    block_groups = min(block_groups, group_count)

    start = int(t.randint(0, min_valid - block_segments + 1, (1,)).item())
    groups = t.randperm(group_count, device=device)[:block_groups]
    seg_idx = t.arange(start, start + block_segments, device=device)
    targets_flat = (seg_idx.unsqueeze(1) * group_count + groups.unsqueeze(0)).flatten()
    return targets_flat, {"start": start, "block_segments": block_segments, "groups": groups}


def _mask_from_flat_targets(
    targets_flat: t.Tensor,
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
) -> MaskIndices:
    """Broadcast one shared flat target index set to every row in the batch."""
    numel = segment_count * group_count
    is_target = t.zeros(numel, dtype=t.bool, device=device)
    is_target[targets_flat] = True
    all_idx = t.arange(numel, device=device)
    return MaskIndices(
        targets=all_idx[is_target].unsqueeze(0).expand(batch_size, -1),
        context=all_idx[~is_target].unsqueeze(0).expand(batch_size, -1),
    )


def mask_random(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    n_targets: int,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, dict]:
    """
    returns:
        context_indices: [B, n_context]
        targets_indices: [B, n_targets]
    """
    min_valid = _min_valid_segments(token_valid, segment_count)
    valid_numel = min_valid * group_count
    n_targets = min(max(1, n_targets), valid_numel - 1)
    targets_flat = t.randperm(valid_numel, device=device)[:n_targets]
    mask = _mask_from_flat_targets(targets_flat, batch_size, segment_count, group_count, device)
    return (mask, {}) if return_meta else mask


def mask_temporal_blocks(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    block_segments: int,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, dict]:
    targets_flat, meta = _sample_block_targets(
        segment_count, group_count, device, token_valid,
        block_segments=block_segments, block_groups=group_count,  # all groups
    )
    mask = _mask_from_flat_targets(targets_flat, batch_size, segment_count, group_count, device)
    return (mask, meta) if return_meta else mask


def mask_spatial_blocks(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    block_groups: int,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, dict]:
    min_valid = _min_valid_segments(token_valid, segment_count)
    targets_flat, meta = _sample_block_targets(
        segment_count, group_count, device, token_valid,
        block_segments=min_valid, block_groups=block_groups,  # full available duration
    )
    mask = _mask_from_flat_targets(targets_flat, batch_size, segment_count, group_count, device)
    return (mask, meta) if return_meta else mask


def mask_tube_blocks(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    block_segments: int,
    block_groups: int,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, dict]:
    targets_flat, meta = _sample_block_targets(
        segment_count, group_count, device, token_valid,
        block_segments=block_segments, block_groups=block_groups,
    )
    mask = _mask_from_flat_targets(targets_flat, batch_size, segment_count, group_count, device)
    return (mask, meta) if return_meta else mask


def mask_mixed(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
    *,
    temporal_block_segments: int = 6,
    spatial_block_groups: int = 6,
    tube_block_segments: int = 4,
    tube_block_groups: int = 5,
    random_n_targets: int = 20,
    weights: tuple[float, float, float, float] = (0.4, 0.2, 0.3, 0.1),
) -> MaskIndices | tuple[MaskIndices, dict]:
    """Picks ONE strategy for the entire batch (not a per-sample blend) --
    the whole batch shares one mask, redrawn fresh every call. See the plan
    write-up: a per-sample-random shared-n_targets design couldn't give
    temporal/spatial/tube independently controlled, non-degenerate shapes.
    """
    strategy = random.choices(
        ["temporal_blocks", "tube_blocks", "spatial_blocks", "random"], weights=weights, k=1,
    )[0]

    if strategy == "temporal_blocks":
        result = mask_temporal_blocks(
            batch_size, segment_count, group_count, device, temporal_block_segments, token_valid, return_meta,
        )
    elif strategy == "tube_blocks":
        result = mask_tube_blocks(
            batch_size, segment_count, group_count, device, tube_block_segments, tube_block_groups, token_valid, return_meta,
        )
    elif strategy == "spatial_blocks":
        result = mask_spatial_blocks(
            batch_size, segment_count, group_count, device, spatial_block_groups, token_valid, return_meta,
        )
    else:
        result = mask_random(batch_size, segment_count, group_count, device, random_n_targets, token_valid, return_meta)

    if not return_meta:
        return result
    mask, meta = result
    return mask, {"strategy": strategy, **meta}
