
import random
from dataclasses import dataclass

import torch as t

@dataclass(frozen=True)
class MaskIndices:
    context: t.Tensor
    targets: t.Tensor


# 5 kinematic chains covering all 11 groups exactly once, as group-list
# indices (see config.training.groups' order in config/experiment.yaml:
# pelvis, right_upper_leg, right_foot, left_upper_leg, left_foot, spine,
# head, right_shoulder_complex, right_lower_arm, left_shoulder_complex,
# left_lower_arm). Chain-based instead of raw contiguous index slices: the
# group axis isn't a translation-invariant grid like an image, so an
# arbitrary index-slice can straddle two anatomically unrelated groups (e.g.
# right_foot + left_upper_leg). Update this if config.training.groups is
# ever reordered/changed.
_CHAINS: tuple[tuple[int, ...], ...] = (
    (1, 2),      # right_leg: right_upper_leg, right_foot
    (3, 4),      # left_leg: left_upper_leg, left_foot
    (0, 5, 6),   # torso: pelvis, spine, head
    (7, 8),      # right_arm: right_shoulder_complex, right_lower_arm
    (9, 10),     # left_arm: left_shoulder_complex, left_lower_arm
)


def _valid_segments_for(token_valid: t.Tensor | None, sample: int, segment_count: int) -> int:
    if token_valid is None:
        return segment_count
    return int(token_valid[sample, :, 0].sum().item())  # any group column, all identical


def _sample_chains(n_chains: int, device: t.device) -> t.Tensor:
    """n_chains whole chains' member group indices, flattened. Chain sizes
    differ (torso=3, limbs=2), so the resulting group count varies per draw
    -- reconciled by _build_mask_from_candidates' batch-wide truncation, not
    here, same as I-JEPA/V-JEPA's own per-sample count variance.
    """
    n_chains = min(max(1, n_chains), len(_CHAINS))
    chosen = random.sample(_CHAINS, n_chains)
    groups = [g for chain in chosen for g in chain]
    return t.tensor(groups, device=device, dtype=t.long)


def _sample_block_targets_for_sample(
    segment_count: int,
    group_count: int,
    device: t.device,
    valid_segments: int,
    *,
    block_segments: int,
    n_chains: int | None,
) -> tuple[t.Tensor, dict]:
    """One sample's randomly-positioned block target indices. n_chains=None
    means all groups (temporal_blocks); otherwise n_chains whole kinematic
    chains are selected (spatial_blocks/tube_blocks). Clipped to this
    sample's OWN valid_segments -- guaranteed valid for this row without
    reference to any other row in the batch.
    """
    groups = t.arange(group_count, device=device) if n_chains is None else _sample_chains(n_chains, device)

    # If every group is covered (all-groups temporal block, or n_chains
    # happens to span every chain), the block must leave at least one
    # segment as context -- otherwise the whole valid range becomes target,
    # this sample's key_padding_mask row is fully True, and softmax
    # attention NaNs. Mirrors mask_random's `valid_numel - 1` reservation.
    # Not reachable under this repo's current config values (min_valid_frames
    # /segment_size keep valid_segments well above block_segments), but
    # nothing stops a future retune from hitting it.
    max_block_segments = valid_segments if len(groups) < group_count else max(valid_segments - 1, 1)
    block_segments = min(block_segments, max_block_segments)

    start = int(t.randint(0, valid_segments - block_segments + 1, (1,)).item())
    seg_idx = t.arange(start, start + block_segments, device=device)

    targets_flat = (seg_idx.unsqueeze(1) * group_count + groups.unsqueeze(0)).flatten()
    return targets_flat, {"start": start, "block_segments": block_segments, "groups": groups}


def _build_mask_from_candidates(
    candidates: list[t.Tensor],
    segment_count: int,
    group_count: int,
    device: t.device,
) -> MaskIndices:
    """Truncate every sample's candidate target set to the batch's minimum
    count (I-JEPA/V-JEPA's collator trick: fixed shape via truncation, not
    padding), then build per-sample targets/context. Context is the
    complement of each sample's own (already-truncated) target set, so both
    land on the same uniform (B, N) shape for free -- unlike I-JEPA/V-JEPA,
    whose context is its own independently-sampled region, not a simple
    complement, and needs its own separate truncation.
    """
    n_targets = min(len(c) for c in candidates)
    numel = segment_count * group_count
    all_idx = t.arange(numel, device=device)

    targets_rows, context_rows = [], []
    for candidate in candidates:
        if len(candidate) > n_targets:
            # Random subset, not a fixed slice -- avoids biasing which part
            # of a temporal block or which chain survives truncation.
            keep = t.randperm(len(candidate), device=device)[:n_targets]
            candidate = candidate[keep]
        is_target = t.zeros(numel, dtype=t.bool, device=device)
        is_target[candidate] = True
        targets_rows.append(all_idx[is_target])
        context_rows.append(all_idx[~is_target])

    return MaskIndices(targets=t.stack(targets_rows, dim=0), context=t.stack(context_rows, dim=0))


def mask_random(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    n_targets: int,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, list[dict]]:
    """
    returns:
        context_indices: [B, n_context]
        targets_indices: [B, n_targets]
    """
    candidates = []
    for i in range(batch_size):
        valid_segments = _valid_segments_for(token_valid, i, segment_count)
        valid_numel = valid_segments * group_count
        n = min(max(1, n_targets), valid_numel - 1)
        candidates.append(t.randperm(valid_numel, device=device)[:n])

    mask = _build_mask_from_candidates(candidates, segment_count, group_count, device)
    return (mask, [{}] * batch_size) if return_meta else mask


def mask_temporal_blocks(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    block_segments: int,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, list[dict]]:
    candidates, metas = [], []
    for i in range(batch_size):
        valid_segments = _valid_segments_for(token_valid, i, segment_count)
        targets_flat, meta = _sample_block_targets_for_sample(
            segment_count, group_count, device, valid_segments,
            block_segments=block_segments, n_chains=None,  # all groups
        )
        candidates.append(targets_flat)
        metas.append(meta)

    mask = _build_mask_from_candidates(candidates, segment_count, group_count, device)
    return (mask, metas) if return_meta else mask


def mask_spatial_blocks(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    block_groups: int,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, list[dict]]:
    candidates, metas = [], []
    for i in range(batch_size):
        valid_segments = _valid_segments_for(token_valid, i, segment_count)
        targets_flat, meta = _sample_block_targets_for_sample(
            segment_count, group_count, device, valid_segments,
            block_segments=valid_segments,  # full available duration
            n_chains=block_groups,
        )
        candidates.append(targets_flat)
        metas.append(meta)

    mask = _build_mask_from_candidates(candidates, segment_count, group_count, device)
    return (mask, metas) if return_meta else mask


def mask_tube_blocks(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    block_segments: int,
    block_groups: int,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, list[dict]]:
    candidates, metas = [], []
    for i in range(batch_size):
        valid_segments = _valid_segments_for(token_valid, i, segment_count)
        targets_flat, meta = _sample_block_targets_for_sample(
            segment_count, group_count, device, valid_segments,
            block_segments=block_segments, n_chains=block_groups,
        )
        candidates.append(targets_flat)
        metas.append(meta)

    mask = _build_mask_from_candidates(candidates, segment_count, group_count, device)
    return (mask, metas) if return_meta else mask


def mask_mixed(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
    *,
    temporal_block_segments: int = 3,
    spatial_block_groups: int = 2,  # of 5 chains -- 6 degenerates to "all chains" (only 5 exist)
    tube_block_segments: int = 6,
    tube_block_groups: int = 2,  # of 5 chains, same reason
    random_n_targets: int = 20,
    weights: tuple[float, float, float, float] = (0.4, 0.2, 0.3, 0.1),
) -> MaskIndices | tuple[MaskIndices, list[dict]]:
    """Each sample independently draws its own strategy, position, and (for
    spatial/tube) chain selection -- genuine per-sample diversity, not one
    shared mask for the whole batch. Different strategies/chain-picks
    produce different native token counts per sample; reconciled by
    _build_mask_from_candidates truncating every sample down to the batch's
    minimum count before stacking, the same fixed-shape-via-truncation trick
    I-JEPA/V-JEPA use for their own per-sample block-position variance.
    """
    candidates, metas = [], []
    for i in range(batch_size):
        valid_segments = _valid_segments_for(token_valid, i, segment_count)
        strategy = random.choices(
            ["temporal_blocks", "tube_blocks", "spatial_blocks", "random"], weights=weights, k=1,
        )[0]

        if strategy == "temporal_blocks":
            targets_flat, meta = _sample_block_targets_for_sample(
                segment_count, group_count, device, valid_segments,
                block_segments=temporal_block_segments, n_chains=None,
            )
        elif strategy == "tube_blocks":
            targets_flat, meta = _sample_block_targets_for_sample(
                segment_count, group_count, device, valid_segments,
                block_segments=tube_block_segments, n_chains=tube_block_groups,
            )
        elif strategy == "spatial_blocks":
            targets_flat, meta = _sample_block_targets_for_sample(
                segment_count, group_count, device, valid_segments,
                block_segments=valid_segments, n_chains=spatial_block_groups,
            )
        else:
            valid_numel = valid_segments * group_count
            n = min(max(1, random_n_targets), valid_numel - 1)
            targets_flat, meta = t.randperm(valid_numel, device=device)[:n], {}

        candidates.append(targets_flat)
        metas.append({"strategy": strategy, **meta})

    mask = _build_mask_from_candidates(candidates, segment_count, group_count, device)
    return (mask, metas) if return_meta else mask
