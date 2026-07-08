
from dataclasses import dataclass

import torch as t

@dataclass(frozen=True)
class MaskIndices:
    context: t.Tensor
    targets: t.Tensor

def _num_targets(segment_count: int, group_count: int, targets_p: float) -> int:
    numel = segment_count * group_count
    return min(max(1, round(targets_p * numel)), numel - 1)


def _num_targets_from_valid(valid_counts: t.Tensor, targets_p: float) -> int:
    """Like `_num_targets`, but sized from each sample's actual valid
    (non-padded) token count instead of the fixed segment_count*group_count.

    Uses the batch's minimum valid count so a single, dense `n_targets` is
    safely satisfiable for every sample without ever needing to select a
    padded token as a target. Recomputing the ratio against `min_valid`
    (rather than naively clipping the fixed-shape n_targets) keeps the
    intended context/target balance for the shortest sample in the batch.
    """
    min_valid = int(valid_counts.min().item())
    return min(max(1, round(targets_p * min_valid)), max(min_valid - 1, 1))


def _rank_to_mask(
    scores: t.Tensor,
    n_targets: int,
    token_valid: t.Tensor | None = None,
) -> MaskIndices:
    if token_valid is not None:
        # Padded tokens must never be picked as prediction targets: forcing
        # their score to -inf guarantees they always sort into `context`.
        scores = scores.masked_fill(~token_valid, float("-inf"))

    permut = scores.flatten(1, 2).argsort(dim=1, descending=True)
    return MaskIndices(
        targets=permut[:, :n_targets],
        context=permut[:, n_targets:],
    )


def _random_scores(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
) -> t.Tensor:
    return t.rand(batch_size, segment_count, group_count, device=device) * 0.01


def mask_random(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    n_targets: int,
    token_valid: t.Tensor | None = None,
) -> MaskIndices:
    """
    returns:
        context_indices: [B, n_context]
        targets_indices: [B, n_targets]
    """

    scores = t.rand(batch_size, segment_count, group_count, device=device)
    return _rank_to_mask(scores, n_targets, token_valid)


def mask_temporal_blocks(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    n_targets: int,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, dict]:
    scores = _random_scores(batch_size, segment_count, group_count, device)

    block = max(1, segment_count // 3)
    max_start = segment_count - block + 1
    starts = t.randint(max_start, (batch_size,), device=device)
    time = t.arange(segment_count, device=device).unsqueeze(0)
    in_block = (time >= starts.unsqueeze(1)) & (time < (starts + block).unsqueeze(1))
    scores = scores + in_block.unsqueeze(-1).float()
    mask = _rank_to_mask(scores, n_targets, token_valid)
    if return_meta:
        return mask, {"starts": starts, "block": block}
    return mask


def mask_spatial_blocks(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    n_targets: int,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, dict]:
    scores = _random_scores(batch_size, segment_count, group_count, device)

    n_groups = max(1, round(group_count * 0.5))
    group_scores = t.rand(batch_size, group_count, device=device)
    groups = group_scores.argsort(dim=1)[:, :n_groups]
    scores.scatter_add_(
        dim=2,
        index=groups.unsqueeze(1).expand(-1, segment_count, -1),
        src=t.ones(batch_size, segment_count, n_groups, device=device),
    )
    mask = _rank_to_mask(scores, n_targets, token_valid)
    if return_meta:
        return mask, {"groups": groups}
    return mask


def mask_tube_blocks(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    n_targets: int,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, dict]:
    scores = _random_scores(batch_size, segment_count, group_count, device)

    block = max(1, segment_count // 3)
    max_start = segment_count - block + 1
    starts = t.randint(max_start, (batch_size,), device=device)
    time = t.arange(segment_count, device=device).unsqueeze(0)
    in_time = (time >= starts.unsqueeze(1)) & (time < (starts + block).unsqueeze(1))

    n_groups = max(1, round(group_count * 0.5))
    group_scores = t.rand(batch_size, group_count, device=device)
    groups = group_scores.argsort(dim=1)[:, :n_groups]
    in_group = t.zeros(batch_size, group_count, device=device, dtype=t.bool)
    in_group.scatter_(dim=1, index=groups, value=True)

    scores = scores + (in_time.unsqueeze(-1) & in_group.unsqueeze(1)).float()
    mask = _rank_to_mask(scores, n_targets, token_valid)
    if return_meta:
        return mask, {"starts": starts, "block": block, "groups": groups}
    return mask


def mask_mixed(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    targets_p: float = 0.6,
    token_valid: t.Tensor | None = None,
    return_meta: bool = False,
) -> MaskIndices | tuple[MaskIndices, dict]:
    if token_valid is not None:
        valid_counts = token_valid.flatten(1, 2).sum(dim=1)
        n_targets = _num_targets_from_valid(valid_counts, targets_p)
    else:
        n_targets = _num_targets(segment_count, group_count, targets_p)

    # return_meta on the sub-calls costs nothing extra: starts/groups are
    # already computed internally to bias scores either way, this just also
    # returns them (used by scripts/viz_masks.py to outline the intended
    # block and label which sub-strategy each sample resolved to).
    choices = t.rand(batch_size, device=device)
    temporal_mask, temporal_meta = mask_temporal_blocks(
        batch_size, segment_count, group_count, device, n_targets, token_valid, return_meta=True
    ) # type: ignore
    tube_mask, tube_meta = mask_tube_blocks(
        batch_size, segment_count, group_count, device, n_targets, token_valid, return_meta=True
    ) # type: ignore
    spatial_mask, spatial_meta = mask_spatial_blocks(
        batch_size, segment_count, group_count, device, n_targets, token_valid, return_meta=True
    ) # type: ignore
    random_mask = mask_random(batch_size, segment_count, group_count, device, n_targets, token_valid)
    masks = (temporal_mask, tube_mask, spatial_mask, random_mask)

    targets = masks[0].targets.clone()
    context = masks[0].context.clone()

    tube = (choices >= 0.4) & (choices < 0.7)
    spatial = (choices >= 0.7) & (choices < 0.9)
    random = choices >= 0.9
    for select, mask in ((tube, masks[1]), (spatial, masks[2]), (random, masks[3])):
        targets[select] = mask.targets[select]
        context[select] = mask.context[select]

    result = MaskIndices(context=context, targets=targets)
    if not return_meta:
        return result

    strategy = ["temporal_blocks"] * batch_size
    for i in range(batch_size):
        if tube[i]:
            strategy[i] = "tube_blocks"
        elif spatial[i]:
            strategy[i] = "spatial_blocks"
        elif random[i]:
            strategy[i] = "random"

    return result, {
        "strategy": strategy,
        "temporal_blocks": temporal_meta,
        "tube_blocks": tube_meta,
        "spatial_blocks": spatial_meta,
    }
