
from dataclasses import dataclass

import torch as t
import torch.nn.functional as f

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
) -> MaskIndices:
    scores = _random_scores(batch_size, segment_count, group_count, device)

    block = max(1, segment_count // 3)
    max_start = segment_count - block + 1
    starts = t.randint(max_start, (batch_size,), device=device)
    time = t.arange(segment_count, device=device).unsqueeze(0)
    in_block = (time >= starts.unsqueeze(1)) & (time < (starts + block).unsqueeze(1))
    scores = scores + in_block.unsqueeze(-1).float()
    return _rank_to_mask(scores, n_targets, token_valid)


def mask_spatial_blocks(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    n_targets: int,
    token_valid: t.Tensor | None = None,
) -> MaskIndices:
    scores = _random_scores(batch_size, segment_count, group_count, device)

    n_groups = max(1, round(group_count * 0.5))
    group_scores = t.rand(batch_size, group_count, device=device)
    groups = group_scores.argsort(dim=1)[:, :n_groups]
    scores.scatter_add_(
        dim=2,
        index=groups.unsqueeze(1).expand(-1, segment_count, -1),
        src=t.ones(batch_size, segment_count, n_groups, device=device),
    )
    return _rank_to_mask(scores, n_targets, token_valid)


def mask_tube_blocks(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    n_targets: int,
    token_valid: t.Tensor | None = None,
) -> MaskIndices:
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
    return _rank_to_mask(scores, n_targets, token_valid)


def mask_mixed(
    batch_size: int,
    segment_count: int,
    group_count: int,
    device: t.device,
    targets_p: float = 0.6,
    token_valid: t.Tensor | None = None,
) -> MaskIndices:
    if token_valid is not None:
        valid_counts = token_valid.flatten(1, 2).sum(dim=1)
        n_targets = _num_targets_from_valid(valid_counts, targets_p)
    else:
        n_targets = _num_targets(segment_count, group_count, targets_p)

    choices = t.rand(batch_size, device=device)
    masks = (
        mask_temporal_blocks(batch_size, segment_count, group_count, device, n_targets, token_valid),
        mask_tube_blocks(batch_size, segment_count, group_count, device, n_targets, token_valid),
        mask_spatial_blocks(batch_size, segment_count, group_count, device, n_targets, token_valid),
        mask_random(batch_size, segment_count, group_count, device, n_targets, token_valid),
    )

    targets = masks[0].targets.clone()
    context = masks[0].context.clone()

    tube = (choices >= 0.4) & (choices < 0.7)
    spatial = (choices >= 0.7) & (choices < 0.9)
    random = choices >= 0.9
    for select, mask in ((tube, masks[1]), (spatial, masks[2]), (random, masks[3])):
        targets[select] = mask.targets[select]
        context[select] = mask.context[select]

    return MaskIndices(context=context, targets=targets)


# Idea from: Masked Motion Predictors are Strong 3D Action Representation Learners
def mask_mamp(intensity: t.Tensor, *, targets_p: float = 0.6) -> MaskIndices:
    """
    intensity: [B, T, C]

    returns:
        context_indices: [B, n_context]
        targets_indices: [B, n_targets]

    Higher intensity tokens are more likely to be targets.
    """

    _eps = 1e-6
    _tau = 0.2

    B, T, G = intensity.shape
    numel = T * G

    n_targets = round(targets_p * numel)
    flat_intensity = intensity.reshape(B, numel).float()
    log_probs = f.log_softmax(
        flat_intensity / _tau,
        dim=-1
    )

    uniform = t.rand_like(log_probs).clamp_(_eps, 1.0 - _eps)
    gumbel  = -t.log(-t.log(uniform))
    scores  = log_probs + gumbel

    # Highest scores become targets.
    permut = scores.argsort(dim=1, descending=True)
    return MaskIndices(
        targets=permut[:, :n_targets],
        context=permut[:, n_targets:]
    )
    

def gather_from_mask(tokens: t.Tensor, mask: MaskIndices) -> tuple[t.Tensor, t.Tensor]:
    return (
        tokens.gather( # targets
            index=mask.targets.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]), 
            dim=1
        ),
        tokens.gather( # context
            index=mask.context.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]), 
            dim=1
        )
    )
