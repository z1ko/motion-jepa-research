
from dataclasses import dataclass

import torch as t
import torch.nn.functional as f

@dataclass(frozen=True)
class MaskIndices:
    context: t.Tensor
    targets: t.Tensor

def mask_random(x: t.Tensor, *, targets_p: float = 0.6) -> MaskIndices:
    """
    x: [B, T, C, D]

    returns:
        context_indices: [B, n_context]
        targets_indices: [B, n_targets]
    """

    B, T, G, C  = x.shape
    numel = T * G

    n_targets = round(targets_p * numel)
    scores = t.rand(B, numel, device=x.device)

    permut = scores.argsort(dim=1)
    return MaskIndices(
        targets=permut[:, :n_targets],
        context=permut[:, n_targets:]
    )

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