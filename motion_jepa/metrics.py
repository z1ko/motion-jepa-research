
import torch as T
import torch.nn.functional as F

# bad:
#   std_mean -> 0
#   std_min -> 0
#   cosine_concentration -> 1
#   effective_rank_ratio -> 0
#   top_eig_ratio -> 1
# 
# healthy:
#   std_mean reasonably nonzero
#   std_min not too close to 0
#   cosine_concentration not too high
#   effective_rank_ratio reasonably high
#   top_eig_ratio not dominant

@T.no_grad()
def measure_collapse(z: T.Tensor, eps: float = 1e-12) -> dict[str, T.Tensor]:
    z = z.detach().reshape(-1, z.shape[-1])  # [S, D]

    samples, dim = z.shape
    if samples < 2:
        raise ValueError("Not enough samples to calculate collapse metrics")
    
    z_normalized = F.normalize(z, dim=1, eps=eps)
    z_centered = z - z.mean(dim=0, keepdim=True)
    z_std = z_centered.std(dim=0, unbiased=False)

    # Near 1 means many embeddings point in the same direction.
    cosine_concentration = z_normalized.mean(dim=0).norm()

    covariance = z_centered.T @ z_centered / max(samples - 1, 1)
    eigvals = T.linalg.eigvalsh(covariance).clamp_min(0)

    eig_sum = eigvals.sum().clamp_min(eps)
    probs = (eigvals / eig_sum).clamp_min(eps)
    max_rank = min(dim, samples - 1)

    effective_rank = T.exp(-(probs * probs.log()).sum())
    effective_rank_ratio = effective_rank / max_rank
    top_eig_ratio = eigvals.max() / eig_sum

    return {
        "std_mean": z_std.mean(),
        "std_min": z_std.min(),
        "dims_below_1e-3": (z_std < 1e-3).sum(),
        "cosine_concentration": cosine_concentration,
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank_ratio,
        "top_eig_ratio": top_eig_ratio,
        "l2_mean": z.norm(dim=-1).mean(),
    }

@T.no_grad()
def measure_embedding_health_score(
    metrics: dict[str, T.Tensor],
    min_std_target: float = 0.1,
    std_target: float = 1.0,
    eps: float = 1e-8
) -> T.Tensor:
    """
    Returns a scalar in roughly [0, 1], where:
      1 = healthier / less collapsed
      0 = more collapsed

    Expects metrics from metric_collapse().
    """

    # Reward std_mean up to std_target.
    std_score = (metrics["std_mean"] / std_target).clamp(0.0, 1.0)
    # Reward std_min up to min_std_target, this catches dead dimensions.
    min_std_score = (metrics["std_min"] / min_std_target).clamp(0.0, 1.0)
    # Already in [0, 1], higher is better.
    rank_score = metrics["effective_rank_ratio"].clamp(0.0, 1.0)
    # cosine_concentration near 1 is bad, near 0 is good.
    isotropy_score = (1.0 - metrics["cosine_concentration"]).clamp(0.0, 1.0)
    # top_eig_ratio near 1 is bad, lower is better.
    spectral_score = (1.0 - metrics["top_eig_ratio"]).clamp(0.0, 1.0)

    # Weighted geometric mean.
    # Geometric mean is useful because one terrible component drags down the score.
    scores = T.stack([std_score, min_std_score, rank_score, isotropy_score, spectral_score])
    weight = T.tensor(
        [0.20, 0.20, 0.25, 0.20, 0.15],
        device=scores.device,
        dtype=scores.dtype,
    )

    health = T.exp((weight * T.log(scores.clamp_min(eps))).sum())
    return health


@T.no_grad()
def measure_prediction_target_alignment(prediction: T.Tensor, target: T.Tensor, eps: float = 1e-12) -> dict[str, T.Tensor]:
    
    prediction = prediction.detach().reshape(-1, prediction.shape[-1])
    target = target.detach().reshape(-1, target.shape[-1])

    cosine = F.cosine_similarity(prediction, target, dim=1, eps=eps).mean()
    smooth_l1 = F.smooth_l1_loss(prediction, target)
    mse = F.mse_loss(prediction, target)

    return {
        "alignment_cosine": cosine,
        "alignment_smooth_l1": smooth_l1,
        "alignment_mse": mse
    }