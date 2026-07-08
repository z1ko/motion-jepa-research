
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
        "dims_below_1e-3": (z_std < 1e-3).sum().float(),
        "cosine_concentration": cosine_concentration,
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank_ratio,
        "top_eig_ratio": top_eig_ratio,
        "l2_mean": z.norm(dim=-1).mean(),
    }
