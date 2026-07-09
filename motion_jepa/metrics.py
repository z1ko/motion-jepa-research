
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


@T.no_grad()
def measure_distillation(p1: T.Tensor, log_p1: T.Tensor, p2: T.Tensor, log_p2: T.Tensor) -> dict[str, T.Tensor]:
    """Health of the predicted/target softmax distributions (CenteredCrossEntropyLoss).

    p1_entropy/p2_entropy near log(embed_dim) means the distribution is too
    flat to carry signal; near 0 means it's collapsed onto one dimension --
    the exact failure mode centering is meant to prevent. argmax_agreement
    is a soft "accuracy" proxy: how often the predicted distribution's peak
    dimension matches the target's.
    """
    return {
        "p1_entropy": -(p1 * log_p1).sum(dim=-1).mean(),
        "p2_entropy": -(p2 * log_p2).sum(dim=-1).mean(),
        "argmax_agreement": (p1.argmax(-1) == p2.argmax(-1)).float().mean(),
    }


@T.no_grad()
def measure_teacher_student_similarity(student_params, teacher_params) -> T.Tensor:
    """Global cosine similarity between the flattened student/teacher encoder weights.

    Not an average of per-tensor cosines (parameter tensors vary wildly in
    size) -- one cosine over the full concatenated weight vector. Near 1
    means EMA momentum is too low for distillation to matter; falling
    steadily apart is expected and healthy.
    """
    student_vec = T.cat([p.reshape(-1) for p in student_params])
    teacher_vec = T.cat([p.reshape(-1) for p in teacher_params])
    return F.cosine_similarity(student_vec, teacher_vec, dim=0)


@T.no_grad()
def measure_grad_norm(parameters) -> T.Tensor:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return T.tensor(0.0)
    return T.norm(T.stack([g.norm(2) for g in grads]), 2)
