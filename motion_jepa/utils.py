
import math

import torch as t
import torch.optim as optim
import numpy as np

CHANNELS: list[str] = ["pos", "vel", "acc", "tau"]
JOINTS: list[str] = [
    "pelvis_tilt", 
    "pelvis_list", 
    "pelvis_rotation",
    "pelvis_tx", 
    "pelvis_ty", 
    "pelvis_tz",
    "hip_flexion_r", 
    "hip_adduction_r", 
    "hip_rotation_r",
    "knee_angle_r", 
    "ankle_angle_r", 
    "subtalar_angle_r", 
    "mtp_angle_r",
    "hip_flexion_l", 
    "hip_adduction_l", 
    "hip_rotation_l",
    "knee_angle_l", 
    "ankle_angle_l", 
    "subtalar_angle_l", 
    "mtp_angle_l",
    "lumbar_bending", 
    "lumbar_extension", 
    "lumbar_twist",
    "thorax_bending", 
    "thorax_extension", 
    "thorax_twist",
    "head_bending", 
    "head_extension", 
    "head_twist",
    "scapula_abduction_r", 
    "scapula_elevation_r", 
    "scapula_upward_rot_r",
    "scapula_abduction_l", 
    "scapula_elevation_l", 
    "scapula_upward_rot_l",
    "shoulder_r_x", 
    "shoulder_r_y", 
    "shoulder_r_z",
    "shoulder_l_x", 
    "shoulder_l_y", 
    "shoulder_l_z",
    "elbow_flexion_r", 
    "elbow_flexion_l",
    "pro_sup_r", 
    "pro_sup_l",
    "wrist_flexion_r", 
    "wrist_deviation_r",
    "wrist_flexion_l", 
    "wrist_deviation_l",
]

_COLUMNS_KINEMATIC = [f"{joint}{suffix}" for joint in JOINTS for suffix in ("", "_vel", "_acc", "_tau")]
_COLUMNS_EXTRA     = ["time", "action", "grf_total_x", "grf_total_y", "grf_total_z"]
_COLUMNS_METADATA  = ["subject_mass_kg", "subject_height_m"]

_SCALE_SUFFIXES = ("_scale_x", "_scale_y", "_scale_z")
_GRAVITY_M_S2 = 9.80665

MIN_ORIGINAL_HZ = 99.0

_ROOT_TRANSLATION_JOINTS = ("pelvis_tx", "pelvis_ty", "pelvis_tz")
_ROOT_ROTATION_JOINTS = ("pelvis_rotation",)

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def center_root_channels(x: np.ndarray) -> np.ndarray:
    """Remove the absolute pelvis origin/heading from a kinematics window.

    Mocap labs place their world origin and subject-facing convention
    arbitrarily, so raw pelvis translation/rotation values leak dataset
    identity rather than carrying motion information (see e.g. the ~2m gap
    in pelvis_ty means across datasets). Subtracting the first frame keeps
    the relative trajectory within the window while discarding that
    per-lab/per-trial constant offset. Velocities/accelerations are left
    untouched since they're already invariant to a constant position shift.
    """
    x = x.copy()
    pos_idx = CHANNELS.index("pos")

    for joint in _ROOT_TRANSLATION_JOINTS:
        j = JOINTS.index(joint)
        x[:, j, pos_idx] -= x[0, j, pos_idx]

    for joint in _ROOT_ROTATION_JOINTS:
        j = JOINTS.index(joint)
        x[:, j, pos_idx] = wrap_to_pi(x[:, j, pos_idx] - x[0, j, pos_idx])

    return x

def ema_linear_scheduler(max_steps: int, global_step: int, ema_momentun: float) -> float:
    if max_steps is None or max_steps <= 0:
        raise ValueError("Linear EMA scheduling requires Trainer(max_steps=...).")

    progress = min(global_step / max_steps, 1.0)
    ema = ema_momentun + progress * (1.0 - ema_momentun)
    return ema
    

# Used for taus
def signed_log1p_tau(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    tau_idx = CHANNELS.index("tau")
    x[:, :, tau_idx] = np.sign(x[:, :, tau_idx]) * np.log1p(
        np.abs(x[:, :, tau_idx])
    )
    return x


def prepare_window(
    x: np.ndarray,
    *,
    window_size: int,
    segment_size: int,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    clip_value: float | None,
    dtype: np.dtype = np.float32,
) -> dict[str, t.Tensor]:
    """Turn a raw (possibly short) kinematics window into a model-ready sample.

    Shared by every place that reads a window off disk (MotionWindowDataset,
    MotionRandomCropDataset, WindowRowsDataset) so this pipeline can't drift
    between them. `x` is (T, D, C) with T <= window_size.
    """
    x = np.asarray(x, dtype=dtype)
    x = center_root_channels(x)

    if mean is not None and std is not None:
        x = signed_log1p_tau(x)
        x = (x - mean) / std
        if clip_value is not None:
            x = np.clip(x, -clip_value, clip_value)

    valid_frames = x.shape[0]
    if valid_frames < window_size:
        # Safe regardless of fill value: padded segments are masked out of
        # attention entirely (see masking.py/architecture/model.py).
        x_padded = np.zeros((window_size, *x.shape[1:]), dtype=dtype)
        x_padded[:valid_frames] = x
        x = x_padded

    # A segment counts as valid only if every one of its frames is real.
    segment_count = window_size // segment_size
    valid_segments = min(valid_frames // segment_size, segment_count)

    # np.clip can sometimes return non-contiguous views.
    x_tensor = t.as_tensor(np.ascontiguousarray(x), dtype=t.float32)
    return {
        "x": x_tensor,
        "valid_segments": t.tensor(valid_segments, dtype=t.long),
    }

def estimate_original_hz(time: np.ndarray) -> float:
    time = np.asarray(time, dtype=np.float64)

    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if dt.size == 0:
        raise ValueError("Could not estimate Hz: no positive time deltas.")

    return float(1.0 / np.median(dt))


def schedule_with_warmup(
    optimizer : optim.Optimizer, 
    num_warmup_steps: int, 
) -> optim.lr_scheduler.LambdaLR:
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        return 1.0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cosine_schedule_with_warmup(
    optimizer : optim.Optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    eta_min_fraction=0.1
) -> optim.lr_scheduler.LambdaLR:
    """
    Create a cosine schedule with warmup using LambdaLR.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Steps for linear warmup.
        num_training_steps: Total training steps.
        eta_min_fraction: Minimum LR as fraction of peak LR (default: 10%).

    Returns:
        LambdaLR scheduler with cosine-plus-warmup multiplier.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        # Scale to [eta_min_fraction, 1.0]
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_fraction + (1.0 - eta_min_fraction) * cosine_factor

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)