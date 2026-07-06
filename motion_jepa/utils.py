
import numpy as np
import torch as T

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

def estimate_original_hz(time: np.ndarray) -> float:
    time = np.asarray(time, dtype=np.float64)

    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if dt.size == 0:
        raise ValueError("Could not estimate Hz: no positive time deltas.")

    return float(1.0 / np.median(dt))