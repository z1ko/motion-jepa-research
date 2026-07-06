"""Fast, schema-aware loading of motion CSV files (see data/example.csv).

Layout of one CSV file (378 columns, all rows = one subject/trial):
- frame, time: per-frame index / timestamp
- subject_mass_kg, subject_height_m: static, one value per file
- <body>_scale_{x,y,z}: static body-segment scale factors, one value per file
- <joint>{,_vel,_acc,_tau}: 49 joints x 4 channels -> the kinematic tensor fed to the model
- <body>_grf_{x,y,z} and grf_total_{x,y,z}: per-frame ground reaction forces (metadata only)
- <body>_contact: ignored entirely
- action: per-frame label (metadata only)
"""

from dataclasses import dataclass
import glob
from pathlib import Path
import random
from typing import Any

import numpy as np
import polars as pl

from motion_jepa.dataset import MotionDatasetWriter, MotionZarrStore, RunningKinematicsStats, _path_of_normalization_stats, _path_of_samples_index, _path_of_windows_index, stable_suid
from motion_jepa.types import MotionSample, NormalizationStats
from motion_jepa.utils import _COLUMNS_EXTRA, _COLUMNS_KINEMATIC, _COLUMNS_METADATA, _GRAVITY_M_S2, _SCALE_SUFFIXES, CHANNELS, JOINTS, MIN_ORIGINAL_HZ, estimate_original_hz

# ================================================================================================================
# SCHEMA
# ================================================================================================================

# Overrides schema for known columns
def _schema_overrides(columns: list[str]) -> dict[str, pl.DataType]:
    overrides = {"frame": pl.Int64, "action": pl.Utf8}
    for col in columns:
        if col in overrides:
            continue
        if col.endswith("_contact"):
            overrides[col] = pl.Int8
        else:
            overrides[col] = pl.Float32
    return overrides

# ================================================================================================================
# LOADING CSV FROM FILES
# ================================================================================================================


def load_sample_from_csv(path: Path) -> MotionSample:
    header = pl.scan_csv(path, n_rows=0).collect_schema().names()

    scale_columns = [c for c in header if c.endswith(_SCALE_SUFFIXES)]
    extra_columns = [c for c in header if c in _COLUMNS_EXTRA or "_grf_" in c]

    lf = pl.scan_csv(path, schema_overrides=_schema_overrides(header))
    kinematics = lf.select(_COLUMNS_KINEMATIC).collect().to_numpy().reshape(-1, len(JOINTS), len(CHANNELS))
    metadata   = lf.select(_COLUMNS_METADATA + scale_columns).first().collect().row(0, named=True)
    extra      = lf.select(extra_columns).collect()

    # Add additional informations
    dataset, subject, trial = path.parts[-3], path.parts[-2], path.parts[-1]

    return MotionSample(
        kinematics=kinematics,
        metadata=metadata,
        extra=extra,
        path=str(path),
        dataset=dataset,
        subject=subject,
        trial=trial
    )

# ================================================================================================================
# RESAMPLING
# ================================================================================================================

ANGLE_JOINTS = [joint for joint in JOINTS if joint not in {"pelvis_tx", "pelvis_ty", "pelvis_tz"}]
ANGLE_JOINT_INDICES = [JOINTS.index(j) for j in ANGLE_JOINTS]

def resample_to_hz(sample: MotionSample, hz: float) -> MotionSample:
    """Resample a MotionSample onto a uniform time grid at `hz`.

    Numeric channels (kinematics + numeric changing_meta, e.g. grf) are linearly
    interpolated. String channels (e.g. action) use nearest-neighbor lookup, since
    interpolating a label doesn't make sense.
    """

    time = sample.extra["time"].to_numpy()
    pos_idx = CHANNELS.index("pos")

    new_timesteps = int(np.floor((time[-1] - time[0]) * hz)) + 1
    new_time = time[0] + np.arange(new_timesteps) / hz

    idx_hi = np.searchsorted(time, new_time, side="right").clip(1, len(time) - 1)
    idx_lo = idx_hi - 1

    t_lo, t_hi = time[idx_lo], time[idx_hi]
    weight = np.where(t_hi > t_lo, (new_time - t_lo) / (t_hi - t_lo), 0.0)

    # Copy as float64 for interpolation accuracy.
    unwrapped = sample.kinematics.astype(np.float64, copy=True)

    # Only unwrap angular position joints, not pelvis translations.
    unwrapped[:, ANGLE_JOINT_INDICES, pos_idx] = np.unwrap(
        unwrapped[:, ANGLE_JOINT_INDICES, pos_idx],
        axis=0,
    )

    kinematics = unwrapped[idx_lo] + weight[:, None, None] * (
        unwrapped[idx_hi] - unwrapped[idx_lo]
    )

    # Wrap angular positions back to [-pi, pi].
    kinematics[:, ANGLE_JOINT_INDICES, pos_idx] = wrap_to_pi(
        kinematics[:, ANGLE_JOINT_INDICES, pos_idx]
    )

    str_columns = [c for c, dt in sample.extra.schema.items() if dt == pl.Utf8]
    num_columns = [c for c in sample.extra.columns if c not in str_columns]

    numeric = sample.extra.select(num_columns).to_numpy()
    numeric_resampled = numeric[idx_lo] + weight[:, None] * (numeric[idx_hi] - numeric[idx_lo])
    nearest_idx = np.where(weight < 0.5, idx_lo, idx_hi)

    # Recompute extra data relative to new time
    extra = pl.DataFrame(
        {**dict(zip(num_columns, numeric_resampled.T)), **{
            c: sample.extra[c].to_numpy()[nearest_idx] for c in str_columns
        }}
    ).select(sample.extra.columns)

    return MotionSample(
        kinematics=kinematics.astype(np.float32),
        metadata=sample.metadata,
        extra=extra,
        path=sample.path,
        dataset=sample.dataset,
        subject=sample.subject,
        trial=sample.trial
    )


# ================================================================================================================
# NORMALIZATION OF DYNAMICS TO REMOVE SUBJECT-MASS BIAS
# ================================================================================================================

def normalize_dynamics(sample: MotionSample) -> MotionSample:
    """Remove subject-mass bias from torque and ground-reaction-force channels.

    Physically grounded, per-file normalization that needs no cross-file statistics:
    - tau (joint torque) is divided by body mass -> Nm/kg
    - GRF (per-body + total) is divided by body weight -> units of bodyweight (BW)
    Joint angles (pos/vel/acc) are already subject-invariant by construction (same
    generalized-coordinate definition regardless of skeleton size) and are left
    untouched here; see `compute_normalization_stats`/`apply_normalization_stats`
    for z-scoring those against dataset-wide statistics.
    """

    mass = sample.metadata["subject_mass_kg"]
    bodyweight = mass * _GRAVITY_M_S2

    kinematics = sample.kinematics.copy()
    tau_idx = CHANNELS.index("tau")
    kinematics[:, :, tau_idx] = kinematics[:, :, tau_idx] / mass

    grf_columns = [c for c in sample.extra.columns if "_grf_" in c or c.startswith("grf_total")]
    extra = sample.extra.with_columns([(pl.col(c) / bodyweight).alias(c) for c in grf_columns])

    return MotionSample(
        kinematics=kinematics, 
        metadata=sample.metadata,
        extra=extra,
        path=sample.path,
        dataset=sample.dataset,
        subject=sample.subject,
        trial=sample.trial
    )

# ================================================================================================================
# DATASET GENERATION
# ================================================================================================================

def create_raw_motion_dataset(
    *,
    raw_glob: str,
    output_root: Path | str,
    hz: float = 60.0,
    chunk_length: int = 256,
    overwrite: bool = False
):
    output_root = Path(output_root)

    files = sorted(Path(p) for p in glob.glob(raw_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {raw_glob}")
    
    writer = MotionDatasetWriter(output_root, chunk_length=chunk_length)
    writer.prepare(overwrite=overwrite)

    # How many files we skipped because of low hz
    skipped_low_hz = 0
    # How many files to stored
    written = 0

    sample_rows: list[dict] = []
    for i, path in enumerate(files, start=1):
        suid = stable_suid(path)
        
        sample = load_sample_from_csv(Path(path))

        original_hz = estimate_original_hz(sample.extra["time"].to_numpy())
        if original_hz < MIN_ORIGINAL_HZ:
            skipped_low_hz += 1
            print(
                f"Skipping low-Hz sample: "
                f"original_hz={original_hz:.2f}, path={path}"
            )
            continue

        sample = resample_to_hz(sample, hz=60.0)
        sample = normalize_dynamics(sample)

        row = writer.write_sample(suid=suid, sample=sample)
        sample_rows.append(row)
        written += 1

        if i % 100 == 0:
            print(f"[store] wrote {i}/{len(files)} samples")

    samples_df = pl.DataFrame(sample_rows)
    samples_df.write_parquet(_path_of_samples_index(output_root))

    print(f"Wrote dataset: {output_root}")
    print(f"Skipped because of low hz: {skipped_low_hz}")
    print(f"Samples: {samples_df.height}")


# ================================================================================================================
# SPLIT AND WINDOWS
# ================================================================================================================

def _make_windows_for_sample(
    *,
    suid: str,
    num_frames: int,
    split: str,
    window_size: int,
    stride: int,
) -> list[dict]:
    rows = []

    if num_frames < window_size:
        return rows

    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size

        rows.append({
            "suid": suid,
            "split": split,
            "start": int(start),
            "end": int(end),
            "window_size": int(window_size),
        })

    return rows


def _assign_subject_splits(
    samples: pl.DataFrame,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 13,
    subject_col: str = "subject",
) -> dict[str, str]:
    if subject_col not in samples.columns:
        raise ValueError(
            f"samples.parquet must contain a {subject_col!r} column "
            "to prevent subject leakage."
        )

    total = train_ratio + val_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total}"
        )

    subjects = (
        samples
        .select(subject_col)
        .unique()
        .drop_nulls()
        .get_column(subject_col)
        .to_list()
    )

    subjects = [str(s) for s in subjects]

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = int(round(train_ratio * n_subjects))
    n_val = int(round(val_ratio * n_subjects))

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]

    subject_to_split = {}

    for subject in train_subjects:
        subject_to_split[subject] = "train"

    for subject in val_subjects:
        subject_to_split[subject] = "val"

    return subject_to_split

def generate_splits_and_windows(
    *,
    root: Path | str,
    window_length: int = 256,
    stride: int = 128,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 13,
) -> None:
    root = Path(root)

    samples_path = _path_of_samples_index(root)
    windows_path = _path_of_windows_index(root)

    samples = pl.read_parquet(samples_path)

    required_cols = {"suid", "num_frames", "subject"}
    missing = required_cols - set(samples.columns)
    if missing:
        raise ValueError(
            f"samples.parquet is missing required columns: {sorted(missing)}"
        )
    
    subject_to_split = _assign_subject_splits(
        samples,
        train_ratio=train_frac,
        val_ratio=val_frac,
        subject_col="subject",
        seed=seed,
    )

    samples = samples.with_columns(
        pl.col("subject")
        .cast(pl.Utf8)
        .replace(subject_to_split)
        .alias("split")
    )

    if samples["split"].null_count() > 0:
        bad = samples.filter(pl.col("split").is_null())
        raise ValueError(
            f"Some samples could not be assigned to a split:\n{bad}"
        )

    window_rows: list[dict[str, Any]] = []
    for row in samples.iter_rows(named=True):
        window_rows.extend(
            _make_windows_for_sample(
                suid=str(row["suid"]),
                num_frames=int(row["num_frames"]),
                split=str(row["split"]),
                window_size=window_length,
                stride=stride,
            )
        )

    windows = pl.DataFrame(window_rows)

    # Store updated parquets
    samples.write_parquet(samples_path)
    windows.write_parquet(windows_path)

    print("Subject-level split complete")
    print("----------------------------")
    print(
        samples
        .group_by("split")
        .agg([
            pl.len().alias("num_samples"),
            pl.col("subject").n_unique().alias("num_subjects"),
            pl.col("dataset").n_unique().alias("num_datasets")
        ])
        .sort("split")
    )

    print()
    print("Windows")
    print("-------")
    print(
        windows
        .group_by("split")
        .len()
        .sort("split")
    )

# ================================================================================================================
# UTILITIES
# ================================================================================================================

# Used for taus
def signed_log1p_tau(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    tau_idx = CHANNELS.index("tau")
    x[:, :, tau_idx] = np.sign(x[:, :, tau_idx]) * np.log1p(
        np.abs(x[:, :, tau_idx])
    )
    return x

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi

# ================================================================================================================
# NORMALIZATION
# ================================================================================================================

def compute_and_store_normalization_stats(
    *,
    root: Path | str,
    split: str = "train",
) -> NormalizationStats:
    root = Path(root)

    samples = pl.read_parquet(_path_of_samples_index(root))
    train_samples = samples.filter(pl.col("split") == split)

    if train_samples.height == 0:
        raise ValueError(f"No samples found for split={split!r}")

    store = MotionZarrStore(root)

    first = train_samples.row(0, named=True)
    first_arr = store.get_kinematics(str(first["suid"]))
    feature_shape = first_arr.shape[1], first_arr.shape[2]

    running = RunningKinematicsStats(shape=feature_shape)
    for i, row in enumerate(train_samples.iter_rows(named=True), start=1):
        suid = str(row["suid"])
        arr = store.get_kinematics(suid)

        # Load one full sequence at a time.
        # For 17k samples, this is usually fine and avoids holding all data.
        x = np.asarray(arr[:], dtype=np.float32)

        # Apply log1p transform to tau
        x = signed_log1p_tau(x)

        running.update(x)

        if i % 100 == 0:
            print(f"[stats] processed {i}/{train_samples.height} train samples")

    stats = running.finalize()
    np.savez(
        _path_of_normalization_stats(root),
        mean=stats.mean,
        std=stats.std,
        split=split,
    )

    print(f"Wrote normalization stats: {_path_of_normalization_stats(root)}")
    return stats

# ================================================================================================================
# MAIN
# ================================================================================================================

def main():

    # 1. Load all samples into the zarr dataset
    # create_raw_motion_dataset(
    #     raw_glob="../motion-jepa/data/raw/**/*.csv",
    #     output_root="data/processed/motion",
    #     hz=100.0,
    #     chunk_length=256,
    #     overwrite=True
    # )
    
    # 2. Generate splits based on subjects
    generate_splits_and_windows(
        root="data/processed/motion",
        window_length=400, # 4s ~ 100hz
        stride=50, # 0.5 ~ 100hz
        train_frac=0.9,
        val_frac=0.1,
        seed=42,
    )

    # 3. Generate normalization
    compute_and_store_normalization_stats(
        root="data/processed/motion",
        split="train",
    )

if __name__ == "__main__":
    main()