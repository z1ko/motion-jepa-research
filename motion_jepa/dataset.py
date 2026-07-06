
import hashlib
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import polars as pl
import torch as T
import zarr

from motion_jepa.types import MotionSample, NormalizationStats
from motion_jepa.utils import CHANNELS, JOINTS, signed_log1p_tau

def _path_of_arrays(root: Path) -> Path:
    return root / "arrays.zarr"

def _path_of_samples_index(root: Path) -> Path:
    return root / "samples.parquet"

def _path_of_windows_index(root: Path) -> Path:
    return root / "windows.parquet"

def _path_of_normalization_stats(root: Path) -> Path:
    return root / "normalization_stats.npz"

def _path_of_sample_group(suid: str) -> str:
    return f"samples/{suid}"


def stable_suid(path: Path, *, raw_root: Path | None = None) -> str:
    """
    Stable sample id. Prefer relative path if raw_root is available.
    """
    if raw_root is not None:
        try:
            key = str(path.relative_to(raw_root))
        except ValueError:
            key = str(path)
    else:
        key = str(path)

    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def choose_kinematics_chunks(
    array: np.ndarray,
    chunk_length: int,
) -> tuple[int, int, int]:
    if array.ndim != 3:
        raise ValueError(f"Expected shape (T, D, C), got {array.shape}")
    return (
        min(chunk_length, array.shape[0]),
        array.shape[1],
        array.shape[2],
    )

class MotionDatasetWriter:
    def __init__(
        self,
        root: Path | str,
        *,
        chunk_length: int = 256,
    ) -> None:
        if chunk_length <= 0:
            raise ValueError("chunk_length must be > 0")

        self.root = Path(root)
        self.chunk_length = chunk_length

    def prepare(self, *, overwrite: bool = False) -> None:
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{self.root} already exists. Pass overwrite=True to replace it."
                )
            if self.root.is_file():
                raise FileExistsError(f"{self.root} exists and is a file")
            shutil.rmtree(self.root)

        self.root.mkdir(parents=True, exist_ok=True)

        zroot = zarr.open_group(str(_path_of_arrays(self.root)), mode="w")
        zroot.attrs["format"] = "motion-zarr-v1"
        zroot.attrs["chunk_length"] = self.chunk_length

    def write_sample(
        self,
        *,
        suid: str,
        sample,
        split: str | None = None,
        overwrite: bool = False,
    ) -> dict:
        zroot = zarr.open_group(str(_path_of_arrays(self.root)), mode="a")
        group_path = _path_of_sample_group(suid)

        if group_path in zroot:
            if not overwrite:
                raise FileExistsError(f"Sample already exists: {suid}")
            del zroot[group_path]

        group = zroot.create_group(group_path)

        kinematics = np.asarray(sample.kinematics, dtype=np.float32)

        group.create_array(
            name="kinematics",
            data=kinematics,
            chunks=choose_kinematics_chunks(kinematics, self.chunk_length),
            overwrite=False,
        )

        row = {
            "suid": suid,
            "path": str(sample.path),
            "dataset": str(sample.dataset),
            "subject": str(sample.subject),
            "trial": str(sample.trial),
            "num_frames": int(kinematics.shape[0]),
            "timesteps": int(kinematics.shape[0]),
            "dofs": int(kinematics.shape[1]),
            "channels": int(kinematics.shape[2]),
        }

        if split is not None:
            row["split"] = split

        for key, value in sample.metadata.items():
            row[str(key)] = float(value)

        return row
    
class MotionZarrStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.zroot = zarr.open_group(str(_path_of_arrays(self.root)), mode="r")

    def get_kinematics(self, suid: str) -> zarr.Array:
        path = f"{_path_of_sample_group(suid)}/kinematics"
        return self.zroot[path]

    def get_kinematics_window(self, suid: str, start: int, end: int) -> np.ndarray:
        arr = self.get_kinematics(suid)[start:end, :, :]
        return arr

class RunningKinematicsStats:
    def __init__(self, shape: tuple[int, int]):
        self.total = np.zeros(shape, dtype=np.float64)
        self.total_sq = np.zeros(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray):
        self.count += int(x.shape[0])
        self.total_sq += np.square(x, dtype=np.float64).sum(axis=0)
        self.total += x.sum(axis=0, dtype=np.float64)

    def finalize(self) -> NormalizationStats:
        mean = self.total / self.count
        var = self.total_sq / self.count - np.square(mean)
        std = np.sqrt(np.clip(var, 1e-8, None))

        return NormalizationStats(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
        )
    
def load_normalization_stats(root: Path) -> tuple[np.ndarray, np.ndarray]:

    path = _path_of_normalization_stats(root)
    if not path.exists():
        raise FileNotFoundError(f"Missing normalization stats: {path}")

    data = np.load(path)
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)

    expected = (len(JOINTS), len(CHANNELS))
    if mean.shape != expected:
        raise ValueError(f"Expected mean shape {expected}, got {mean.shape}")
    if std.shape != expected:
        raise ValueError(f"Expected std shape {expected}, got {std.shape}")

    return mean, std

