
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

@dataclass
class MotionSample:
    kinematics: np.ndarray      # (T, D=len(JOINTS), C=len(CHANNELS)), float32
    metadata: dict[str, Any]    # subject_mass_kg, subject_height_m
    extra: pl.DataFrame         # (T, ...) time, grf, action

    path:    str
    dataset: str
    subject: str
    trial:   str

@dataclass
class NormalizationStats:
    mean: np.ndarray            # (D=len(JOINTS), C=len(CHANNELS))
    std:  np.ndarray            # (D=len(JOINTS), C=len(CHANNELS))