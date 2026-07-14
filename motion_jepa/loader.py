
import math
from pathlib import Path
from typing import Any

import torch as T
import lightning as L
import polars as pl
import numpy as np

from torch.utils.data import DataLoader

from motion_jepa.dataset import (
    MotionZarrStore,
    _path_of_samples_index,
    enumerate_windows,
    load_normalization_stats,
)
from motion_jepa.utils import prepare_window


class MotionWindowDataset(T.utils.data.Dataset):
    def __init__(
        self,
        root: Path | str,
        *,
        window_size: int = 400,
        segment_size: int = 40,
        stride: int,
        min_valid_frames: int,
        split: str | None = None,
        normalize: bool = True,
        clip_value: float | None = 10.0,
        dtype: np.dtype | type | str = np.float32,
    ) -> None:

        self.root = Path(root)
        self.window_size = window_size
        self.segment_size = segment_size
        self.clip_value = clip_value
        self.dtype = np.dtype(dtype)

        self._store: MotionZarrStore | None = None

        samples = pl.read_parquet(_path_of_samples_index(self.root))
        if split is not None:
            samples = samples.filter(pl.col("split") == split)

        # Deterministic sliding-window enumeration computed here, straight
        # from each trial's num_frames -- see dataset.enumerate_windows. No
        # precomputed windows.parquet needed; this is why val stays fixed
        # across epochs (see class docstring on MotionRandomCropDataset).
        self.rows: list[dict] = []
        for row in samples.select(["suid", "num_frames"]).iter_rows(named=True):
            suid = str(row["suid"])
            for start, end in enumerate_windows(
                num_frames=int(row["num_frames"]), window_size=window_size,
                stride=stride, min_valid_frames=min_valid_frames,
            ):
                self.rows.append({"suid": suid, "start": start, "end": end})

        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if normalize:
            mean, std = load_normalization_stats(self.root)
            self.mean = np.asarray(mean, dtype=np.float32)
            self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-8)

    @property
    def store(self) -> MotionZarrStore:
        if self._store is None:
            self._store = MotionZarrStore(self.root)
        return self._store

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_store"] = None
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, T.Tensor]:

        row = self.rows[index]
        x = self.store.get_kinematics_window(
            suid=str(row["suid"]),
            start=int(row["start"]),
            end=int(row["end"]),
        )

        return prepare_window(
            x,
            window_size=self.window_size,
            segment_size=self.segment_size,
            mean=self.mean,
            std=self.std,
            clip_value=self.clip_value,
            dtype=self.dtype,
        )


def _seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for `MotionRandomCropDataset`'s train DataLoader.

    When `num_workers > 0`, PyTorch creates each worker by forking the main
    process, so every worker starts with an *identical* copy of the
    dataset's `np.random.Generator` state. Without reseeding here, all
    workers would draw the exact same sequence of "random" crop offsets in
    lockstep -- silently defeating the whole point of randomizing crops.
    This runs once per worker, right after it starts, and reseeds that
    worker's copy of the dataset with a seed offset by its worker_id so the
    workers' random streams are independent of each other.
    """
    worker_info = T.utils.data.get_worker_info()
    dataset: MotionRandomCropDataset = worker_info.dataset  # type: ignore[assignment]
    dataset._rng = np.random.default_rng(dataset.seed + worker_id)


class MotionRandomCropDataset(T.utils.data.Dataset):
    """One randomly-positioned window per trial per draw, count proportional to trial length.

    `MotionWindowDataset` reads a precomputed table of sliding-window
    positions (fixed stride) from windows.parquet, so a long trial
    contributes many *overlapping* windows -- neighboring rows can share
    ~90% of their frames. Treating those as independent training samples is
    misleading (they're barely different observations) and, because long
    trials generate proportionally more of them, silently lets a handful of
    long recordings dominate a training epoch's composition.

    This class instead samples directly from samples.parquet (one row per
    trial) and draws a *fresh, randomly-offset* window each time
    `__getitem__` is called, rather than reading a fixed row. A trial
    contributes `ceil(num_frames / window_size)` draws per epoch -- roughly
    "how many non-overlapping window_size chunks would tile this trial" --
    so epoch composition scales with how much real content a trial has,
    without ever materializing (or reusing) a fixed set of overlapping
    windows. Because the offset is re-rolled on every call, the same trial
    sees a different crop epoch to epoch, so training still sees its full
    content over time even though any single epoch only takes a few crops.

    Only used for the train split. Validation stays on
    `MotionWindowDataset`'s static windows.parquet table so val loss is
    computed on the same fixed windows every epoch and is comparable across
    training -- randomizing val crops would add noise to exactly the
    numbers used to judge training progress.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        window_size: int = 400,
        segment_size: int = 40,
        split: str = "train",
        min_valid_frames: int,
        normalize: bool = True,
        clip_value: float | None = 10.0,
        dtype: np.dtype | type | str = np.float32,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.window_size = window_size
        self.segment_size = segment_size
        self.clip_value = clip_value
        self.dtype = np.dtype(dtype)
        self.seed = seed

        self._store: MotionZarrStore | None = None
        # Real RNG state lives here; overwritten per-worker by _seed_worker
        # when num_workers > 0 (see its docstring). Used as-is (single
        # process) when num_workers == 0.
        self._rng = np.random.default_rng(seed)

        samples = pl.read_parquet(_path_of_samples_index(self.root))
        samples = samples.filter(pl.col("split") == split)
        # Same floor used when preprocessing emits a padded window for a
        # short trial (motion_jepa/preprocess.py) -- below this there isn't
        # enough real signal for a meaningful context/target split.
        samples = samples.filter(pl.col("num_frames") >= min_valid_frames)

        # Flatten "trial -> crops per epoch" into one entry per crop, so
        # __len__/indexing behave like any other map-style Dataset (this is
        # what DataLoader shuffling operates over). A trial's suid appears
        # `crops` times; note this formula gives crops == 1 automatically
        # for a short trial (num_frames <= window_size), matching
        # MotionWindowDataset's "one padded window" behaviour for those,
        # with no separate branch needed.
        self.num_frames_by_suid: dict[str, int] = {}
        self.index_to_suid: list[str] = []
        for row in samples.select(["suid", "num_frames"]).iter_rows(named=True):
            suid = str(row["suid"])
            num_frames = int(row["num_frames"])
            self.num_frames_by_suid[suid] = num_frames

            crops = max(1, math.ceil(num_frames / window_size))
            self.index_to_suid.extend([suid] * crops)

        if not self.index_to_suid:
            raise ValueError(f"No trials in split={split!r} with num_frames >= {min_valid_frames}")

        self.mean: np.ndarray | None = None
        self.std:  np.ndarray | None = None
        if normalize:
            mean, std = load_normalization_stats(self.root)
            self.mean = np.asarray(mean, dtype=np.float32)
            self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-8)

    @property
    def store(self) -> MotionZarrStore:
        if self._store is None:
            self._store = MotionZarrStore(self.root)
        return self._store

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_store"] = None
        return state

    def __len__(self) -> int:
        return len(self.index_to_suid)

    def __getitem__(self, index: int) -> dict[str, T.Tensor]:
        suid = self.index_to_suid[index]
        num_frames = self.num_frames_by_suid[suid]

        if num_frames >= self.window_size:
            # Roll a fresh random start offset every call -- this is the
            # whole point: two calls for the same trial (this epoch or a
            # later one) land on different, largely non-overlapping crops
            # instead of always re-reading the same fixed window.
            max_start = num_frames - self.window_size
            start = int(self._rng.integers(0, max_start + 1))
            end = start + self.window_size
        else:
            # Short trial: only one possible crop (the whole thing), same
            # as MotionWindowDataset's short-trial handling. Padded to
            # window_size below.
            start, end = 0, num_frames

        x = self.store.get_kinematics_window(suid=suid, start=start, end=end)
        return prepare_window(
            x,
            window_size=self.window_size,
            segment_size=self.segment_size,
            mean=self.mean,
            std=self.std,
            clip_value=self.clip_value,
            dtype=self.dtype,
        )


class MotionDataset(L.LightningDataModule):
    def __init__(
        self,
        root: Path | str,
        *,
        window_size: int = 400,
        segment_size: int = 40,
        stride: int = 400,
        min_valid_frames: int = 200,
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True,
        clip_value: float | None = 10.0,
        pin_memory: bool = True,
        drop_last_train: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.clip_value = clip_value
        self.pin_memory = pin_memory
        self.drop_last_train = drop_last_train

        # Train uses randomly-positioned, length-proportional crops (see
        # MotionRandomCropDataset docstring) instead of a fixed table of
        # overlapping sliding-window positions.
        self.train = MotionRandomCropDataset(
            self.root,
            window_size=window_size,
            segment_size=segment_size,
            split="train",
            min_valid_frames=min_valid_frames,
            normalize=normalize,
            clip_value=clip_value,
            seed=seed,
        )

        # Val stays on the precomputed windows.parquet table: fixed windows
        # every epoch, so val loss is comparable across training instead of
        # being noised by a different random crop each time it's measured.
        self.val = MotionWindowDataset(
            self.root,
            window_size=window_size,
            segment_size=segment_size,
            stride=stride,
            min_valid_frames=min_valid_frames,
            split="val",
            normalize=normalize,
            clip_value=clip_value,
        )

    def _loader(
        self,
        dataset: MotionWindowDataset | MotionRandomCropDataset,
        *,
        shuffle: bool,
        drop_last: bool,
        worker_init_fn=None,
    ) -> DataLoader:
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": drop_last,
        }

        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
            if worker_init_fn is not None:
                kwargs["worker_init_fn"] = worker_init_fn

        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        # worker_init_fn reseeds each worker's copy of MotionRandomCropDataset
        # so parallel workers don't draw identical "random" crops -- see
        # _seed_worker's docstring for why this is necessary.
        return self._loader(
            self.train, drop_last=self.drop_last_train, shuffle=True, worker_init_fn=_seed_worker,
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val, drop_last=False, shuffle=False)