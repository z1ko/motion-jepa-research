
from pathlib import Path
from typing import Any

import torch as T
import lightning as L
import polars as pl
import numpy as np

from torch.utils.data import DataLoader

from motion_jepa.dataset import MotionZarrStore, _path_of_windows_index, load_normalization_stats
from motion_jepa.utils import signed_log1p_tau


class MotionWindowDataset(T.utils.data.Dataset):
    def __init__(
        self,
        root: Path | str,
        *,
        split: str | None = None,
        normalize: bool = True,
        clip_value: float | None = 10.0,
        dtype: np.dtype | type | str = np.float32,
        return_metadata: bool = False
    ) -> None:
        
        self.root = Path(root)
        self.clip_value = clip_value
        self.dtype = np.dtype(dtype)
        self.metadata = return_metadata

        self._store: MotionZarrStore | None = None

        windows = pl.read_parquet(_path_of_windows_index(self.root))
        if split is not None:
            windows = windows.filter(pl.col("split") == split)

        self.rows = windows.to_dicts()
        if normalize:
            mean, std = load_normalization_stats(self.root)
            self.mean = np.asarray(mean, dtype=np.float32)
            self.std  = np.asarray(std, dtype=np.float32)
            self.std  = np.maximum(self.std, 1e-8)

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
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return x

        # Same transform used when computing normalization stats.
        x = signed_log1p_tau(x)
        # Dataset-wide z-score.
        x = (x - self.mean) / self.std
        # Prevent rare acc/vel spikes from dominating loss.
        if self.clip_value is not None:
            x = np.clip(x, -self.clip_value, self.clip_value)

        return x

    def __getitem__(self, index: int) -> T.Tensor:

        row = self.rows[index]
        x = self.store.get_kinematics_window(
            suid=str(row["suid"]),
            start=int(row["start"]),
            end=int(row["end"]),
        )

        x = np.asarray(x, dtype=self.dtype)
        x = self._normalize(x)

        # Important: np.clip can sometimes return non-contiguous views.
        x_tensor = T.as_tensor(np.ascontiguousarray(x), dtype=T.float32)
        return x_tensor
    
class MotionDataset(L.LightningDataModule):
    def __init__(
        self,
        root: Path | str,
        *,
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True,
        clip_value: float | None = 10.0,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
        drop_last_train: bool = True,
    ):
        super().__init__()

        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.clip_value = clip_value
        self.pin_memory = pin_memory
        self.drop_last_train = drop_last_train

        self.train = MotionWindowDataset(
            self.root,
            split="train",
            normalize=normalize,
            clip_value=clip_value,
        )

        self.val = MotionWindowDataset(
            self.root,
            split="val",
            normalize=normalize,
            clip_value=clip_value,
        )

    def _loader(
        self,
        dataset: MotionWindowDataset,
        *,
        shuffle: bool,
        drop_last: bool,
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

        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train, drop_last=self.drop_last_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val, drop_last=False, shuffle=False)
    

#dm = MotionDataset(
#    root="data/processed/motion",
#    batch_size=4,
#    num_workers=0,
#)
#
#dm.setup("fit")
#
#batch = next(iter(dm.train_dataloader()))
#print(batch.shape)