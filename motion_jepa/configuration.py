
from dataclasses import dataclass
from pathlib import Path

import yaml

@dataclass
class Config:
    batch_size: int = 100
    masks_per_sample: int = 4
    epochs: int = 100
    ema_momentum: float = 0.996
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    device: str = "auto"
    output_dir: Path = Path("runs")
    num_workers: int = 0
    seed: int = 0

def load_config(path: Path) -> Config:
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in train config: {path}")
    
    return Config(**raw)