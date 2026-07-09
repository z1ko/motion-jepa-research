
import sys
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

DEFAULT_CONFIG = {
    "data": {
        "root": "data/processed/motion",
        "window_size": 400,
        "stride": 50,
        "min_valid_frames": 200,
    },
    "training": {
        "batch_size": 256,
        "epochs": 600,
        "seed": 42,
        "groups": {
            "pelvis":                 [0, 1, 2, 3, 4, 5],
            "right_upper_leg":        [6, 7, 8, 9],
            "right_foot":             [10, 11, 12],
            "left_upper_leg":         [13, 14, 15, 16],
            "left_foot":              [17, 18, 19],
            "spine":                  [20, 21, 22, 23, 24, 25],
            "head":                   [26, 27, 28],
            "right_shoulder_complex": [29, 30, 31, 35, 36, 37],
            "right_lower_arm":        [41, 43, 45, 46],
            "left_shoulder_complex":  [32, 33, 34, 38, 39, 40],
            "left_lower_arm":         [42, 44, 47, 48],
        }
    },
    "architecture": {
        "embed_dim": 256,
        "segment_size": 40,
        "channels": 4,
        "encoder": {
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "heads": 8,
            "depth": 8,
        },
        "predictor": {
            "inner_dim": 128,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "heads": 8,
            "depth": 5,
        },
    },
    "optim": {
        "loss": "centered_ce",
        "ema_momentum": 0.996,
        "learning_rate": 5e-4,
        "weight_decay": 0.05,
        "tau_predict": 0.1,
        "tau_target": 0.06,
        "center_momentum": 0.9,
    }
}

def load_config(path: Path | str = "config/experiment.yaml") -> DictConfig:

    config_default = OmegaConf.create(DEFAULT_CONFIG)
    config_cli = OmegaConf.from_dotlist(sys.argv[1:])
    config_file = OmegaConf.load(path)

    cfg = OmegaConf.merge(config_default, config_file, config_cli)
    OmegaConf.resolve(cfg)
    return cfg # type: ignore

