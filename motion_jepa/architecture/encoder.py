
from typing import Any

import torch as t
import torch.nn as nn

from omegaconf import DictConfig

from motion_jepa.architecture.components import _encoder

class MotionEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

class MotionPredictor(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.embed_dim = config.architecture.embed_dim
        self.mask_token = nn.Parameter(t.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.encoder = _encoder(
            d_model=config.architecture.embed_dim,
            depth=config.architecture.predictor.depth, 
            heads=config.architecture.predictor.heads, 
            mlp_ratio=config.architecture.predictor.mlp_ratio, 
            dropout=config.architecture.predictor.embed_dim
        )