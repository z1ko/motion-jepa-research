
from copy import deepcopy
from typing import Any

import torch as t
import torch.nn as nn

from omegaconf import DictConfig

from motion_jepa.architecture.components import TokenEmbed, _encoder
from motion_jepa.architecture.positional import PositionalEncoding
from motion_jepa.masking import MaskIndices, mask_mixed

class MotionEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.group_count = len(config.training.groups)
        self.segment_count = config.data.window_size // config.architecture.segment_size
        self.embed_dim = config.architecture.embed_dim

        self.embed = TokenEmbed(config)

        # Positional encoding
        self.pos = PositionalEncoding(
            segment_count=self.segment_count,
            group_count=self.group_count,
            embed_dim=self.embed_dim
        )

        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.encoder = _encoder(
            d_model=self.embed_dim,
            depth=config.architecture.encoder.depth,
            heads=config.architecture.encoder.heads,
            mlp_ratio=config.architecture.encoder.mlp_ratio,
            dropout=config.architecture.encoder.dropout,
        )
        

    def forward_full_and_gather(self, x: t.Tensor, idx: t.Tensor) -> t.Tensor:
        x = self.forward(x, idx=None)
        return x.gather(
            index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
            dim=1
        )

    def forward(self, x: t.Tensor, idx: t.Tensor | None = None) -> t.Tensor:
        
        tokens = self.embed(x)
        tokens = tokens.flatten(1, 2) # B, SG, E

        if idx is not None:
            # Gather only tokens present in idx
            tokens = tokens.gather(
                index=idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]), 
                dim=1
            )

        tokens = self.pos.add_flat(tokens, idx)
        x = self.encoder(tokens)
        return self.norm(x)
        

class MotionPredictor(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.segment_count = config.data.window_size // config.architecture.segment_size
        self.group_count = len(config.training.groups)
        self.pred_dim = config.architecture.predictor.inner_dim
        self.embed_dim = config.architecture.embed_dim

        self.mask_token = nn.Parameter(t.zeros(1, 1, self.pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.i_proj = nn.Linear(self.embed_dim, self.pred_dim)
        self.o_proj = nn.Linear(self.pred_dim, self.embed_dim)

        self.pos = PositionalEncoding(
            segment_count=self.segment_count,
            group_count=self.group_count,
            embed_dim=self.pred_dim
        )

        self.norm = nn.LayerNorm(self.pred_dim, eps=1e-6)
        self.encoder = _encoder(
            d_model=config.architecture.predictor.inner_dim,
            depth=config.architecture.predictor.depth, 
            heads=config.architecture.predictor.heads, 
            mlp_ratio=config.architecture.predictor.mlp_ratio, 
            dropout=config.architecture.predictor.dropout
        )

    def forward(self, context: t.Tensor, context_idx: t.Tensor, targets_idx: t.Tensor) -> t.Tensor:
        B, A = targets_idx.shape

        x_context = self.i_proj(context)
        x_context = self.pos.add_flat(x_context, context_idx)

        x_targets = self.mask_token.expand(B, A, -1)
        x_targets = self.pos.add_flat(x_targets, targets_idx)

        x = t.cat([x_context, x_targets], dim=1)
        x = self.encoder(x)

        y = self.norm(x[:, -A:, :])
        return self.o_proj(y)
    

class MotionJEPA(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.segment_count = config.data.window_size // config.architecture.segment_size
        self.group_count = len(config.training.groups)

        self.student_encoder = MotionEncoder(config)
        self.teacher_encoder = deepcopy(self.student_encoder)
        self._freeze_teacher_encoder()

        self.predictor = MotionPredictor(config)        

    def _freeze_teacher_encoder(self):
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

    @t.no_grad()
    def update_teacher(self, ema_momentum: float) -> None:
        for student_param, teacher_param in zip(
            self.student_encoder.parameters(),
            self.teacher_encoder.parameters(),
            strict=True,
        ):
            teacher_param.data.mul_(ema_momentum).add_(
                student_param.data,
                alpha=1.0 - ema_momentum,
            )

    def forward(self, x: t.Tensor, masks: MaskIndices | None = None) -> tuple[t.Tensor, t.Tensor]: 

        # Generate random masks if not provided
        batch_size = x.shape[0]
        if masks is None:
            masks = mask_mixed(
                batch_size=batch_size,
                segment_count=self.segment_count,
                group_count=self.group_count,
                device=x.device,
                targets_p=0.6
            )

        context = self.student_encoder.forward(x, masks.context) # B, SG, E
        predict = self.predictor.forward(
            context=context,
            context_idx=masks.context,
            targets_idx=masks.targets
        )

        with t.no_grad():
            targets = self.teacher_encoder.forward_full_and_gather(x, masks.targets)

        return predict, targets
