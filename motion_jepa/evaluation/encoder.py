"""Loading a trained Motion-JEPA checkpoint and extracting pooled window embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as t
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from motion_jepa.jepa import MotionJEPAModule

# ================================================================================================================
# ENCODER LOADING
# ================================================================================================================

def load_encoder(
    *,
    checkpoint: Path | str,
    device: t.device,
) -> tuple[t.nn.Module, object]:
    """Load a trained teacher encoder, reconstructed from ITS OWN saved config.

    Deliberately does not accept a `config_path` to override the architecture:
    `config/experiment.yaml`/`config.py` can (and will) change over the life
    of the project, but a checkpoint's weights only fit the exact
    window_size/segment_size/groups/depth/etc. it was trained with. Passing
    an independently-loaded config here would silently reconstruct the wrong
    architecture the moment those two drift apart (as they did the moment
    `load_config` was fixed to actually read the yaml -- see git history).
    Lightning already stored the real config at train time via
    `save_hyperparameters`; using that is the only way this stays correct.
    """
    module = MotionJEPAModule.load_from_checkpoint(
        str(checkpoint),
        map_location=device,
        # Lightning checkpoints contain OmegaConf hyperparameters. Use only
        # checkpoints you created or otherwise trust.
        weights_only=False,
    )
    module.eval()
    module.to(device)
    return module.model.teacher_encoder, module.config


def read_checkpoint_provenance(checkpoint: Path, config) -> dict:
    """Epoch/step plus the full training config, for tracing a result back
    to exactly which run and which hyperparameters produced it.

    Read directly off the raw checkpoint dict rather than the loaded
    LightningModule, since `current_epoch`/`global_step` on a module loaded
    outside of a `Trainer` aren't reliably populated.
    """
    raw = t.load(str(checkpoint), map_location="cpu", weights_only=False)
    return {
        "checkpoint": str(checkpoint),
        "epoch": raw.get("epoch"),
        "global_step": raw.get("global_step"),
        "config": OmegaConf.to_container(config, resolve=True),
    }

# ================================================================================================================
# EMBEDDING EXTRACTION
# ================================================================================================================

@t.inference_mode()
def compute_embeddings(
    *,
    encoder: t.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: t.device,
    segment_count: int,
    group_count: int,
    pooling: str = "mean",
) -> np.ndarray:
    """Run the encoder over every window and pool its tokens into one embedding.

    Padded (short-trial) tokens are excluded from both attention, via a
    key_padding_mask, and pooling, via a masked mean -- a plain `.mean(dim=1)`
    would silently dilute embeddings for any window shorter than window_size.

    `pooling`:
      - "mean": collapse both the segment (time) and group (anatomical) axes
        into one masked mean -> one `embed_dim`-wide vector per window. This
        is the standard linear-probe embedding.
      - "per_group": collapse only the segment axis, keep the group axis ->
        one `group_count * embed_dim`-wide vector per window (each
        anatomical group's own time-pooled embedding, concatenated). Exists
        to test whether uniform mean pooling over both axes is throwing away
        signal that's spatially localized to a few joints -- e.g. a label
        that hinges on hand/head motion specifically could be diluted by
        averaging in the other 9 groups. Not a general-purpose embedding:
        `group_count`x wider, so it needs `group_count`x more probe
        training data per free parameter than "mean" does.
      - "per_segment": collapse only the group axis, keep the segment axis ->
        one `segment_count * embed_dim`-wide vector per window (each time
        segment's own across-body embedding, concatenated). The temporal
        counterpart to "per_group": tests whether averaging over time is
        throwing away signal concentrated in a specific part of the window
        (e.g. a label that hinges on the motion's onset or a single
        transition) rather than spread evenly across it. Same caveat:
        `segment_count`x wider than "mean", so more overfitting-prone with
        few training windows per fold. Fully-padded segments (past a short
        trial's `valid_segments`) pool to an all-zero slice rather than
        being dropped, so the vector stays a fixed `segment_count *
        embed_dim` width regardless of how much of the window was real.
      - "max": collapse both axes like "mean", but via an element-wise max
        over valid tokens instead of an average -> same `embed_dim`-wide
        vector, so no extra probe-overfitting cost vs. "mean". Common in the
        GNN pooling literature as a "did this pattern fire anywhere" signal,
        though that reading assumes rectified (>=0) activations where 0
        means absent; this encoder's outputs pass through LayerNorm and can
        be negative, so treat it as "a different nonlinear aggregate that
        privileges peak/salient tokens over the sustained average" rather
        than a strict presence detector. Padded tokens are masked to -inf
        before the max so they can never win it.
    """
    if pooling not in ("mean", "per_group", "per_segment", "max"):
        raise ValueError(f"Unknown pooling: {pooling!r}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    chunks: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        valid_segments = batch["valid_segments"].to(device, non_blocking=True)

        segment_valid = t.arange(segment_count, device=device).unsqueeze(0) < valid_segments.unsqueeze(1)
        token_valid = segment_valid.unsqueeze(-1).expand(-1, -1, group_count)  # (B, segment_count, group_count)
        key_padding_mask = ~token_valid.flatten(1, 2)

        tokens = encoder(x, idx=None, key_padding_mask=key_padding_mask)  # (B, segment_count * group_count, D)

        if pooling == "mean":
            valid_flat = token_valid.flatten(1, 2).float().unsqueeze(-1)
            pooled = (tokens * valid_flat).sum(dim=1) / valid_flat.sum(dim=1).clamp(min=1.0)
        elif pooling == "per_group":
            batch_size_, _, embed_dim = tokens.shape
            tokens_grouped = tokens.view(batch_size_, segment_count, group_count, embed_dim)
            valid_grouped = token_valid.float().unsqueeze(-1)  # (B, segment_count, group_count, 1)
            summed = (tokens_grouped * valid_grouped).sum(dim=1)  # (B, group_count, D) -- sum over time
            counts = valid_grouped.sum(dim=1).clamp(min=1.0)  # (B, group_count, 1)
            pooled = (summed / counts).flatten(1, 2)  # (B, group_count * D)
        elif pooling == "per_segment":
            batch_size_, _, embed_dim = tokens.shape
            tokens_grouped = tokens.view(batch_size_, segment_count, group_count, embed_dim)
            valid_grouped = token_valid.float().unsqueeze(-1)  # (B, segment_count, group_count, 1)
            summed = (tokens_grouped * valid_grouped).sum(dim=2)  # (B, segment_count, D) -- sum over groups
            counts = valid_grouped.sum(dim=2).clamp(min=1.0)  # (B, segment_count, 1)
            pooled = (summed / counts).flatten(1, 2)  # (B, segment_count * D)
        else:  # max
            valid_flat = token_valid.flatten(1, 2).unsqueeze(-1)  # (B, segment_count * group_count, 1)
            masked_tokens = tokens.masked_fill(~valid_flat, float("-inf"))
            pooled = masked_tokens.max(dim=1).values

        chunks.append(pooled.cpu().numpy())

    return np.concatenate(chunks, axis=0)
