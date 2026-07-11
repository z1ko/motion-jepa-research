
import torch.nn.functional as f
import torch.nn as nn
import torch as t
import lightning as L

from motion_jepa.architecture.model import MotionJEPA
from motion_jepa.config import DictConfig
from motion_jepa.metrics import (
    measure_collapse,
    measure_distillation,
    measure_grad_norm,
    measure_teacher_student_similarity,
)
from motion_jepa.utils import cosine_schedule_with_warmup, ema_linear_scheduler, schedule_with_warmup


class CenteredCrossEntropyLoss(nn.Module):
    """S-JEPA's distillation loss (papers/SJepa.pdf Sec. 3 "Stabilizing
    training", Eq. 3): center + sharpen the frozen target distribution,
    cross-entropy against the predicted distribution, instead of a raw
    L2/L1 distance. Softmax makes the loss care about each token's relative
    activation across embedding dimensions rather than absolute magnitude,
    so one high-variance dimension can't dominate the way it can under
    smooth-L1 -- their own ablation found MSE costs 1.6-3.4 points vs CE on
    this architecture. `center` is a running EMA of the batch-mean target
    (DINO-style), preventing a dimension with a systematically higher
    baseline from always winning the softmax regardless of input.
    """

    def __init__(self, embed_dim: int, tau_predict: float, tau_target: float, center_momentum: float):
        super().__init__()
        self.tau_predict = tau_predict
        self.tau_target = tau_target
        self.center_momentum = center_momentum
        self.register_buffer("center", t.zeros(1, 1, embed_dim))

    def forward(self, predict: t.Tensor, targets: t.Tensor) -> tuple[t.Tensor, dict[str, t.Tensor]]:
        log_p1 = f.log_softmax(predict / self.tau_predict, dim=-1)
        with t.no_grad():
            log_p2 = f.log_softmax((targets - self.center) / self.tau_target, dim=-1)
            if self.training:
                self._update_center(targets)

        # log_p2.exp() instead of a separate f.softmax(...) call: mathematically
        # equal to softmax up to float32 noise (~1e-8, verified empirically),
        # and we need log_p2 anyway for the entropy metric below -- computing
        # it via log_softmax first is also the numerically safe direction
        # (avoids p2.log() blowing up to -inf on near-zero tail probabilities).
        # No gradient runs through this branch (no_grad above) so there's no
        # compounding-error-through-backward risk either.
        loss = -(log_p2.exp() * log_p1).sum(dim=-1).mean()

        with t.no_grad():
            aux = measure_distillation(log_p1.exp(), log_p1, log_p2.exp(), log_p2)
            aux["center_norm"] = self.center.norm()

        return loss, aux

    @t.no_grad()
    def _update_center(self, targets: t.Tensor) -> None:
        batch_center = targets.mean(dim=(0, 1), keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)


class SmoothL1Loss(nn.Module):
    """Pre-S-JEPA baseline: raw regression on embeddings, no centering/softmax.

    Wrapped to match CenteredCrossEntropyLoss's (loss, aux_metrics) interface
    so `_step` doesn't need to branch on which loss is active -- aux is empty
    since distillation/* (entropy, argmax_agreement, center_norm) don't apply
    to a regression loss. collapse/*, grad/*, and alignment/* metrics are
    loss-agnostic and still fully comparable between the two.
    """

    def forward(self, predict: t.Tensor, targets: t.Tensor) -> tuple[t.Tensor, dict[str, t.Tensor]]:
        loss = f.smooth_l1_loss(predict, targets, beta=1.0)
        return loss, {}


def _build_loss_fn(config: DictConfig) -> nn.Module:
    
    if "loss" not in config.optim:
        return CenteredCrossEntropyLoss(
            embed_dim=config.architecture.embed_dim,
            center_momentum=config.optim.center_momentum,
            tau_predict=config.optim.tau_predict,
            tau_target=config.optim.tau_target,
        )

    loss_name = config.optim.loss
    if "loss" not in config.optim or loss_name == "centered_ce":
        return CenteredCrossEntropyLoss(
            embed_dim=config.architecture.embed_dim,
            center_momentum=config.optim.center_momentum,
            tau_predict=config.optim.tau_predict,
            tau_target=config.optim.tau_target,
        )
    if loss_name == "smooth_l1":
        return SmoothL1Loss()
    
    raise ValueError(f"Unknown optim.loss: {loss_name!r} (expected 'centered_ce' or 'smooth_l1')")
    


class MotionJEPAModule(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.save_hyperparameters(config)
        self.model = MotionJEPA(config)
        self.config = config
        self.loss_fn = _build_loss_fn(config)

    # General step
    def _step(self, batch: dict[str, t.Tensor], stage: str):
        motion, valid_segments = batch["x"], batch["valid_segments"]
        B = motion.shape[0]

        # Masks are generated inside the model.
        predict, targets = self.model(motion, valid_segments=valid_segments)

        loss, distill_metrics = self.loss_fn(predict, targets)
        self.log(
            f"loss/{stage}", loss.detach().float(),
            prog_bar=True, on_step=(stage == "train"),
            on_epoch=True, batch_size=B
        )
        if stage == "val":
            # Bare key, unprefixed: train.py's ModelCheckpoint(monitor="val_loss")
            # matches this exact name, kept separate from the loss/ category.
            self.log(
                "val_loss",
                loss.detach().float(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )

        self.log_dict(
            {f"distillation/{stage}/{name}": value for name, value in distill_metrics.items()},
            on_step=False, on_epoch=True, prog_bar=False,
            batch_size=B, sync_dist=False,
        )
        self._log_collapse_metrics(targets.detach(), batch_size=B, prefix="targets", stage=stage)
        self._log_collapse_metrics(predict, batch_size=B, prefix="predict", stage=stage)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    # EMA update of the teacher, with linear ema momentum update.
    # max_steps comes from the trainer itself (accounts for actual
    # max_epochs/dataset size/accumulation), not a hardcoded config value --
    # otherwise the ramp to 1.0 never completes if the run's real step count
    # doesn't match whatever was guessed when ema_steps was set.
    #
    # on_train_batch_end, not on_before_zero_grad: the latter fires before
    # that step's backward()/optimizer.step() (verified against Lightning's
    # own hook-ordering source), so update_teacher would EMA in the
    # *previous* step's student weights, not the ones just produced by this
    # step's optimizer update. on_train_batch_end fires after the full step
    # (backward + optimizer.step() + zero_grad) completes, so self.global_step
    # is already incremented and student params already reflect this step.
    def on_train_batch_end(self, outputs, batch, batch_idx):
        max_steps = int(self.trainer.estimated_stepping_batches)
        ema = ema_linear_scheduler(max_steps, self.global_step, self.config.optim.ema_momentum)
        self.log("optim/ema_momentum", ema, prog_bar=False, on_step=False, on_epoch=True)
        self.model.update_teacher(ema)

    # Grad norm pre-clip (no clipping configured) -- fires after backward(),
    # before optimizer.step(). Directly checkable signal for a divergence
    # spike instead of inferring one from the loss curve after the fact.
    def on_before_optimizer_step(self, optimizer):
        grad_norm = measure_grad_norm(self.model.parameters())
        self.log("grad/norm", grad_norm, on_step=True, on_epoch=True, prog_bar=False)

    # Student/teacher drift: near 1 means EMA momentum is too low for
    # distillation to matter; steadily falling apart is expected/healthy.
    def on_train_epoch_end(self):
        similarity = measure_teacher_student_similarity(
            self.model.student_encoder.parameters(),
            self.model.teacher_encoder.parameters(),
        )
        self.log(
            "alignment/teacher_student_cosine_sim", similarity,
            on_step=False, on_epoch=True, prog_bar=False,
            batch_size=1, sync_dist=False,
        )

    def _log_collapse_metrics(self, z: t.Tensor, batch_size: int, prefix: str, stage: str):
        metrics = measure_collapse(z)
        self.log_dict(
            {
                f"collapse/{stage}/{prefix}_{name}" : value
                for name, value in metrics.items()
            },
            on_step=False, on_epoch=True, prog_bar=False,
            batch_size=batch_size,
            sync_dist=False,
        )

    # We use AdamW
    def configure_optimizers(self): # type: ignore
        optimizer = t.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            weight_decay = self.config.optim.weight_decay,
            lr = self.config.optim.learning_rate,
            betas=(0.9, 0.95),
        )

        # 0.0 -> lr -> lr/2. num_training_steps comes from the trainer itself
        # (see on_before_zero_grad's ema note) -- a hardcoded constant here
        # silently decouples the decay curve from the run's actual length.
        scheduler = cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=2000,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
            eta_min_fraction=0.5
        )

        #scheduler = schedule_with_warmup(optimizer=optimizer, num_warmup_steps=2000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
