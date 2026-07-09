
import torch.nn.functional as f
import torch.nn as nn
import torch as t
import lightning as L

from motion_jepa.architecture.model import MotionJEPA
from motion_jepa.config import DictConfig
from motion_jepa.metrics import measure_collapse
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

    def forward(self, predict: t.Tensor, targets: t.Tensor) -> t.Tensor:
        log_p1 = f.log_softmax(predict / self.tau_predict, dim=-1)
        with t.no_grad():
            p2 = f.softmax((targets - self.center) / self.tau_target, dim=-1)
            if self.training:
                self._update_center(targets)

        return -(p2 * log_p1).sum(dim=-1).mean()

    @t.no_grad()
    def _update_center(self, targets: t.Tensor) -> None:
        batch_center = targets.mean(dim=(0, 1), keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)


class MotionJEPAModule(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.save_hyperparameters(config)
        self.model = MotionJEPA(config)
        self.config = config
        self.loss_fn = CenteredCrossEntropyLoss(
            embed_dim=config.architecture.embed_dim,
            center_momentum=config.optim.center_momentum,
            tau_predict=config.optim.tau_predict,
            tau_target=config.optim.tau_target,
        )

    # General step
    def _step(self, batch: dict[str, t.Tensor], stage: str):
        motion, valid_segments = batch["x"], batch["valid_segments"]
        B = motion.shape[0]

        # Masks are generated inside the model.
        predict, targets = self.model(motion, valid_segments=valid_segments)

        loss = self.loss_fn(predict, targets)
        self.log(
            f"{stage}/loss", loss.detach().float(), 
            prog_bar=True, on_step=(stage == "train"), 
            on_epoch=True, batch_size=B
        )
        if stage == "val":
            self.log(
                "val_loss",
                loss.detach().float(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=B,
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
    def on_before_zero_grad(self, optimizer):
        max_steps = int(self.trainer.estimated_stepping_batches)
        ema = ema_linear_scheduler(max_steps, self.global_step, self.config.optim.ema_momentum)
        self.log("ema_momentum", ema, prog_bar=False, on_step=False, on_epoch=True)
        self.model.update_teacher(ema)

    def _log_collapse_metrics(self, z: t.Tensor, batch_size: int, prefix: str, stage: str):
        metrics = measure_collapse(z)
        self.log_dict(
            {
                f"{stage}/{prefix}_{name}" : value
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
