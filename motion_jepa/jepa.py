
import torch.nn.functional as f
import torch as t
import lightning as L

from motion_jepa.architecture.model import MotionJEPA
from motion_jepa.config import DictConfig
from motion_jepa.metrics import measure_collapse
from motion_jepa.utils import cosine_schedule_with_warmup, ema_linear_scheduler

class MotionJEPAModule(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.save_hyperparameters(config)
        self.model = MotionJEPA(config)
        self.config = config

    # General step
    def _step(self, batch: dict[str, t.Tensor], stage: str):
        motion, valid_segments = batch["x"], batch["valid_segments"]
        B = motion.shape[0]

        # Masks are generated inside the model.
        predict, targets = self.model(motion, valid_segments=valid_segments)

        loss = f.smooth_l1_loss(predict, targets, beta=1.0)
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

    # EMA update of the teacher, with linear ema momentum update
    def on_before_zero_grad(self, optimizer):
        ema = ema_linear_scheduler(self.config.optim.ema_steps, self.global_step, self.config.optim.ema_momentum)
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

        # 0.0 -> lr -> lr/10
        scheduler = cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=2000,
            num_training_steps=120000,
            eta_min_fraction=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
