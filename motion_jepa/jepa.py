
from dataclasses import asdict

import torch as T
import torch.nn.functional as F
import lightning as L

from motion_jepa.configuration import Config
from motion_jepa.metrics import metric_collapse
from motion_jepa.utils import ema_linear_scheduler

class MotionJEPA(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()

        self.save_hyperparameters(asdict(config))
        self.config = config

        self.model = T.nn.Module()

    # General step
    def _step(self, batch, stage: str):

        motion: T.Tensor = batch["motion"]
        B, T, D, C = motion.shape

        # TODO: Generate multiple masks for the batch
        # TODO: Forward the batch to the model
        # TODO: Log metrics

        # NOTE: WE SHOULD USE SmoothL1 and not MSE, is safer! Also used in the real code of I-JEPA

        pass

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    # EMA update of the teacher, with linear ema momentum update
    def on_before_zero_grad(self, optimizer):
        ema = ema_linear_scheduler(self.trainer.max_steps, self.global_step, self.config.ema_momentum)
        self.log("ema_momentum", ema, prog_bar=False, on_step=False, on_epoch=True)
        #self.model.update_teacher(ema)

    def _log_collapse_metrics(self, z: T.Tensor, batch_size: int, prefix: str, stage: str):
        metrics = metric_collapse(z)
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
    def configure_optimizers(self):
        return T.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            weight_decay = float(self.config.weight_decay),
            lr = float(self.config.learning_rate),
            betas=(0.9, 0.95),
        )