

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from pathlib import Path

import torch as t
import lightning as L

from motion_jepa.config import load_config

from motion_jepa.loader import MotionDataset
from motion_jepa.jepa import MotionJEPAModule

# Run training
def train(config: DictConfig, output: Path):
    print(config)

    L.seed_everything(config.training.seed, workers=True)
    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=config.training.epochs,
        default_root_dir=output,
        # Default (50) is coarser than one epoch here (~72 steps/epoch at
        # batch_size=512) -- grad/norm exists specifically to catch a
        # transient spike, which epoch-level resolution would miss entirely.
        log_every_n_steps=1,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                filename="best-{epoch:04d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
            ),
            ModelCheckpoint(
                filename="periodic-{epoch:04d}",
                every_n_epochs=100,
                save_top_k=-1,
                save_last=True,
            )
        ],
        logger=TensorBoardLogger(
            save_dir=output,
            default_hp_metric=True,
            name="v1",
        ),
    )

    # Standard dataset configuration
    dataset = MotionDataset(
        root=config.data.root,
        window_size=config.data.window_size,
        segment_size=config.architecture.segment_size,
        min_valid_frames=config.data.min_valid_frames,
        batch_size=config.training.batch_size,
        clip_value=10.0,
        normalize=True,
        seed=config.training.seed,
    )

    model = MotionJEPAModule(config)

    t.set_float32_matmul_precision('high')
    trainer.fit(model, datamodule=dataset)


def main():
    cfg = load_config("config/experiment.yaml")
    train(cfg, Path("runs"))

if __name__ == "__main__":
    main()
