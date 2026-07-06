

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from collections.abc import Sequence
from pathlib import Path

import torch as T
import lightning as L
import argparse

from motion_jepa.configuration import Config, load_config
from motion_jepa.jepa import MotionJEPA

# Creates cli flags and parameters
def cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", type=Path, default=Path("config/training.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)

# Run training
def train(config: Config, output: Path):
    print(config)

    L.seed_everything(config.seed, workers=True)
    trainer = L.Trainer(
        accelerator=config.device,
        max_epochs=config.epochs,
        default_root_dir=config.output_dir,
        callbacks=[
            ModelCheckpoint(
                filename="best-{epoch:04d}-{val_loss_epoch:.4f}",
                monitor="val_loss_epoch",
                mode="min",
                save_top_k=1,
                save_last=False,
            ),
            ModelCheckpoint(
                filename="periodic-{epoch:04d}",
                every_n_epochs=10,
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

    model = MotionJEPA(config)
    trainer.fit(model, datamodule=None)
    pass

def main(argv: Sequence[str] | None = None) -> int:
    args = cli(argv)
    config = load_config(args.config)
    train(config, args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())