
import datetime
from pathlib import Path

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

import torch as t
import lightning as L

from motion_jepa.config import load_config

from motion_jepa.loader import MotionDataset
from motion_jepa.jepa import MotionJEPAModule

# Config sections that affect tensor shapes -- anything else (loss, LR,
# epochs, ...) is safe to change across a resume.
_ARCHITECTURE_AFFECTING_FIELDS = ["architecture", "training.groups", "data.window_size"]


def _resolve_resume(config: DictConfig, output: Path) -> tuple[str | None, int | None]:
    """Validate a resume checkpoint and return (ckpt_path, tb_version).

    Returns (None, None) if config.resume.checkpoint is unset -- the normal,
    fresh-run path, unchanged.
    """
    checkpoint = config.resume.checkpoint
    if checkpoint is None:
        return None, None

    raw = t.load(checkpoint, map_location="cpu", weights_only=False)
    saved_config: DictConfig = raw["hyper_parameters"]

    for field in _ARCHITECTURE_AFFECTING_FIELDS:
        current = OmegaConf.select(config, field)
        saved = OmegaConf.select(saved_config, field)
        if current != saved:
            raise ValueError(
                f"Resume checkpoint {checkpoint!r} was trained with {field}={saved!r}, "
                f"but the current config has {field}={current!r}. Architecture-affecting "
                "fields must match exactly to resume -- fix the config, don't force it."
            )

    saved_epoch = raw["epoch"]
    if config.training.epochs <= saved_epoch:
        raise ValueError(
            f"Resume checkpoint {checkpoint!r} already reached epoch {saved_epoch}, but "
            f"config.training.epochs={config.training.epochs}. Bump training.epochs past "
            "that to actually continue training -- otherwise this is a silent no-op fit."
        )

    # runs/v1/version_16/checkpoints/last.ckpt -> version_16 -> 16. Reusing
    # this exact version number keeps TensorBoardLogger writing into the same
    # run directory, so every diagnostic tag stays one continuous series
    # across the resume boundary instead of splitting into an untraceable
    # new version_N.
    version_dir = Path(checkpoint).parents[1].name
    version = int(version_dir.removeprefix("version_"))

    # hparams.yaml is only written once per version dir and never updated on
    # resume (TensorBoardLogger skips it if already present), so it can't be
    # trusted to reflect a resumed run's actual config -- this log is the
    # real provenance record, and it's append-only so multi-hop resumes keep
    # their full history instead of overwriting it.
    log_path = Path(output) / "v1" / version_dir / "resume_log.txt"
    with open(log_path, "a") as f:
        f.write(
            f"{datetime.datetime.now().isoformat()} resumed_from={checkpoint} "
            f"resumed_from_epoch={saved_epoch} resumed_from_step={raw['global_step']} "
            f"target_epochs={config.training.epochs}\n"
        )

    return checkpoint, version


# Run training
def train(config: DictConfig, output: Path):
    print(config)

    ckpt_path, resume_version = _resolve_resume(config, output)

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
            version=resume_version,
        ),
    )

    # Standard dataset configuration
    dataset = MotionDataset(
        root=config.data.root,
        window_size=config.data.window_size,
        segment_size=config.architecture.segment_size,
        stride=config.data.stride,
        min_valid_frames=config.data.min_valid_frames,
        batch_size=config.training.batch_size,
        clip_value=10.0,
        normalize=True,
        seed=config.training.seed,
    )

    model = MotionJEPAModule(config)

    t.set_float32_matmul_precision('high')
    # weights_only=False: checkpoint hyperparameters are an OmegaConf
    # DictConfig, which PyTorch 2.6+'s default weights_only=True can't
    # unpickle. Same trust boundary already accepted elsewhere in this repo
    # for its own checkpoints (evaluation/encoder.py, load_from_checkpoint).
    trainer.fit(model, datamodule=dataset, ckpt_path=ckpt_path, weights_only=False)


def main():
    cfg = load_config("config/experiment.yaml")
    train(cfg, Path("runs"))

if __name__ == "__main__":
    main()
