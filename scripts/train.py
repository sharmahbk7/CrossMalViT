#!/usr/bin/env python
"""Main training script for CrossMal-ViT."""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from crossmal_vit.data import MultiViewTransform
from crossmal_vit.data.datamodule import MalwareDataModule
from crossmal_vit.training import CrossMalViTModule
from crossmal_vit.utils import set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CrossMal-ViT")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="crossmal-vit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    setup_logging()

    config = OmegaConf.load(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")

    transform = MultiViewTransform(
        img_size=config.data.img_size,
        entropy_window=config.data.entropy_window,
    )

    datamodule = MalwareDataModule(
        data_dir=args.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        img_size=config.data.img_size,
        multi_view_transform=transform,
    )
    datamodule.setup()

    cls_num_list = list(datamodule.train_dataset.class_counts.values())

    model = CrossMalViTModule(
        model_config=OmegaConf.to_container(config.model),
        train_config=OmegaConf.to_container(config.training),
        cls_num_list=cls_num_list,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="crossmal-vit-{epoch:02d}-{val/macro_f1:.4f}",
            monitor="val/macro_f1",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/macro_f1",
            mode="max",
            patience=config.training.get("patience", 15),
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if args.wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=config.get("experiment_name", "crossmal-vit"),
            save_dir=str(output_dir),
        )
    else:
        logger = TensorBoardLogger(save_dir=str(output_dir), name="tensorboard")

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        strategy="ddp" if args.gpus > 1 else "auto",
        callbacks=callbacks,
        logger=logger,
        precision="16-mixed" if config.training.get("mixed_precision", True) else 32,
        gradient_clip_val=config.training.get("gradient_clip", 1.0),
        accumulate_grad_batches=config.training.get("accumulate_grad", 1),
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print(f"Training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
