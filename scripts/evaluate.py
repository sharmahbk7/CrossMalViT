#!/usr/bin/env python
"""Evaluation script for CrossMal-ViT."""

import argparse
import json
from pathlib import Path
import sys

import pytorch_lightning as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from crossmal_vit.data import MultiViewTransform
from crossmal_vit.data.datamodule import MalwareDataModule
from crossmal_vit.training import CrossMalViTModule
from crossmal_vit.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CrossMal-ViT")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./eval_metrics.json")
    parser.add_argument("--gpus", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    config = OmegaConf.load(args.config)

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

    model = CrossMalViTModule.load_from_checkpoint(
        args.checkpoint,
        model_config=OmegaConf.to_container(config.model),
        train_config=OmegaConf.to_container(config.training),
        cls_num_list=cls_num_list,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        logger=False,
    )

    results = trainer.test(model, datamodule=datamodule)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
