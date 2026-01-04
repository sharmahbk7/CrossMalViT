#!/usr/bin/env python
"""Run BBBC hyperparameter optimization."""

import argparse
from pathlib import Path
import sys

import pytorch_lightning as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from crossmal_vit.optimization import BBBC, SearchSpace
from crossmal_vit.data import MultiViewTransform
from crossmal_vit.data.datamodule import MalwareDataModule
from crossmal_vit.training import CrossMalViTModule
from crossmal_vit.utils import set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BBBC optimization for CrossMal-ViT")
    parser.add_argument("--config", type=str, required=True, help="Experiment config")
    parser.add_argument("--search", type=str, required=True, help="BBBC search config")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--limit_train_batches", type=float, default=0.2)
    parser.add_argument("--limit_val_batches", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    setup_logging()

    exp_config = OmegaConf.load(args.config)
    search_config = OmegaConf.load(args.search)

    params = {}
    for name, spec in search_config.search_space.items():
        low, high, ptype = spec
        params[name] = (float(low), float(high), ptype)

    search_space = SearchSpace(params)

    transform = MultiViewTransform(
        img_size=exp_config.data.img_size,
        entropy_window=exp_config.data.entropy_window,
    )

    datamodule = MalwareDataModule(
        data_dir=args.data_dir,
        batch_size=exp_config.training.batch_size,
        num_workers=exp_config.training.num_workers,
        img_size=exp_config.data.img_size,
        multi_view_transform=transform,
    )
    datamodule.setup()
    cls_num_list = list(datamodule.train_dataset.class_counts.values())

    def fitness_fn(hparams: dict) -> float:
        model_cfg = OmegaConf.to_container(exp_config.model)
        model_cfg["fusion_alpha"] = hparams.get("fusion_alpha", model_cfg.get("fusion_alpha"))
        model_cfg["cross_weights"] = {
            "raw_entropy": hparams.get("cross_weight_raw_entropy", 0.34),
            "raw_frequency": hparams.get("cross_weight_raw_frequency", 0.28),
            "entropy_frequency": hparams.get("cross_weight_entropy_frequency", 0.18),
        }

        train_cfg = OmegaConf.to_container(exp_config.training)
        train_cfg["learning_rate"] = hparams.get("learning_rate", train_cfg.get("learning_rate"))
        train_cfg["weight_decay"] = hparams.get("weight_decay", train_cfg.get("weight_decay"))
        train_cfg["lambda_contrast"] = hparams.get("lambda_contrast", train_cfg.get("lambda_contrast"))
        train_cfg["temperature"] = hparams.get("temperature", train_cfg.get("temperature"))
        train_cfg["ldam_margin"] = hparams.get("ldam_margin", train_cfg.get("ldam_margin"))

        module = CrossMalViTModule(model_config=model_cfg, train_config=train_cfg, cls_num_list=cls_num_list)

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=False,
            enable_checkpointing=False,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )
        trainer.fit(module, datamodule=datamodule)

        metric = trainer.callback_metrics.get("val/macro_f1")
        return float(metric) if metric is not None else 0.0

    optimizer = BBBC(
        search_space=search_space,
        population_size=search_config.population_size,
        max_iterations=search_config.max_iterations,
        fitness_fn=fitness_fn,
        elite_ratio=search_config.get("elite_ratio", 0.1),
        initial_std=search_config.get("initial_std", 0.3),
        std_decay=search_config.get("std_decay", 0.95),
    )

    best_config, best_fitness = optimizer.optimize()
    print("Best fitness:", best_fitness)
    print("Best config:", best_config)


if __name__ == "__main__":
    main()
