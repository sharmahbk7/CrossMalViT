#!/usr/bin/env python
"""Reproduce all paper results."""

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    "main": {
        "config": "configs/experiment/main_experiment.yaml",
        "description": "Main CrossMal-ViT results (Table 2)",
    },
    "ablation_views": {
        "config": "configs/experiment/ablation_views.yaml",
        "description": "View ablation study (Table 3)",
    },
    "ablation_fusion": {
        "config": "configs/experiment/ablation_fusion.yaml",
        "description": "Fusion strategy ablation (Table 4)",
    },
    "ablation_loss": {
        "config": "configs/experiment/ablation_loss.yaml",
        "description": "Loss function ablation (Table 5)",
    },
    "bbbc_optimization": {
        "config": "configs/optimizer/bbbc_default.yaml",
        "description": "BBBC hyperparameter optimization (Table 7)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce paper results")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./paper_results")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_experiment(name: str, config: str, args: argparse.Namespace) -> None:
    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(f"Description: {EXPERIMENTS[name]['description']}")
    print(f"{'=' * 60}\n")

    output_dir = Path(args.output_dir) / name

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        config,
        "--data_dir",
        args.data_dir,
        "--output_dir",
        str(output_dir),
        "--gpus",
        str(args.gpus),
        "--seed",
        str(args.seed),
    ]

    subprocess.run(cmd, check=True)
    print(f"\nDone: {name}. Results in {output_dir}")


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    experiments = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    for exp_name in experiments:
        config = EXPERIMENTS[exp_name]["config"]
        run_experiment(exp_name, config, args)

    print(f"\n{'=' * 60}")
    print("All experiments complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
