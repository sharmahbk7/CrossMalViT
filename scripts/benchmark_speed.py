#!/usr/bin/env python
"""Benchmark inference speed for CrossMal-ViT."""

import argparse
import time
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from crossmal_vit.models import build_crossmal_vit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CrossMal-ViT speed")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)

    model = build_crossmal_vit(OmegaConf.to_container(config.model))
    state = torch.load(args.checkpoint, map_location="cpu")
    if "model" in state:
        model.load_state_dict(state["model"], strict=False)
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dummy = {
        "raw": torch.randn(1, 1, config.model.img_size, config.model.img_size, device=device),
        "entropy": torch.randn(1, 1, config.model.img_size, config.model.img_size, device=device),
        "frequency": torch.randn(1, 1, config.model.img_size, config.model.img_size, device=device),
    }

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(args.runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

    avg_ms = (end - start) * 1000 / args.runs
    print(f"Average latency: {avg_ms:.2f} ms")


if __name__ == "__main__":
    main()
