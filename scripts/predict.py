#!/usr/bin/env python
"""Prediction script for CrossMal-ViT."""

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from crossmal_vit.data import MultiViewTransform
from crossmal_vit.models import build_crossmal_vit
from crossmal_vit.data.datasets.kaggle_malware import KaggleMalwareDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict malware family")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)

    model = build_crossmal_vit(OmegaConf.to_container(config.model))
    state = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    elif "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()

    image = Image.open(args.image).convert("L")
    transform = MultiViewTransform(img_size=config.data.img_size, entropy_window=config.data.entropy_window)
    views = transform(image)
    views = {k: v.unsqueeze(0) for k, v in views.items()}

    with torch.no_grad():
        outputs = model(views)
        probs = torch.softmax(outputs["logits"], dim=-1).squeeze(0)

    topk = min(args.topk, probs.numel())
    values, indices = torch.topk(probs, topk)

    class_names = KaggleMalwareDataset.CLASSES
    for score, idx in zip(values.tolist(), indices.tolist()):
        print(f"{class_names[idx]}: {score:.4f}")


if __name__ == "__main__":
    main()
