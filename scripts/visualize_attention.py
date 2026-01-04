#!/usr/bin/env python
"""Visualize attention maps for a single image."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from crossmal_vit.data import MultiViewTransform
from crossmal_vit.models import build_crossmal_vit
from crossmal_vit.visualization import extract_attention_maps, save_attention_overlays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize attention maps")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./attention_outputs")
    parser.add_argument("--layer", type=str, default="layer_4")
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

    image = Image.open(args.image).convert("L")
    transform = MultiViewTransform(img_size=config.data.img_size, entropy_window=config.data.entropy_window)
    views = transform(image)
    views = {k: v.unsqueeze(0) for k, v in views.items()}

    attention_maps = extract_attention_maps(model, views, layer_key=args.layer)
    save_attention_overlays(np.array(image), attention_maps, args.output_dir)

    print(f"Saved attention maps to {args.output_dir}")


if __name__ == "__main__":
    main()
