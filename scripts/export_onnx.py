#!/usr/bin/env python
"""Export CrossMal-ViT to ONNX."""

import argparse
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from crossmal_vit.models import build_crossmal_vit


class OnnxWrapper(torch.nn.Module):
    """Wrap CrossMal-ViT for ONNX export with tensor inputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, raw: torch.Tensor, entropy: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        outputs = self.model({"raw": raw, "entropy": entropy, "frequency": frequency})
        return outputs["logits"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CrossMal-ViT to ONNX")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="crossmal_vit.onnx")
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

    model.eval()
    wrapper = OnnxWrapper(model)

    dummy_raw = torch.randn(1, 1, config.model.img_size, config.model.img_size)
    dummy_entropy = torch.randn(1, 1, config.model.img_size, config.model.img_size)
    dummy_frequency = torch.randn(1, 1, config.model.img_size, config.model.img_size)

    torch.onnx.export(
        wrapper,
        (dummy_raw, dummy_entropy, dummy_frequency),
        args.output,
        input_names=["raw", "entropy", "frequency"],
        output_names=["logits"],
        dynamic_axes={
            "raw": {0: "batch"},
            "entropy": {0: "batch"},
            "frequency": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
