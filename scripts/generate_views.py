#!/usr/bin/env python
"""Generate multi-view images from byteplot images."""

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from crossmal_vit.data.transforms import MultiViewTransform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-view images")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--entropy_window", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = MultiViewTransform(img_size=args.img_size, entropy_window=args.entropy_window)

    for img_path in input_dir.rglob("*.png"):
        image = Image.open(img_path).convert("L")
        views = transform(image)

        rel = img_path.relative_to(input_dir).with_suffix("")
        for view_name, tensor in views.items():
            view_dir = output_dir / view_name / rel.parent
            view_dir.mkdir(parents=True, exist_ok=True)
            arr = (tensor.squeeze(0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(view_dir / f"{rel.name}.png")

    print(f"Saved views to {output_dir}")


if __name__ == "__main__":
    main()
