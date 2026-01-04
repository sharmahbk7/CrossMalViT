#!/usr/bin/env python
"""Preprocess raw binaries into byteplot images."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def infer_width(file_size: int) -> int:
    if file_size < 10 * 1024:
        return 32
    if file_size < 30 * 1024:
        return 64
    if file_size < 60 * 1024:
        return 128
    if file_size < 100 * 1024:
        return 256
    if file_size < 200 * 1024:
        return 384
    if file_size < 500 * 1024:
        return 512
    if file_size < 1000 * 1024:
        return 768
    return 1024


def binary_to_image(binary_path: Path, output_path: Path) -> None:
    data = binary_path.read_bytes()
    file_size = len(data)
    width = infer_width(file_size)
    array = np.frombuffer(data, dtype=np.uint8)
    height = int(np.ceil(len(array) / width))
    padded_len = height * width
    if len(array) < padded_len:
        array = np.pad(array, (0, padded_len - len(array)), mode="constant")
    image = array.reshape(height, width)
    Image.fromarray(image).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert binaries to byteplot images")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ext", type=str, default=".bin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in input_dir.rglob(f"*{args.ext}"):
        rel = path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        binary_to_image(path, out_path)

    print(f"Processed binaries from {input_dir} to {output_dir}")


if __name__ == "__main__":
    main()
