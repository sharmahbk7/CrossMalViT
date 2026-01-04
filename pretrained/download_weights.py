"""Download pretrained weights for CrossMal-ViT."""

import argparse
from pathlib import Path
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CrossMal-ViT weights")
    parser.add_argument("--url", type=str, required=True, help="Direct URL to weights")
    parser.add_argument("--output", type=str, default="crossmal_vit_best.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(__file__).parent / args.output

    with urllib.request.urlopen(args.url) as response:
        data = response.read()
    output_path.write_bytes(data)

    print(f"Saved weights to {output_path}")


if __name__ == "__main__":
    main()
