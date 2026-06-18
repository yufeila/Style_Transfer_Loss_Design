from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import apply_overrides, load_config
from src.masks import build_semantic_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create semantic masks for masked Gram style transfer.")
    parser.add_argument("--config", default="configs/semantic_mask_style_transfer.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), {"output_dir": args.output_dir})
    output_dir = Path(config["output_dir"])
    masks = build_semantic_masks(
        config,
        int(config["image_size"]),
        torch.device(args.device),
        output_dir,
    )
    print(f"labels: {', '.join(masks['labels'])}")
    print(f"source: {masks['source']}")
    for key, value in masks["stats"].items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
