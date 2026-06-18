from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def plot_loss_curve(metrics_csv: str | Path, output_path: str | Path) -> Path:
    df = pd.read_csv(metrics_csv)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    weighted_columns = [
        "total_loss",
        "weighted_content_loss",
        "weighted_style_loss",
        "weighted_tv_loss",
    ]
    raw_columns = ["total_loss", "content_loss", "style_loss", "tv_loss"]
    columns = weighted_columns if all(col in df for col in weighted_columns) else raw_columns
    for column in columns:
        if column in df:
            plt.plot(df["step"], df[column], label=column)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    return output


def make_comparison_grid(
    image_paths: list[str | Path],
    labels: list[str],
    output_path: str | Path,
) -> Path:
    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels must have the same length")

    images = [Image.open(path).convert("RGB") for path in image_paths]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    if len(images) == 1:
        axes = [axes]
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    return output
