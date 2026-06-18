from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MASK_LABELS = ["sky", "water", "bridge_building", "mountain_land"]
MASK_COLORS = {
    "sky": (88, 165, 255),
    "water": (0, 155, 150),
    "bridge_building": (225, 70, 55),
    "mountain_land": (122, 154, 72),
}


def rgb_to_hsv_array(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = rgb.astype(np.float32) / 255.0
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    maxc = array.max(axis=-1)
    minc = array.min(axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-6
    red = nonzero & (maxc == r)
    green = nonzero & (maxc == g)
    blue = nonzero & (maxc == b)
    hue[red] = ((g[red] - b[red]) / delta[red]) % 6.0
    hue[green] = ((b[green] - r[green]) / delta[green]) + 2.0
    hue[blue] = ((r[blue] - g[blue]) / delta[blue]) + 4.0
    hue /= 6.0

    sat = np.zeros_like(maxc)
    sat[maxc > 1e-6] = delta[maxc > 1e-6] / maxc[maxc > 1e-6]
    val = maxc
    return hue, sat, val


def normalized_polygon(points: list[tuple[float, float]], size: int) -> list[tuple[int, int]]:
    return [(int(round(x * (size - 1))), int(round(y * (size - 1)))) for x, y in points]


def draw_label_polygon(
    label_map: Image.Image,
    label: str,
    points: list[tuple[float, float]],
    size: int,
) -> None:
    draw = ImageDraw.Draw(label_map)
    draw.polygon(normalized_polygon(points, size), fill=MASK_LABELS.index(label))


def create_content_label_map(image: Image.Image) -> np.ndarray:
    size = image.size[0]
    label_map = Image.new("L", (size, size), MASK_LABELS.index("sky"))

    draw_label_polygon(
        label_map,
        "mountain_land",
        [
            (0.00, 0.55),
            (0.12, 0.53),
            (0.28, 0.58),
            (0.42, 0.55),
            (0.53, 0.51),
            (0.68, 0.55),
            (0.82, 0.58),
            (1.00, 0.65),
            (1.00, 0.77),
            (0.77, 0.74),
            (0.55, 0.71),
            (0.33, 0.70),
            (0.00, 0.75),
        ],
        size,
    )
    draw_label_polygon(
        label_map,
        "water",
        [
            (0.00, 0.74),
            (0.25, 0.71),
            (0.47, 0.72),
            (0.63, 0.75),
            (0.83, 0.69),
            (1.00, 0.68),
            (1.00, 1.00),
            (0.00, 1.00),
        ],
        size,
    )

    bridge = Image.new("L", (size, size), 0)
    bridge_draw = ImageDraw.Draw(bridge)
    bridge_polygons = [
        [(0.00, 0.50), (0.45, 0.55), (0.98, 0.61), (0.98, 0.64), (0.42, 0.59), (0.00, 0.54)],
        [(0.00, 0.48), (0.24, 0.48), (0.24, 1.00), (0.00, 1.00)],
        [(0.48, 0.07), (0.55, 0.07), (0.57, 0.84), (0.48, 0.84)],
        [(0.77, 0.47), (0.82, 0.47), (0.83, 0.74), (0.77, 0.74)],
        [(0.25, 0.91), (0.61, 0.90), (0.62, 1.00), (0.25, 1.00)],
    ]
    for polygon in bridge_polygons:
        bridge_draw.polygon(normalized_polygon(polygon, size), fill=255)

    line_width = max(4, size // 120)
    bridge_lines = [
        [(0.08, 0.50), (0.50, 0.09), (0.84, 0.50)],
        [(0.00, 0.55), (0.50, 0.18), (0.86, 0.58)],
        [(0.56, 0.09), (0.70, 0.35), (0.82, 0.70)],
        [(0.50, 0.12), (0.38, 0.52), (0.36, 0.60)],
    ]
    for line in bridge_lines:
        bridge_draw.line(normalized_polygon(line, size), fill=255, width=line_width)

    rgb = np.asarray(image.convert("RGB"))
    hue, sat, val = rgb_to_hsv_array(rgb)
    yy, xx = np.indices(hue.shape)
    x_norm = xx / float(size - 1)
    y_norm = yy / float(size - 1)
    bridge_zone = (
        ((y_norm < 0.56) & (x_norm < 0.86))
        | ((x_norm < 0.25) & (y_norm > 0.45))
        | ((x_norm > 0.46) & (x_norm < 0.58) & (y_norm < 0.86))
        | ((x_norm > 0.75) & (x_norm < 0.84) & (y_norm < 0.76))
        | ((x_norm < 0.62) & (y_norm > 0.88))
    )
    red_orange = (((hue < 0.09) | (hue > 0.94) | ((hue > 0.04) & (hue < 0.15))) & (sat > 0.20) & (val > 0.16))
    red_orange &= bridge_zone
    bridge_array = np.maximum(np.asarray(bridge), red_orange.astype(np.uint8) * 255)
    bridge = Image.fromarray(bridge_array, mode="L").filter(ImageFilter.MaxFilter(7))

    labels = np.asarray(label_map).copy()
    labels[np.asarray(bridge) > 0] = MASK_LABELS.index("bridge_building")
    return labels


def create_style_label_map(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    hue, sat, val = rgb_to_hsv_array(rgb)
    labels = np.full(rgb.shape[:2], MASK_LABELS.index("mountain_land"), dtype=np.uint8)

    light_background = (val > 0.76) & (sat < 0.43)
    cool_light = (hue > 0.50) & (hue < 0.66) & (sat > 0.12) & (sat < 0.65) & (val > 0.45)
    labels[light_background | cool_light] = MASK_LABELS.index("sky")

    blue_green = (hue > 0.38) & (hue < 0.70) & (sat > 0.27) & (val > 0.16)
    labels[blue_green] = MASK_LABELS.index("water")

    warm_blocks = (((hue < 0.16) | (hue > 0.90)) & (sat > 0.34) & (val > 0.18))
    black_lines = (val < 0.22) & (sat > 0.10)
    labels[warm_blocks | black_lines] = MASK_LABELS.index("bridge_building")

    return labels


def label_map_to_masks(label_map: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    for index, label in enumerate(MASK_LABELS):
        mask = (label_map == index).astype(np.float32)
        masks[label] = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    return masks


def overlay_label_map(image: Image.Image, label_map: np.ndarray, alpha: float = 0.42) -> Image.Image:
    base = image.convert("RGB")
    color = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for index, label in enumerate(MASK_LABELS):
        color[label_map == index] = MASK_COLORS[label]
    overlay = Image.fromarray(color, mode="RGB")
    return Image.blend(base, overlay, alpha)


def save_binary_masks(label_map: np.ndarray, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for index, label in enumerate(MASK_LABELS):
        mask = ((label_map == index).astype(np.uint8) * 255)
        Image.fromarray(mask, mode="L").save(prefix.parent / f"{prefix.name}_{label}.png")


def save_mask_visualization(
    content_image: Image.Image,
    style_image: Image.Image,
    content_labels: np.ndarray,
    style_labels: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content_overlay = overlay_label_map(content_image, content_labels)
    style_overlay = overlay_label_map(style_image, style_labels)
    width, height = content_overlay.size
    canvas = Image.new("RGB", (width * 2, height + 96), (255, 255, 255))
    canvas.paste(content_overlay, (0, 0))
    canvas.paste(style_overlay, (width, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, height + 10), "content semantic masks", fill=(20, 20, 20), font=font)
    draw.text((width + 12, height + 10), "style semantic masks", fill=(20, 20, 20), font=font)
    x = 12
    y = height + 42
    for label in MASK_LABELS:
        draw.rectangle((x, y, x + 22, y + 22), fill=MASK_COLORS[label])
        draw.text((x + 30, y + 4), label, fill=(20, 20, 20), font=font)
        x += 180
    canvas.save(output_path)


def build_semantic_masks(
    config: dict,
    image_size: int,
    device: torch.device,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    mask_config = config.get("semantic_masks", {})
    method = mask_config.get("method", "manual_golden_gate_kandinsky")
    if method != "manual_golden_gate_kandinsky":
        raise ValueError(f"Unsupported semantic mask method: {method}")

    size = int(image_size)
    content_image = Image.open(config["content_image"]).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    style_image = Image.open(config["style_image"]).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    content_labels = create_content_label_map(content_image)
    style_labels = create_style_label_map(style_image)

    if output_dir is not None:
        out = Path(output_dir)
        save_mask_visualization(content_image, style_image, content_labels, style_labels, out / "masks.png")
        save_binary_masks(content_labels, out / "content_mask")
        save_binary_masks(style_labels, out / "style_mask")

    stats: dict[str, float] = {}
    total = float(size * size)
    for label in MASK_LABELS:
        index = MASK_LABELS.index(label)
        stats[f"content_mask_fraction_{label}"] = float((content_labels == index).sum() / total)
        stats[f"style_mask_fraction_{label}"] = float((style_labels == index).sum() / total)

    return {
        "labels": list(MASK_LABELS),
        "content": label_map_to_masks(content_labels, device),
        "style": label_map_to_masks(style_labels, device),
        "stats": stats,
        "source": mask_config.get(
            "source",
            "Manual Golden Gate polygons plus HSV bridge extraction; Kandinsky HSV weak masks.",
        ),
    }
