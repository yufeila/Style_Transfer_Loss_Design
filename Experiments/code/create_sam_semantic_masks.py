from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from segment_anything import SamPredictor, sam_model_registry

LABELS = ["sky", "water", "bridge", "building", "mountain_land"]
COLORS = {
    "sky": (88, 165, 255),
    "water": (0, 155, 150),
    "bridge": (225, 70, 55),
    "building": (225, 170, 80),
    "mountain_land": (122, 154, 72),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create prompt-guided SAM semantic masks.")
    parser.add_argument("--content-image", required=True)
    parser.add_argument("--style-image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_square_image(path: str | Path, image_size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)


def scale_box(box: tuple[float, float, float, float], size: int) -> np.ndarray:
    return np.array([box[0] * size, box[1] * size, box[2] * size, box[3] * size], dtype=np.float32)


def scale_points(points: list[tuple[float, float]], size: int) -> np.ndarray:
    return np.array([(x * size, y * size) for x, y in points], dtype=np.float32)


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
    return hue, sat, maxc


def prompt_mask(
    predictor: SamPredictor,
    size: int,
    box: tuple[float, float, float, float],
    positive: list[tuple[float, float]],
    negative: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    negative = negative or []
    point_coords = scale_points(positive + negative, size)
    point_labels = np.array([1] * len(positive) + [0] * len(negative), dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords if len(point_coords) else None,
        point_labels=point_labels if len(point_coords) else None,
        box=scale_box(box, size),
        multimask_output=True,
    )
    return masks[int(np.argmax(scores))]


def union_prompts(
    predictor: SamPredictor,
    size: int,
    prompts: list[dict[str, object]],
) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for prompt in prompts:
        mask |= prompt_mask(
            predictor,
            size,
            prompt["box"],
            prompt.get("positive", []),
            prompt.get("negative", []),
        )
    return mask


def draw_normalized_line(mask: Image.Image, points: list[tuple[float, float]], width: int) -> None:
    size = mask.size[0]
    draw = ImageDraw.Draw(mask)
    draw.line([(int(x * size), int(y * size)) for x, y in points], fill=255, width=width)


def bridge_structure_constraint(image: Image.Image, sam_bridge: np.ndarray) -> np.ndarray:
    size = image.size[0]
    rgb = np.asarray(image.convert("RGB"))
    hue, sat, val = rgb_to_hsv_array(rgb)
    yy, xx = np.indices(hue.shape)
    x = xx / float(size - 1)
    y = yy / float(size - 1)

    red_orange = (((hue < 0.085) | (hue > 0.95) | ((hue > 0.04) & (hue < 0.16))) & (sat > 0.18) & (val > 0.15))
    structure_zone = (
        ((y > 0.45) & (y < 0.66))
        | ((x > 0.46) & (x < 0.58) & (y < 0.86))
        | ((x > 0.75) & (x < 0.84) & (y > 0.42) & (y < 0.78))
        | ((x < 0.24) & (y > 0.42) & (y < 0.88))
    )

    cable_zone = Image.new("L", (size, size), 0)
    line_width = max(4, size // 180)
    draw_normalized_line(cable_zone, [(0.06, 0.52), (0.50, 0.08), (0.84, 0.52)], line_width)
    draw_normalized_line(cable_zone, [(0.00, 0.55), (0.50, 0.18), (0.86, 0.59)], line_width)
    draw_normalized_line(cable_zone, [(0.55, 0.10), (0.68, 0.35), (0.82, 0.70)], line_width)
    draw_normalized_line(cable_zone, [(0.50, 0.13), (0.38, 0.52), (0.36, 0.60)], line_width)
    cable = np.asarray(cable_zone) > 0

    color_bridge = red_orange & (structure_zone | cable)
    color_bridge = np.asarray(Image.fromarray(color_bridge.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0
    sam_limited = sam_bridge & (structure_zone | cable) & color_bridge
    return color_bridge | sam_limited


def fill_remaining(label_map: np.ndarray) -> np.ndarray:
    size = label_map.shape[0]
    yy = np.indices(label_map.shape)[0] / float(size)
    empty = label_map < 0
    label_map[empty & (yy < 0.50)] = LABELS.index("sky")
    label_map[empty & (yy > 0.70)] = LABELS.index("water")
    label_map[label_map < 0] = LABELS.index("mountain_land")
    return label_map


def disjoint_label_map(raw_masks: dict[str, np.ndarray], priority: list[str]) -> np.ndarray:
    first_mask = next(iter(raw_masks.values()))
    label_map = np.full(first_mask.shape, -1, dtype=np.int16)
    assigned = np.zeros(first_mask.shape, dtype=bool)
    for label in priority:
        current = raw_masks[label] & ~assigned
        label_map[current] = LABELS.index(label)
        assigned |= current
    return fill_remaining(label_map).astype(np.uint8)


def content_horizon_mask(size: int) -> np.ndarray:
    points = [
        (0.00, 0.46),
        (0.10, 0.48),
        (0.22, 0.51),
        (0.35, 0.50),
        (0.48, 0.46),
        (0.58, 0.48),
        (0.66, 0.52),
        (0.73, 0.50),
        (0.82, 0.56),
        (0.92, 0.62),
        (1.00, 0.66),
        (1.00, 0.78),
        (0.00, 0.78),
    ]
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(int(x * size), int(y * size)) for x, y in points], fill=255)
    return np.asarray(mask) > 0


def content_water_region_mask(size: int) -> np.ndarray:
    points = [
        (0.00, 0.68),
        (0.16, 0.68),
        (0.34, 0.67),
        (0.52, 0.68),
        (0.70, 0.69),
        (0.86, 0.66),
        (1.00, 0.65),
        (1.00, 1.00),
        (0.00, 1.00),
    ]
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(int(x * size), int(y * size)) for x, y in points], fill=255)
    return np.asarray(mask) > 0


def force_content_structures(label_map: np.ndarray) -> np.ndarray:
    size = label_map.shape[0]
    yy, xx = np.indices(label_map.shape)
    x = xx / float(size - 1)
    y = yy / float(size - 1)
    bridge_idx = LABELS.index("bridge")
    building_idx = LABELS.index("building")
    mountain_idx = LABELS.index("mountain_land")
    water_idx = LABELS.index("water")
    sky_idx = LABELS.index("sky")

    pier = (x > 0.135) & (x < 0.175) & (y > 0.43) & (y < 0.88)
    label_map[pier] = bridge_idx

    fort_building = (x > 0.06) & (x < 0.36) & (y > 0.80) & (y < 0.97)
    label_map[fort_building] = building_idx

    water_region = content_water_region_mask(size)
    horizon_region = content_horizon_mask(size)
    label_map[(label_map == mountain_idx) & ~horizon_region] = sky_idx
    label_map[(label_map == mountain_idx) & water_region] = water_idx

    # The bridge mask is intentionally thin; remove isolated red spillover on the hillside/water.
    main_deck = (y > 0.47) & (y < 0.61)
    towers = (
        ((x > 0.47) & (x < 0.58) & (y > 0.05) & (y < 0.84))
        | ((x > 0.76) & (x < 0.84) & (y > 0.43) & (y < 0.76))
        | ((x < 0.24) & (y > 0.42) & (y < 0.90))
        | pier
    )
    cable_band = (y < 0.50) & (x < 0.90)
    allowed_bridge = main_deck | towers | cable_band
    bridge_spill = (label_map == bridge_idx) & ~allowed_bridge
    label_map[bridge_spill & water_region] = water_idx
    label_map[bridge_spill & horizon_region & ~water_region] = mountain_idx
    label_map[bridge_spill & ~horizon_region & ~water_region] = sky_idx

    label_map[pier] = bridge_idx
    label_map[fort_building] = building_idx
    return label_map


def save_binary_masks(label_map: np.ndarray, prefix: Path) -> None:
    for index, label in enumerate(LABELS):
        mask = ((label_map == index).astype(np.uint8) * 255)
        Image.fromarray(mask, mode="L").save(prefix.parent / f"{prefix.name}_{label}.png")


def overlay(image: Image.Image, label_map: np.ndarray, alpha: float = 0.42) -> Image.Image:
    colors = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for index, label in enumerate(LABELS):
        colors[label_map == index] = COLORS[label]
    return Image.blend(image, Image.fromarray(colors, mode="RGB"), alpha)


def save_overview(
    content_image: Image.Image,
    style_image: Image.Image,
    content_labels: np.ndarray,
    style_labels: np.ndarray,
    output_path: Path,
) -> None:
    content_overlay = overlay(content_image, content_labels)
    style_overlay = overlay(style_image, style_labels)
    width, height = content_overlay.size
    canvas = Image.new("RGB", (width * 2, height + 104), (255, 255, 255))
    canvas.paste(content_overlay, (0, 0))
    canvas.paste(style_overlay, (width, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, height + 10), "content SAM semantic masks", fill=(20, 20, 20), font=font)
    draw.text((width + 12, height + 10), "style SAM-guided style regions", fill=(20, 20, 20), font=font)
    x = 12
    y = height + 44
    for label in LABELS:
        draw.rectangle((x, y, x + 22, y + 22), fill=COLORS[label])
        draw.text((x + 30, y + 4), label, fill=(20, 20, 20), font=font)
        x += 160
    canvas.save(output_path)


def build_content_masks(predictor: SamPredictor, image: Image.Image, size: int) -> np.ndarray:
    prompts = {
        "sky": [
            {
                "box": (0.00, 0.00, 1.00, 0.56),
                "positive": [(0.15, 0.13), (0.35, 0.18), (0.73, 0.18), (0.92, 0.30)],
                "negative": [(0.51, 0.21), (0.35, 0.53), (0.80, 0.57)],
            }
        ],
        "water": [
            {
                "box": (0.00, 0.64, 1.00, 1.00),
                "positive": [(0.70, 0.80), (0.90, 0.78), (0.42, 0.76)],
                "negative": [(0.52, 0.70), (0.18, 0.88), (0.55, 0.58)],
            }
        ],
        "bridge": [
            {
                "box": (0.00, 0.43, 1.00, 0.67),
                "positive": [(0.18, 0.52), (0.46, 0.56), (0.76, 0.60)],
                "negative": [(0.55, 0.72), (0.50, 0.40)],
            },
            {
                "box": (0.44, 0.04, 0.60, 0.86),
                "positive": [(0.51, 0.16), (0.52, 0.42), (0.52, 0.70)],
                "negative": [(0.42, 0.40), (0.65, 0.55)],
            },
            {
                "box": (0.72, 0.43, 0.86, 0.77),
                "positive": [(0.79, 0.52), (0.80, 0.66)],
                "negative": [(0.72, 0.70), (0.90, 0.62)],
            },
            {
                "box": (0.00, 0.03, 0.90, 0.62),
                "positive": [(0.25, 0.37), (0.40, 0.24), (0.62, 0.39), (0.79, 0.51)],
                "negative": [(0.20, 0.15), (0.55, 0.75)],
            },
        ],
        "building": [
            {
                "box": (0.00, 0.78, 0.48, 1.00),
                "positive": [(0.15, 0.90), (0.32, 0.94)],
                "negative": [(0.12, 0.65), (0.55, 0.88)],
            }
        ],
        "mountain_land": [
            {
                "box": (0.10, 0.50, 1.00, 0.76),
                "positive": [(0.30, 0.62), (0.58, 0.62), (0.86, 0.64)],
                "negative": [(0.50, 0.77), (0.53, 0.50)],
            }
        ],
    }
    raw_masks = {label: union_prompts(predictor, size, label_prompts) for label, label_prompts in prompts.items()}
    raw_masks["bridge"] = bridge_structure_constraint(image, raw_masks["bridge"])
    label_map = disjoint_label_map(raw_masks, ["building", "bridge", "water", "mountain_land", "sky"])
    return force_content_structures(label_map)


def build_style_masks(predictor: SamPredictor, size: int) -> np.ndarray:
    prompts = {
        "sky": [
            {
                "box": (0.28, 0.05, 0.88, 0.70),
                "positive": [(0.58, 0.35), (0.70, 0.45)],
                "negative": [(0.48, 0.30), (0.80, 0.18)],
            }
        ],
        "water": [
            {
                "box": (0.18, 0.20, 0.72, 0.90),
                "positive": [(0.35, 0.42), (0.55, 0.72), (0.70, 0.65)],
                "negative": [(0.50, 0.28), (0.18, 0.76)],
            }
        ],
        "bridge": [
            {
                "box": (0.00, 0.05, 0.95, 0.75),
                "positive": [(0.33, 0.47), (0.47, 0.52), (0.72, 0.33)],
                "negative": [(0.58, 0.35), (0.78, 0.62)],
            }
        ],
        "building": [
            {
                "box": (0.00, 0.00, 0.45, 1.00),
                "positive": [(0.10, 0.35), (0.22, 0.78)],
                "negative": [(0.35, 0.48), (0.60, 0.75)],
            }
        ],
        "mountain_land": [
            {
                "box": (0.25, 0.35, 1.00, 1.00),
                "positive": [(0.65, 0.78), (0.88, 0.86)],
                "negative": [(0.50, 0.72), (0.75, 0.35)],
            }
        ],
    }
    raw_masks = {label: union_prompts(predictor, size, label_prompts) for label, label_prompts in prompts.items()}
    return disjoint_label_map(raw_masks, ["bridge", "building", "water", "mountain_land", "sky"])


def mask_stats(label_map: np.ndarray, prefix: str) -> dict[str, float]:
    total = float(label_map.size)
    return {
        f"{prefix}_mask_fraction_{label}": float((label_map == index).sum() / total)
        for index, label in enumerate(LABELS)
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    predictor = SamPredictor(sam)

    content = load_square_image(args.content_image, args.image_size)
    style = load_square_image(args.style_image, args.image_size)

    predictor.set_image(np.asarray(content))
    content_labels = build_content_masks(predictor, content, args.image_size)
    predictor.set_image(np.asarray(style))
    style_labels = build_style_masks(predictor, args.image_size)

    save_binary_masks(content_labels, output_dir / "content_sam_mask")
    save_binary_masks(style_labels, output_dir / "style_sam_mask")
    save_overview(content, style, content_labels, style_labels, output_dir / "masks.png")

    metadata = {
        "method": "prompt_guided_sam",
        "sam_checkpoint": str(Path(args.checkpoint)),
        "model_type": args.model_type,
        "labels": LABELS,
        "image_size": args.image_size,
        "note": "SAM is promptable but not semantic by itself; labels come from fixed boxes/points.",
        "stats": {
            **mask_stats(content_labels, "content"),
            **mask_stats(style_labels, "style"),
        },
    }
    with (output_dir / "sam_mask_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(json.dumps(metadata["stats"], indent=2, ensure_ascii=False))
    print(output_dir / "masks.png")


if __name__ == "__main__":
    main()
