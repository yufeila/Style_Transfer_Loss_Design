from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import apply_overrides, load_config, save_config
from src.image_io import (
    chrominance_visualization,
    combine_luminance_with_chrominance,
    denormalize,
    load_image,
    load_image_raw,
    luminance_channel,
    luminance_channel_to_rgb,
    luminance_rgb,
    match_luminance_statistics,
    normalize,
    save_raw_tensor_image,
    save_tensor_image,
)
from src.losses import content_loss, gram_matrix, masked_gram_matrix, masked_style_loss, style_loss, total_variation_loss
from src.masks import build_semantic_masks
from src.metrics import write_metrics_csv
from src.model import VGG19FeatureExtractor
from src.plotting import plot_loss_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural style transfer.")
    parser.add_argument("--config", default="configs/strong_baseline.json")
    parser.add_argument("--content-image")
    parser.add_argument("--style-image")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--optimizer", choices=["adam", "lbfgs"])
    parser.add_argument("--pooling", choices=["max", "avg"])
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--content-weight", type=float)
    parser.add_argument("--style-weight", type=float)
    parser.add_argument("--tv-weight", type=float)
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_image(config: dict, content: torch.Tensor) -> torch.Tensor:
    init = config.get("init", "content")
    if optimizes_luminance_channel(config):
        if init == "noise":
            generated = torch.rand_like(content)
        elif init == "content_noise":
            noise_std = float(config.get("init_noise_std", 0.05))
            generated = (content.clone() + noise_std * torch.randn_like(content)).clamp(0.0, 1.0)
        else:
            generated = content.clone()
    elif init == "noise":
        generated = normalize(torch.rand_like(content))
    elif init == "content_noise":
        noise_std = float(config.get("init_noise_std", 0.05))
        generated = content.clone() + noise_std * torch.randn_like(content)
    else:
        generated = content.clone()
    generated.requires_grad_(True)
    return generated


def uses_luminance_color_preservation(config: dict) -> bool:
    return config.get("color_preservation") == "luminance_only"


def optimizes_luminance_channel(config: dict) -> bool:
    return uses_luminance_color_preservation(config) and bool(
        config.get("optimize_luminance_channel", True)
    )


def generated_model_input(generated: torch.Tensor, config: dict) -> torch.Tensor:
    if optimizes_luminance_channel(config):
        return normalize(luminance_channel_to_rgb(generated))
    return generated


def generated_luminance_rgb(generated: torch.Tensor, config: dict) -> torch.Tensor:
    if optimizes_luminance_channel(config):
        return luminance_channel_to_rgb(generated)
    return luminance_rgb(denormalize(generated))


def clamp_generated(generated: torch.Tensor, config: dict) -> None:
    if optimizes_luminance_channel(config):
        generated.clamp_(0.0, 1.0)
    else:
        generated.clamp_(-3.0, 3.0)


def project_to_luminance(generated: torch.Tensor) -> None:
    raw = denormalize(generated)
    generated.copy_(normalize(luminance_rgb(raw)))


def render_output_raw(
    generated: torch.Tensor,
    content_raw: torch.Tensor | None,
    config: dict,
) -> torch.Tensor:
    if uses_luminance_color_preservation(config):
        if content_raw is None:
            raise ValueError("content_raw is required for luminance-only color preservation")
        return combine_luminance_with_chrominance(
            generated_luminance_rgb(generated, config),
            content_raw,
        )
    raw = denormalize(generated)
    return raw


def save_generated_image(
    generated: torch.Tensor,
    path: Path,
    content_raw: torch.Tensor | None,
    config: dict,
) -> Path:
    if uses_luminance_color_preservation(config):
        return save_raw_tensor_image(render_output_raw(generated, content_raw, config), path)
    return save_tensor_image(generated, path)


def add_color_metrics(
    values: dict[str, float],
    generated: torch.Tensor,
    content_raw: torch.Tensor | None,
    config: dict,
) -> None:
    if not uses_luminance_color_preservation(config) or content_raw is None:
        return
    output = render_output_raw(generated, content_raw, config)
    output_mean = output.mean(dim=(2, 3))
    content_mean = content_raw.mean(dim=(2, 3))
    output_std = output.std(dim=(2, 3), unbiased=False)
    content_std = content_raw.std(dim=(2, 3), unbiased=False)
    mean_loss = torch.mean((output_mean - content_mean) ** 2)
    std_loss = torch.mean((output_std - content_std) ** 2)
    values["color_mean_loss"] = float(mean_loss.detach().cpu())
    values["color_std_loss"] = float(std_loss.detach().cpu())
    values["color_loss"] = float((mean_loss + std_loss).detach().cpu())


def build_targets(
    extractor: VGG19FeatureExtractor,
    content: torch.Tensor,
    style: torch.Tensor,
    content_layers: list[str],
    style_layers: list[str],
    style_loss_mode: str = "global_gram",
    style_masks: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    with torch.no_grad():
        content_features = extractor(content)
        style_features = extractor(style)
        content_targets = {
            layer: value.detach()
            for layer, value in content_features.items()
            if layer in content_layers
        }
        if style_loss_mode == "masked_gram":
            if not style_masks:
                raise ValueError("style_masks are required for masked_gram style loss")
            style_targets = {
                layer: {
                    label: masked_gram_matrix(value, mask).detach()
                    for label, mask in style_masks.items()
                }
                for layer, value in style_features.items()
                if layer in style_layers
            }
        else:
            style_targets = {
                layer: gram_matrix(value).detach()
                for layer, value in style_features.items()
                if layer in style_layers
            }
    return content_targets, style_targets


def compute_losses(
    generated: torch.Tensor,
    extractor: VGG19FeatureExtractor,
    content_targets: dict[str, torch.Tensor],
    style_targets: dict,
    config: dict,
    content_style_masks: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    model_input = generated_model_input(generated, config)
    features = extractor(model_input)

    raw_content = sum(
        content_loss(features[layer], target)
        for layer, target in content_targets.items()
    )
    style_loss_mode = config.get("style_loss_mode", "global_gram")
    region_style_values: dict[str, torch.Tensor] = {}
    if style_loss_mode == "masked_gram":
        if not content_style_masks:
            raise ValueError("content_style_masks are required for masked_gram style loss")
        region_weights = config.get("semantic_masks", {}).get("region_weights", {})
        for layer, layer_targets in style_targets.items():
            for label, target in layer_targets.items():
                weight = float(region_weights.get(label, 1.0))
                current_loss = masked_style_loss(features[layer], target, content_style_masks[label]) * weight
                region_style_values[label] = region_style_values.get(
                    label,
                    torch.zeros((), device=model_input.device, dtype=model_input.dtype),
                ) + current_loss
        raw_style = sum(region_style_values.values())
    else:
        raw_style = sum(
            style_loss(features[layer], target)
            for layer, target in style_targets.items()
        )
    raw_tv = total_variation_loss(generated if optimizes_luminance_channel(config) else model_input)

    weighted_content = config["content_weight"] * raw_content
    weighted_style = config["style_weight"] * raw_style
    weighted_tv = config["tv_weight"] * raw_tv
    total = weighted_content + weighted_style + weighted_tv
    values = {
        "total_loss": float(total.detach().cpu()),
        "content_loss": float(raw_content.detach().cpu()),
        "style_loss": float(raw_style.detach().cpu()),
        "tv_loss": float(raw_tv.detach().cpu()),
        "weighted_content_loss": float(weighted_content.detach().cpu()),
        "weighted_style_loss": float(weighted_style.detach().cpu()),
        "weighted_tv_loss": float(weighted_tv.detach().cpu()),
    }
    if style_loss_mode == "masked_gram":
        values["masked_style_loss"] = values["style_loss"]
        for label, value in region_style_values.items():
            values[f"style_loss_{label}"] = float(value.detach().cpu())
    return total, values


def run_optimization(config: dict) -> list[dict[str, float]]:
    device = choose_device(config.get("device"))
    set_seed(int(config.get("seed", 42)))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    image_size = int(config["image_size"])
    content_raw = None
    if uses_luminance_color_preservation(config):
        content_raw = load_image_raw(config["content_image"], image_size, device)
        style_raw = load_image_raw(config["style_image"], image_size, device)
        content_y = luminance_channel(content_raw)
        style_y = luminance_channel(style_raw)
        if config.get("luminance_histogram_matching", False):
            style_y = match_luminance_statistics(style_raw, content_raw)
        content = normalize(luminance_channel_to_rgb(content_y))
        style = normalize(luminance_channel_to_rgb(style_y))
        save_raw_tensor_image(content_raw, output_dir / "content_color_reference.png")
        save_raw_tensor_image(
            luminance_channel_to_rgb(content_y),
            output_dir / "content_luminance_reference.png",
        )
        save_raw_tensor_image(
            chrominance_visualization(content_raw),
            output_dir / "content_chrominance_reference.png",
        )
        save_raw_tensor_image(luminance_rgb(style_raw), output_dir / "style_luminance_reference.png")
        if config.get("luminance_histogram_matching", False):
            save_raw_tensor_image(
                luminance_channel_to_rgb(style_y),
                output_dir / "style_luminance_matched_reference.png",
            )
    else:
        content = load_image(config["content_image"], image_size, device)
        style = load_image(config["style_image"], image_size, device)
    generated_seed = content_y if optimizes_luminance_channel(config) else content
    generated = initialize_image(config, generated_seed)

    content_layers = list(config["content_layers"])
    style_layers = list(config["style_layers"])
    extractor = VGG19FeatureExtractor(
        content_layers + style_layers,
        pooling=config.get("pooling", "max"),
    ).to(device)
    semantic_masks = None
    if config.get("style_loss_mode") == "masked_gram":
        semantic_masks = build_semantic_masks(config, image_size, device, output_dir)
    content_targets, style_targets = build_targets(
        extractor,
        content,
        style,
        content_layers,
        style_layers,
        style_loss_mode=config.get("style_loss_mode", "global_gram"),
        style_masks=None if semantic_masks is None else semantic_masks["style"],
    )

    history: list[dict[str, float]] = []
    steps = int(config["steps"])
    save_every = int(config.get("save_every", 0))

    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam([generated], lr=float(config["learning_rate"]))
        iterator = tqdm(range(1, steps + 1), desc=config["experiment_name"])
        for step in iterator:
            optimizer.zero_grad()
            loss, values = compute_losses(
                generated,
                extractor,
                content_targets,
                style_targets,
                config,
                None if semantic_masks is None else semantic_masks["content"],
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                clamp_generated(generated, config)
                if config.get("luminance_projection", False) and not optimizes_luminance_channel(config):
                    project_to_luminance(generated)
            add_color_metrics(values, generated, content_raw, config)
            values["step"] = step
            history.append(values)
            iterator.set_postfix(total=f"{values['total_loss']:.3g}")
            if save_every and step % save_every == 0:
                save_generated_image(
                    generated,
                    output_dir / f"step_{step:04d}.png",
                    content_raw,
                    config,
                )
    else:
        optimizer = torch.optim.LBFGS(
            [generated],
            lr=float(config["learning_rate"]),
            max_iter=1,
            line_search_fn="strong_wolfe",
        )
        iterator = tqdm(range(1, steps + 1), desc=config["experiment_name"])
        last_values: dict[str, float] = {}

        for step in iterator:
            def closure() -> torch.Tensor:
                nonlocal last_values
                optimizer.zero_grad()
                loss, values = compute_losses(
                    generated,
                    extractor,
                    content_targets,
                    style_targets,
                    config,
                    None if semantic_masks is None else semantic_masks["content"],
                )
                loss.backward()
                last_values = values
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                clamp_generated(generated, config)
                if config.get("luminance_projection", False) and not optimizes_luminance_channel(config):
                    project_to_luminance(generated)
            row = dict(last_values)
            add_color_metrics(row, generated, content_raw, config)
            row["step"] = step
            history.append(row)
            iterator.set_postfix(total=f"{row['total_loss']:.3g}")
            if save_every and step % save_every == 0:
                save_generated_image(
                    generated,
                    output_dir / f"step_{step:04d}.png",
                    content_raw,
                    config,
                )

    result_path = output_dir / "result.png"
    metrics_path = output_dir / "metrics.csv"
    loss_curve_path = output_dir / "loss_curve.png"
    if uses_luminance_color_preservation(config):
        save_raw_tensor_image(
            generated_luminance_rgb(generated, config),
            output_dir / "luminance_result.png",
        )
    save_generated_image(generated, result_path, content_raw, config)
    write_metrics_csv(history, metrics_path)
    plot_loss_curve(metrics_path, loss_curve_path)
    return history


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        "content_image": args.content_image,
        "style_image": args.style_image,
        "output_dir": args.output_dir,
        "device": args.device,
        "steps": args.steps,
        "optimizer": args.optimizer,
        "pooling": args.pooling,
        "learning_rate": args.learning_rate,
        "content_weight": args.content_weight,
        "style_weight": args.style_weight,
        "tv_weight": args.tv_weight,
    }
    config = apply_overrides(config, overrides)
    run_optimization(config)


if __name__ == "__main__":
    main()
