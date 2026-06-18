from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image as torchvision_save_image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
RGB_TO_YIQ = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [0.595716, -0.274453, -0.321263],
        [0.211456, -0.522591, 0.311135],
    ]
)
YIQ_TO_RGB = torch.tensor(
    [
        [1.0, 0.9563, 0.6210],
        [1.0, -0.2721, -0.6474],
        [1.0, -1.1070, 1.7046],
    ]
)


def load_image_raw(path: str | Path, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return transform(image).unsqueeze(0).to(device)


def load_image(path: str | Path, image_size: int, device: torch.device) -> torch.Tensor:
    return normalize(load_image_raw(path, image_size, device))


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(tensor.device, tensor.dtype)
    std = IMAGENET_STD.to(tensor.device, tensor.dtype)
    return (tensor - mean) / std


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(tensor.device, tensor.dtype)
    std = IMAGENET_STD.to(tensor.device, tensor.dtype)
    return (tensor * std + mean).clamp(0.0, 1.0)


def rgb_to_yiq(rgb: torch.Tensor) -> torch.Tensor:
    matrix = RGB_TO_YIQ.to(rgb.device, rgb.dtype)
    return torch.einsum("oc,bchw->bohw", matrix, rgb)


def yiq_to_rgb(yiq: torch.Tensor) -> torch.Tensor:
    matrix = YIQ_TO_RGB.to(yiq.device, yiq.dtype)
    return torch.einsum("oc,bchw->bohw", matrix, yiq).clamp(0.0, 1.0)


def luminance_rgb(rgb: torch.Tensor) -> torch.Tensor:
    return luminance_channel_to_rgb(luminance_channel(rgb))


def luminance_channel(rgb: torch.Tensor) -> torch.Tensor:
    return rgb_to_yiq(rgb)[:, :1].clamp(0.0, 1.0)


def luminance_channel_to_rgb(luminance: torch.Tensor) -> torch.Tensor:
    return luminance.repeat(1, 3, 1, 1).clamp(0.0, 1.0)


def match_luminance_statistics(
    style_rgb: torch.Tensor,
    content_rgb: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    style_y = luminance_channel(style_rgb)
    content_y = luminance_channel(content_rgb)
    dims = (2, 3)
    style_mean = style_y.mean(dim=dims, keepdim=True)
    content_mean = content_y.mean(dim=dims, keepdim=True)
    style_std = style_y.std(dim=dims, keepdim=True, unbiased=False)
    content_std = content_y.std(dim=dims, keepdim=True, unbiased=False)
    matched = (style_y - style_mean) * (content_std / (style_std + eps)) + content_mean
    return matched.clamp(0.0, 1.0)


def chrominance_visualization(rgb: torch.Tensor) -> torch.Tensor:
    yiq = rgb_to_yiq(rgb)
    neutral_y = torch.full_like(yiq[:, :1], 0.5)
    return yiq_to_rgb(torch.cat([neutral_y, yiq[:, 1:]], dim=1))


def combine_luminance_with_chrominance(
    luminance_source_rgb: torch.Tensor,
    chrominance_source_rgb: torch.Tensor,
) -> torch.Tensor:
    y = rgb_to_yiq(luminance_source_rgb)[:, :1]
    chroma = rgb_to_yiq(chrominance_source_rgb)[:, 1:]
    return yiq_to_rgb(torch.cat([y, chroma], dim=1))


def save_raw_tensor_image(tensor: torch.Tensor, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision_save_image(tensor.detach().cpu().clamp(0.0, 1.0), output_path)
    return output_path


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision_save_image(denormalize(tensor.detach().cpu()), output_path)
    return output_path
