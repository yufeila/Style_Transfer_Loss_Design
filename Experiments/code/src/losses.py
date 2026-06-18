from __future__ import annotations

import torch
import torch.nn.functional as F


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = features.shape
    flattened = features.view(batch, channels, height * width)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    return gram / (channels * height * width)


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if mask.shape[-2:] == size:
        return mask
    return F.interpolate(mask, size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def masked_gram_matrix(features: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    batch, channels, height, width = features.shape
    layer_mask = resize_mask(mask, (height, width)).to(features.device, features.dtype)
    norm = torch.sqrt(layer_mask.pow(2).sum(dim=(2, 3), keepdim=True).clamp_min(eps))
    weighted = features * (layer_mask / norm)
    flattened = weighted.view(batch, channels, height * width)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    return gram / channels


def content_loss(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(current, target)


def style_loss(current: torch.Tensor, target_gram: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(gram_matrix(current), target_gram)


def masked_style_loss(
    current: torch.Tensor,
    target_gram: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(masked_gram_matrix(current, mask), target_gram)


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    horizontal = torch.pow(image[:, :, :, 1:] - image[:, :, :, :-1], 2).mean()
    vertical = torch.pow(image[:, :, 1:, :] - image[:, :, :-1, :], 2).mean()
    return horizontal + vertical
