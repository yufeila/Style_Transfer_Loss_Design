from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torchvision.models import vgg19

try:
    from torchvision.models import VGG19_Weights
except ImportError:  # pragma: no cover - compatibility with old torchvision
    VGG19_Weights = None


VGG19_LAYER_NAMES = {
    "conv1_1": 0,
    "relu1_1": 1,
    "conv1_2": 2,
    "relu1_2": 3,
    "pool1": 4,
    "conv2_1": 5,
    "relu2_1": 6,
    "conv2_2": 7,
    "relu2_2": 8,
    "pool2": 9,
    "conv3_1": 10,
    "relu3_1": 11,
    "conv3_2": 12,
    "relu3_2": 13,
    "conv3_3": 14,
    "relu3_3": 15,
    "conv3_4": 16,
    "relu3_4": 17,
    "pool3": 18,
    "conv4_1": 19,
    "relu4_1": 20,
    "conv4_2": 21,
    "relu4_2": 22,
    "conv4_3": 23,
    "relu4_3": 24,
    "conv4_4": 25,
    "relu4_4": 26,
    "pool4": 27,
    "conv5_1": 28,
    "relu5_1": 29,
    "conv5_2": 30,
    "relu5_2": 31,
    "conv5_3": 32,
    "relu5_3": 33,
    "conv5_4": 34,
    "relu5_4": 35,
}


class VGG19FeatureExtractor(nn.Module):
    def __init__(self, layers: Iterable[str], pooling: str = "max") -> None:
        super().__init__()
        requested = set(layers)
        unknown = requested.difference(VGG19_LAYER_NAMES)
        if unknown:
            raise ValueError(f"Unknown VGG19 layer names: {sorted(unknown)}")
        if pooling not in {"max", "avg"}:
            raise ValueError("pooling must be either 'max' or 'avg'")

        if VGG19_Weights is None:
            features = vgg19(pretrained=True).features
        else:
            features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        if pooling == "avg":
            for idx, layer in enumerate(features):
                if isinstance(layer, nn.MaxPool2d):
                    features[idx] = nn.AvgPool2d(
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                    )

        self.features = features.eval()
        for param in self.features.parameters():
            param.requires_grad_(False)

        self.layer_indices = {name: VGG19_LAYER_NAMES[name] for name in requested}
        self.max_index = max(self.layer_indices.values())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        reverse = {idx: name for name, idx in self.layer_indices.items()}
        for idx, layer in enumerate(self.features[: self.max_index + 1]):
            x = layer(x)
            if idx in reverse:
                outputs[reverse[idx]] = x
        return outputs
