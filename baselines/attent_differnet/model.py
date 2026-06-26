from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import AlexNet_Weights, alexnet

from .attention import make_attention
from .freia_funcs import (
    F_fully_connected,
    InputNode,
    Node,
    OutputNode,
    ReversibleGraphNet,
    glow_coupling_layer,
    permute_layer,
)


@dataclass
class DifferNetConfig:
    attention: str = "none"
    pretrained: bool = True
    img_size: int = 448
    n_scales: int = 3
    n_coupling_blocks: int = 8
    clamp_alpha: float = 3.0
    fc_internal: int = 2048
    dropout: float = 0.0
    attention_reduction: int = 16
    freeze_backbone: bool = True

    @property
    def n_features(self) -> int:
        return 256 * self.n_scales


def nf_head(config: DifferNetConfig) -> ReversibleGraphNet:
    nodes = [InputNode(config.n_features, name="input")]
    for index in range(config.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {"seed": index}, name=f"permute_{index}"))
        nodes.append(
            Node(
                [nodes[-1].out0],
                glow_coupling_layer,
                {
                    "clamp": config.clamp_alpha,
                    "F_class": F_fully_connected,
                    "F_args": {
                        "internal_size": config.fc_internal,
                        "dropout": config.dropout,
                    },
                },
                name=f"fc_{index}",
            )
        )
    nodes.append(OutputNode([nodes[-1].out0], name="output"))
    return ReversibleGraphNet(nodes)


class AlexNetFeaturesWithAttention(nn.Module):
    """AlexNet feature stack with AB1/AB2/AB3 inserted after the three pooling blocks."""

    def __init__(
        self,
        attention: str = "none",
        pretrained: bool = True,
        attention_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.attention = attention.lower()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            make_attention(self.attention, 64, reduction=attention_reduction),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            make_attention(self.attention, 192, reduction=attention_reduction),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            make_attention(self.attention, 256, reduction=attention_reduction),
        )

        if pretrained:
            self._load_alexnet_feature_weights()

    def _load_alexnet_feature_weights(self) -> None:
        weights = AlexNet_Weights.IMAGENET1K_V1
        source_features = alexnet(weights=weights).features
        source_convs = [module for module in source_features if isinstance(module, nn.Conv2d)]
        target_convs = [module for module in self.features if isinstance(module, nn.Conv2d)]
        for source, target in zip(source_convs, target_convs):
            target.load_state_dict(source.state_dict())

    def freeze_convolutions(self) -> None:
        for module in self.features:
            if isinstance(module, nn.Conv2d):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def attention_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.features:
            if not isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Identity)):
                yield from module.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class AttentDifferNet(nn.Module):
    def __init__(self, config: DifferNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or DifferNetConfig()
        self.feature_extractor = AlexNetFeaturesWithAttention(
            attention=self.config.attention,
            pretrained=self.config.pretrained,
            attention_reduction=self.config.attention_reduction,
        )
        if self.config.freeze_backbone:
            self.feature_extractor.freeze_convolutions()
        self.nf = nf_head(self.config)

    @property
    def n_features(self) -> int:
        return self.config.n_features

    def optim_parameters(self, train_backbone: bool = False) -> list[nn.Parameter]:
        if train_backbone:
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = True
        params = list(self.nf.parameters())
        params.extend(parameter for parameter in self.feature_extractor.parameters() if parameter.requires_grad)
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for scale in range(self.config.n_scales):
            if scale == 0:
                x_scaled = x
            else:
                size = self.config.img_size // (2**scale)
                x_scaled = F.interpolate(x, size=(size, size))
            feature_map = self.feature_extractor(x_scaled)
            features.append(torch.mean(feature_map, dim=(2, 3)))
        y = torch.cat(features, dim=1)
        return self.nf(y)


def get_loss(z: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
    return torch.mean(0.5 * torch.sum(z**2, dim=1) - jacobian) / z.shape[1]
