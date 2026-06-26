from __future__ import annotations

import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    """SENet channel attention block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.excitation(weights).view(batch, channels, 1, 1)
        return x * weights


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        avg = torch.mean(x, dim=(2, 3))
        max_values = torch.amax(x, dim=(2, 3))
        weights = self.sigmoid(self.mlp(avg) + self.mlp(max_values))
        return x * weights.view(batch, channels, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_values = torch.amax(x, dim=1, keepdim=True)
        weights = self.sigmoid(self.conv(torch.cat([avg, max_values], dim=1)))
        return x * weights


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


def make_attention(kind: str, channels: int, reduction: int = 16) -> nn.Module:
    kind = kind.lower()
    if kind == "none":
        return nn.Identity()
    if kind == "se":
        return SqueezeExcitation(channels, reduction=reduction)
    if kind == "cbam":
        return CBAM(channels, reduction=reduction)
    raise ValueError(f"Unsupported attention kind: {kind}")
