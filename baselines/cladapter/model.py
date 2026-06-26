from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

try:
    import timm
except ImportError as exc:  # pragma: no cover - handled at runtime with a clear error.
    raise ImportError("CLAdapter experiments require timm. Install it in the kepco environment.") from exc


AdapterStyle = Literal["official", "residual"]


class ClusterAttention(nn.Module):
    """Cluster attention adapter from CLAdapter, adapted for timm feature tokens."""

    def __init__(
        self,
        dim: int,
        centers: int = 20,
        temp_dim: int = 256,
        proj_drop: float = 0.0,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(dim, centers) * 0.02)
        self.tran_ms = nn.Parameter(torch.empty(centers, temp_dim, temp_dim))
        self.down_layer = nn.Linear(dim, temp_dim)
        self.up_layer = nn.Linear(temp_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reset_parameters(identity_init=identity_init)

    def reset_parameters(self, identity_init: bool) -> None:
        nn.init.xavier_uniform_(self.down_layer.weight)
        nn.init.zeros_(self.down_layer.bias)
        nn.init.xavier_uniform_(self.up_layer.weight)
        nn.init.zeros_(self.up_layer.bias)
        if identity_init:
            eye = torch.eye(self.tran_ms.shape[-1])
            with torch.no_grad():
                self.tran_ms.copy_(eye.unsqueeze(0).repeat(self.tran_ms.shape[0], 1, 1))
                self.tran_ms.add_(torch.randn_like(self.tran_ms) * 0.001)
        else:
            nn.init.xavier_uniform_(self.tran_ms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x.mean(dim=1)
        h = self.down_layer(x)
        attn = torch.mm(F.normalize(q, dim=-1), F.normalize(self.centers, dim=0)).softmax(dim=-1)
        tm = torch.einsum("bk,kij->bij", attn, self.tran_ms)
        h = torch.bmm(h, tm)
        h = self.up_layer(h)
        return self.proj_drop(h)


class CLAdapterBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        centers: int = 20,
        temp_dim: int = 256,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        style: AdapterStyle = "residual",
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.style = style
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = ClusterAttention(
            dim=dim,
            centers=centers,
            temp_dim=temp_dim,
            proj_drop=drop,
            identity_init=identity_init,
        )
        self.ln_2 = nn.LayerNorm(dim)
        mlp_width = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(dim, mlp_width)),
                    ("gelu", nn.GELU()),
                    ("drop_1", nn.Dropout(drop)),
                    ("c_proj", nn.Linear(mlp_width, dim)),
                    ("drop_2", nn.Dropout(drop)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.ln_1(x))
        if self.style == "residual":
            x = x + h
        elif self.style == "official":
            x = h
        else:  # pragma: no cover - argparse prevents this.
            raise ValueError(f"Unsupported CLAdapter style: {self.style}")
        return x + self.mlp(self.ln_2(x))


class CLAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        centers: int = 20,
        temp_dim: int = 256,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        style: AdapterStyle = "residual",
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CLAdapterBlock(
                    dim=dim,
                    centers=centers,
                    temp_dim=temp_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    style=style,
                    identity_init=identity_init,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TimmTokenBackbone(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.num_features = int(getattr(self.backbone, "num_features"))
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if features.ndim == 4:
            return features.flatten(2).transpose(1, 2)
        if features.ndim == 3:
            return features
        if features.ndim == 2:
            return features.unsqueeze(1)
        raise ValueError(f"Unsupported feature shape from {self.backbone_name}: {tuple(features.shape)}")


class CLAdapterFeatureModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        adapter_depth: int = 1,
        centers: int = 20,
        temp_dim: int = 256,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        style: AdapterStyle = "residual",
        identity_init: bool = True,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.token_backbone = TimmTokenBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        dim = self.token_backbone.num_features
        self.adapter = CLAdapter(
            dim=dim,
            depth=adapter_depth,
            centers=centers,
            temp_dim=min(temp_dim, dim),
            mlp_ratio=mlp_ratio,
            drop=drop,
            style=style,
            identity_init=identity_init,
        )
        self.norm = nn.LayerNorm(dim)
        self.normalize_output = normalize_output

    @property
    def num_features(self) -> int:
        return self.token_backbone.num_features

    def forward_tokens(self, x: torch.Tensor, return_original: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        original = self.token_backbone.forward_tokens(x)
        adapted = self.adapter(original)
        adapted = self.norm(adapted)
        if return_original:
            return adapted, original
        return adapted

    def forward(self, x: torch.Tensor, return_original: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_original:
            adapted, original = self.forward_tokens(x, return_original=True)
            pooled = adapted.mean(dim=1)
            original_pooled = original.mean(dim=1)
            if self.normalize_output:
                pooled = F.normalize(pooled, dim=-1)
                original_pooled = F.normalize(original_pooled, dim=-1)
            return pooled, original_pooled
        tokens = self.forward_tokens(x)
        pooled = tokens.mean(dim=1)
        return F.normalize(pooled, dim=-1) if self.normalize_output else pooled


class CLAdapterClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        adapter_depth: int = 1,
        centers: int = 20,
        temp_dim: int = 256,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        style: AdapterStyle = "residual",
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.features = CLAdapterFeatureModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            adapter_depth=adapter_depth,
            centers=centers,
            temp_dim=temp_dim,
            mlp_ratio=mlp_ratio,
            drop=drop,
            style=style,
            identity_init=identity_init,
            normalize_output=False,
        )
        self.head = nn.Linear(self.features.num_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.features.forward_tokens(x)
        return self.head(tokens.mean(dim=1))

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def adapter_parameters(self) -> list[nn.Parameter]:
        return list(self.features.adapter.parameters()) + list(self.features.norm.parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.features.token_backbone.backbone.parameters())
