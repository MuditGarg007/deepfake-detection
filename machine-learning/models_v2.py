from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONES: dict[str, tuple[str, int]] = {
    "efficientnet_b0": ("efficientnet_b0", 224),
    "xception": ("legacy_xception", 224),
    "clip_vit_b16": ("vit_base_patch16_clip_224.openai", 224),
    "clip_vit_l14": ("vit_large_patch14_clip_224.openai", 224),
    "clip_vit_l14_datacomp": ("vit_large_patch14_clip_224.datacompxl", 224),
    "dinov2_vit_b14": ("vit_base_patch14_dinov2.lvd142m", 224),
    "dinov3_vit_b16": ("vit_base_patch16_dinov3.lvd1689m", 224),
    "dinov3_vit_l16": ("vit_large_patch16_dinov3.lvd1689m", 224),
    "convnext_base_clip": ("convnext_base.clip_laion2b_augreg", 224),
}

TUNE_MODES = ("full", "ln", "head")


def _is_norm_param(name: str) -> bool:
    lowered = name.lower()
    return any(key in lowered for key in ("norm", ".ln", "bn."))


class HypersphericalHead(nn.Module):

    def __init__(self, in_features: int, scale: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(1, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.log_scale = nn.Parameter(torch.tensor(float(math.log(scale))))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(F.normalize(features.float(), dim=-1))
        weight = F.normalize(self.weight.float(), dim=-1)
        cosine = F.linear(features, weight)
        scale = self.log_scale.clamp(min=0.0, max=math.log(64.0)).exp()
        return cosine * scale + self.bias


class MLPHead(nn.Module):

    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float())


class Detector(nn.Module):

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        tune: str = "ln",
        head: str = "hypersphere",
        dropout: float = 0.1,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        if name not in BACKBONES:
            raise ValueError(f"unknown backbone '{name}' — choose from {sorted(BACKBONES)}")
        if tune not in TUNE_MODES:
            raise ValueError(f"unknown tune mode '{tune}' — choose from {TUNE_MODES}")

        model_id, self.input_size = BACKBONES[name]
        self.name = name
        self.tune = tune
        self.backbone = timm.create_model(model_id, pretrained=pretrained, num_classes=0)
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)

        features = self.backbone.num_features
        self.head = (
            HypersphericalHead(features, dropout=dropout)
            if head == "hypersphere"
            else MLPHead(features, dropout=dropout)
        )
        self._apply_tune_mode()

    def _apply_tune_mode(self) -> None:
        if self.tune == "full":
            for param in self.backbone.parameters():
                param.requires_grad_(True)
            return
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        if self.tune == "ln":
            for param_name, param in self.backbone.named_parameters():
                if _is_norm_param(param_name):
                    param.requires_grad_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))


def get_model_v2(name: str, **kwargs) -> Detector:
    return Detector(name, **kwargs)


def normalization_for(name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    model_id, _ = BACKBONES[name]
    config = timm.get_pretrained_cfg(model_id, allow_unregistered=True)
    mean = tuple(config.mean) if config and config.mean else (0.485, 0.456, 0.406)
    std = tuple(config.std) if config and config.std else (0.229, 0.224, 0.225)
    return mean, std
