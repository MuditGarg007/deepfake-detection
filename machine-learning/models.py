"""T5 — backbone builders for the two compared architectures."""

from __future__ import annotations

import timm
import torch.nn as nn

# Friendly name -> timm model id. timm renamed the original Chollet Xception to
# ``legacy_xception``; the plan's "Xception" means that one.
MODEL_NAMES = {
    "efficientnet_b0": "efficientnet_b0",
    "xception": "legacy_xception",
}


def get_model(name: str, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    """Build ``name`` with a fresh binary head emitting a single logit.

    The backbone is created with ``num_classes=0``, which makes timm return the
    pooled feature vector; the head below maps it to one logit for
    ``BCEWithLogitsLoss``.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model '{name}' — choose from {sorted(MODEL_NAMES)}")

    backbone = timm.create_model(MODEL_NAMES[name], pretrained=pretrained, num_classes=0)
    return nn.Sequential(
        backbone,
        nn.Linear(backbone.num_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(128, 1),
    )
