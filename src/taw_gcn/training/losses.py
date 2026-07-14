from __future__ import annotations

import torch
from torch import nn


def binary_classification_loss(pos_weight: torch.Tensor | None = None) -> nn.BCEWithLogitsLoss:
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
