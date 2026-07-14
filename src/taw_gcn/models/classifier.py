from __future__ import annotations

from torch import nn


def make_binary_classifier(input_dim: int) -> nn.Linear:
    return nn.Linear(input_dim, 1)
