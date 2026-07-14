from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def create_optimizer(parameters: Iterable[torch.nn.Parameter], config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    return torch.optim.Adam(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
