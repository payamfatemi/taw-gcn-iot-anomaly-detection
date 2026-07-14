from __future__ import annotations

from typing import Any


def validate_binary_metrics(metrics: dict[str, Any]) -> None:
    for name in ("accuracy", "precision", "recall", "f1"):
        value = float(metrics[name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} is outside [0, 1]: {value}")
