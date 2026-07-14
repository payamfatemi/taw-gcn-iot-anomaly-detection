from __future__ import annotations

from typing import Any


def compare_measured_latency(measured_ms: float, targets_ms: list[float]) -> list[dict[str, Any]]:
    return [
        {"target_ms": float(target), "measured_ms": float(measured_ms), "meets_target": measured_ms <= target}
        for target in targets_ms
    ]
