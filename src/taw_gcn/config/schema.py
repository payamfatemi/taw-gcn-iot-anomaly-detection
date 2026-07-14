from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GraphParameters:
    alpha: float
    beta: float
    gamma: float
    tau: float
    delta: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GraphParameters:
        graph = config["graph"]
        return cls(*(float(graph[key]) for key in ("alpha", "beta", "gamma", "tau", "delta")))


@dataclass(frozen=True, slots=True)
class SplitParameters:
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    random_state: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SplitParameters:
        split = config["split"]
        return cls(
            float(split["train_ratio"]),
            float(split["validation_ratio"]),
            float(split["test_ratio"]),
            int(split.get("random_state", 42)),
        )
