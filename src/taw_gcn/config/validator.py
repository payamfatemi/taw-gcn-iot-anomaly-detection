from __future__ import annotations

from typing import Any

from taw_gcn.exceptions import ConfigurationError


def validate_config(config: dict[str, Any]) -> None:
    required = ("project", "paths", "split", "preprocessing", "graph", "model", "training")
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigurationError(f"Missing required configuration sections: {missing}")

    split = config["split"]
    total = sum(float(split[key]) for key in ("train_ratio", "validation_ratio", "test_ratio"))
    if abs(total - 1.0) > 1e-8:
        raise ConfigurationError("Train/validation/test ratios must sum to 1.")

    graph = config["graph"]
    weights = [float(graph[name]) for name in ("alpha", "beta", "gamma")]
    if any(value < 0 for value in weights):
        raise ConfigurationError("alpha, beta, and gamma must be non-negative.")
    if abs(sum(weights) - 1.0) > 1e-5:
        raise ConfigurationError("alpha + beta + gamma must equal 1.")
    if float(graph["tau"]) <= 0:
        raise ConfigurationError("tau must be positive.")
    if not 0 <= float(graph["delta"]) <= 1:
        raise ConfigurationError("delta must be in [0, 1].")

    if int(config["training"]["maximum_epochs"]) <= 0:
        raise ConfigurationError("maximum_epochs must be positive.")
