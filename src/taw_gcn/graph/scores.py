from __future__ import annotations

import numpy as np


def temporal_score(t_i: float, t_j: float, tau: float) -> float:
    return float(np.exp(-abs(float(t_i) - float(t_j)) / float(tau)))


def communication_score(left: dict[str, object], right: dict[str, object], fields: list[str]) -> float:
    for field in fields:
        a, b = left.get(field), right.get(field)
        if a is None or b is None:
            continue
        if str(a) not in {"", "unknown", "nan", "-1"} and str(a) == str(b):
            return 1.0
    return 0.0


def behavioral_score(x_i: np.ndarray, x_j: np.ndarray, epsilon: float = 1e-12) -> float:
    denominator = float(np.linalg.norm(x_i) * np.linalg.norm(x_j) + epsilon)
    value = float(np.dot(x_i, x_j) / denominator)
    return float(np.clip(value, 0.0, 1.0))


def combined_edge_weight(
    s_temp: float,
    s_comm: float,
    s_beh: float,
    alpha: float,
    beta: float,
    gamma: float,
    enabled: tuple[bool, bool, bool] = (True, True, True),
    renormalize: bool = True,
) -> float:
    raw_weights = np.asarray([alpha, beta, gamma], dtype=float)
    active = np.asarray(enabled, dtype=bool)
    effective = raw_weights * active
    if renormalize and effective.sum() > 0:
        effective = effective / effective.sum()
    scores = np.asarray([s_temp, s_comm, s_beh], dtype=float)
    return float(np.dot(effective, scores))
