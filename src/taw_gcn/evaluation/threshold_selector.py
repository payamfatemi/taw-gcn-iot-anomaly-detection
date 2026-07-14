from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray, candidates: int = 101) -> float:
    thresholds = np.linspace(0.0, 1.0, candidates)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])
