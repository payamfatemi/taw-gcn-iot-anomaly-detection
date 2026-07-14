from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def binary_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.asarray(confusion_matrix(y_true, y_pred, labels=[0, 1]), dtype=np.int64)
