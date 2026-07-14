from __future__ import annotations

from typing import Any

import pandas as pd

from taw_gcn.constants import CANONICAL_REQUIRED_COLUMNS
from taw_gcn.exceptions import DataIntegrityError


def validate_canonical_frame(frame: pd.DataFrame) -> dict[str, Any]:
    missing = [column for column in CANONICAL_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataIntegrityError(f"Canonical frame is missing columns: {missing}")
    if frame.empty:
        raise DataIntegrityError("Canonical frame is empty.")
    if frame["record_id"].duplicated().any():
        raise DataIntegrityError("record_id values must be unique.")
    labels = set(frame["label"].dropna().astype(int).unique().tolist())
    if not labels.issubset({0, 1}) or len(labels) < 2:
        raise DataIntegrityError(f"Binary labels 0 and 1 are required; observed {sorted(labels)}")
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "normal": int((frame["label"] == 0).sum()),
        "anomaly": int((frame["label"] == 1).sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }
