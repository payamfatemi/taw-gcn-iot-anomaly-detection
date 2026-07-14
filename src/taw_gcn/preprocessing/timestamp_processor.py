from __future__ import annotations

import pandas as pd


def normalize_timestamp(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    missing = numeric.isna()
    if missing.any():
        parsed = pd.to_datetime(values[missing], errors="coerce", utc=True)
        numeric.loc[missing] = parsed.astype("int64") / 1e9
    return numeric.astype(float)
