from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def map_binary_labels(values: pd.Series, normal_tokens: Iterable[str]) -> pd.Series:
    normalized = {str(token).strip().lower() for token in normal_tokens}
    return values.map(lambda value: 0 if str(value).strip().lower() in normalized else 1).astype("int64")
