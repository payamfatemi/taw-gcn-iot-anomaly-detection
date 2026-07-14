from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def clean_frame(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    working = frame.copy()
    before = len(working)
    if config.get("replace_infinite", True):
        working = working.replace([np.inf, -np.inf], np.nan)
    duplicate_count = int(working.duplicated(subset=["record_id"]).sum())
    if config.get("remove_duplicates", True):
        working = working.drop_duplicates(subset=["record_id"], keep="first")
    working = working.reset_index(drop=True)
    return working, {
        "rows_before": int(before),
        "rows_after": int(len(working)),
        "duplicate_record_ids_removed": duplicate_count,
    }
