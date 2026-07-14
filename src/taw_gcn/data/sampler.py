from __future__ import annotations

import pandas as pd


def sample_records(frame: pd.DataFrame, records: int, seed: int = 42) -> pd.DataFrame:
    if records > len(frame):
        raise ValueError("Requested sample exceeds available records.")
    return frame.sample(n=records, random_state=seed).reset_index(drop=True)
