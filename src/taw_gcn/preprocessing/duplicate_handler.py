from __future__ import annotations

import pandas as pd


def drop_duplicate_record_ids(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop_duplicates(subset=["record_id"], keep="first").reset_index(drop=True)
