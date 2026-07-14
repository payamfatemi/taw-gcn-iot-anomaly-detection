from __future__ import annotations

import pandas as pd

COMMUNICATION_COLUMNS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]


def communication_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[column for column in COMMUNICATION_COLUMNS if column in frame.columns]].copy()
