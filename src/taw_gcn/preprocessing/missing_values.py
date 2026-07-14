from __future__ import annotations

import numpy as np
import pandas as pd


def replace_infinite_with_missing(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)
