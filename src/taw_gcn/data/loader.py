from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from taw_gcn.data.factory import create_adapter


def load_canonical_dataset(config: dict[str, Any], raw: str | Path) -> pd.DataFrame:
    return create_adapter(config).adapt(raw)
