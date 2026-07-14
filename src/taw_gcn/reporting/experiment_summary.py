from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def collect_experiment_metrics(root: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(root).glob("*/test_metrics.json"):
        rows.append({"run": path.parent.name, **json.loads(path.read_text(encoding="utf-8"))})
    return pd.DataFrame(rows)
