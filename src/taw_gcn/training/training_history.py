from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TrainingHistory:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
