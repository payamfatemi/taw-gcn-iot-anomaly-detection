from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class TrafficDataset:
    frame: pd.DataFrame
    name: str

    @property
    def records(self) -> int:
        return len(self.frame)
