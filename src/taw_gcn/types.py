from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch


@dataclass(slots=True)
class PreparedSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    output_dir: Path


@dataclass(slots=True)
class GraphData:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    record_ids: list[str]
    feature_columns: list[str]
    metadata: dict[str, Any]

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_weight.numel())

    def to(self, device: torch.device | str) -> GraphData:
        return GraphData(
            x=self.x.to(device),
            y=self.y.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            record_ids=self.record_ids,
            feature_columns=self.feature_columns,
            metadata=self.metadata,
        )
