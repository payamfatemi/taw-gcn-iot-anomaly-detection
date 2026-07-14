from __future__ import annotations

from pathlib import Path

import torch

from taw_gcn.types import GraphData
from taw_gcn.utils.file_io import ensure_dir


def save_graph(graph: GraphData, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "x": graph.x,
            "y": graph.y,
            "edge_index": graph.edge_index,
            "edge_weight": graph.edge_weight,
            "record_ids": graph.record_ids,
            "feature_columns": graph.feature_columns,
            "metadata": graph.metadata,
        },
        path,
    )


def load_graph(path: str | Path) -> GraphData:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return GraphData(**payload)
