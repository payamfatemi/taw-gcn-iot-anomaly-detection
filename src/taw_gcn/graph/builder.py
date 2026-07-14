from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from taw_gcn.graph.candidates import generate_candidates
from taw_gcn.graph.normalizer import (
    add_self_loops,
    coalesce_max,
    symmetric_normalize,
    symmetrize_max,
)
from taw_gcn.graph.scores import (
    behavioral_score,
    combined_edge_weight,
    communication_score,
    temporal_score,
)
from taw_gcn.graph.validator import graph_statistics, validate_graph
from taw_gcn.types import GraphData


def build_graph(frame: pd.DataFrame, feature_columns: list[str], config: dict[str, Any]) -> GraphData:
    if frame.empty:
        raise ValueError("Cannot build a graph from an empty frame.")
    graph_cfg = config["graph"] if "graph" in config else config
    x = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    y = frame["label"].to_numpy(dtype=np.float32)
    timestamps = frame["timestamp"].to_numpy(dtype=float)
    candidates = generate_candidates(frame, x, graph_cfg)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    enabled = (
        bool(graph_cfg.get("temporal_enabled", True)),
        bool(graph_cfg.get("communication_enabled", True)),
        bool(graph_cfg.get("behavioral_enabled", True)),
    )
    communication_fields = list(graph_cfg.get("communication_fields", []))
    if communication_fields:
        comm_records: list[dict[str, object]] = [
            {str(key): value for key, value in record.items()}
            for record in frame[communication_fields].to_dict("records")
        ]
    else:
        comm_records = [{} for _ in range(len(frame))]
    threshold = float(graph_cfg["delta"])

    for i, j in sorted(candidates):
        s_temp = temporal_score(timestamps[i], timestamps[j], float(graph_cfg["tau"])) if enabled[0] else 0.0
        s_comm = communication_score(comm_records[i], comm_records[j], communication_fields) if enabled[1] else 0.0
        s_beh = behavioral_score(x[i], x[j], float(graph_cfg.get("epsilon", 1e-12))) if enabled[2] else 0.0
        weight = combined_edge_weight(
            s_temp,
            s_comm,
            s_beh,
            float(graph_cfg["alpha"]),
            float(graph_cfg["beta"]),
            float(graph_cfg["gamma"]),
            enabled,
            bool(graph_cfg.get("renormalize_enabled_weights", True)),
        )
        if weight >= threshold:
            rows.append(i)
            cols.append(j)
            values.append(weight if graph_cfg.get("weighted_edges", True) else 1.0)

    n = len(frame)
    if rows:
        matrix = coalesce_max(
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
            np.asarray(values, dtype=np.float32),
            n,
        )
    else:
        matrix = coalesce_max(
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float32),
            n,
        )
    if graph_cfg.get("undirected", True):
        matrix = symmetrize_max(matrix)
    matrix = add_self_loops(matrix, float(graph_cfg.get("self_loop_weight", 1.0)))
    normalized = symmetric_normalize(matrix, float(graph_cfg.get("epsilon", 1e-12)))
    stats = graph_statistics(matrix)
    stats["candidate_pairs"] = len(candidates)
    stats["retained_undirected_pairs_before_symmetrization"] = len(rows)
    validate_graph(stats, graph_cfg)

    edge_index = torch.from_numpy(np.vstack([normalized.row, normalized.col]).astype(np.int64))
    edge_weight = torch.from_numpy(normalized.data.astype(np.float32))
    return GraphData(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        edge_index=edge_index,
        edge_weight=edge_weight,
        record_ids=frame["record_id"].astype(str).tolist(),
        feature_columns=feature_columns,
        metadata=stats,
    )
