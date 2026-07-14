from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def temporal_candidates(
    timestamps: np.ndarray,
    window: float,
    neighbors_per_node: int,
) -> set[tuple[int, int]]:
    order = np.argsort(timestamps, kind="mergesort")
    sorted_t = timestamps[order]
    pairs: set[tuple[int, int]] = set()
    for position, node in enumerate(order):
        added = 0
        cursor = position + 1
        while cursor < len(order) and sorted_t[cursor] - sorted_t[position] <= window:
            other = int(order[cursor])
            pairs.add((min(int(node), other), max(int(node), other)))
            added += 1
            if added >= neighbors_per_node:
                break
            cursor += 1
    return pairs


def communication_candidates(
    frame: pd.DataFrame,
    fields: list[str],
    neighbors_per_value: int,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for field in fields:
        if field not in frame.columns:
            continue
        groups: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(frame[field].astype(str).tolist()):
            if value not in {"", "unknown", "nan", "-1"}:
                groups[value].append(index)
        for indices in groups.values():
            for position, node in enumerate(indices):
                for other in indices[position + 1 : position + 1 + neighbors_per_value]:
                    pairs.add((min(node, other), max(node, other)))
    return pairs


def behavioral_candidates(features: np.ndarray, top_k: int) -> set[tuple[int, int]]:
    if len(features) <= 1 or top_k <= 0:
        return set()
    neighbors = min(top_k + 1, len(features))
    model = NearestNeighbors(n_neighbors=neighbors, metric="cosine", algorithm="auto")
    model.fit(features)
    indices = model.kneighbors(features, return_distance=False)
    pairs: set[tuple[int, int]] = set()
    for node, row in enumerate(indices):
        for other in row:
            other = int(other)
            if other != node:
                pairs.add((min(node, other), max(node, other)))
    return pairs


def limit_candidates_per_node(
    pairs: set[tuple[int, int]],
    maximum: int,
    timestamps: np.ndarray,
) -> set[tuple[int, int]]:
    if maximum <= 0:
        return pairs
    adjacency: dict[int, list[int]] = defaultdict(list)
    for i, j in pairs:
        adjacency[i].append(j)
        adjacency[j].append(i)
    limited: set[tuple[int, int]] = set()
    for node, neighbors in adjacency.items():
        ranked = sorted(neighbors, key=lambda other: (abs(timestamps[node] - timestamps[other]), other))
        for other in ranked[:maximum]:
            limited.add((min(node, other), max(node, other)))
    return limited


def generate_candidates(frame: pd.DataFrame, features: np.ndarray, config: dict[str, Any]) -> set[tuple[int, int]]:
    candidate_cfg = config["candidate_generation"]
    timestamps = frame["timestamp"].to_numpy(dtype=float)
    pairs: set[tuple[int, int]] = set()
    if config.get("temporal_enabled", True):
        pairs |= temporal_candidates(
            timestamps,
            float(candidate_cfg["temporal_window"]),
            int(candidate_cfg["temporal_neighbors_per_node"]),
        )
    if config.get("communication_enabled", True):
        pairs |= communication_candidates(
            frame,
            list(config.get("communication_fields", [])),
            int(candidate_cfg["communication_neighbors_per_value"]),
        )
    if config.get("behavioral_enabled", True):
        pairs |= behavioral_candidates(features, int(candidate_cfg["behavioral_top_k"]))
    return limit_candidates_per_node(
        pairs,
        int(candidate_cfg.get("maximum_candidates_per_node", 0)),
        timestamps,
    )
