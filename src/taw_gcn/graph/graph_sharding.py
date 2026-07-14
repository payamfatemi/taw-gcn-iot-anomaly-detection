from __future__ import annotations

from collections.abc import Iterator

from taw_gcn.types import GraphData


def contiguous_node_ranges(num_nodes: int, shard_size: int) -> Iterator[tuple[int, int]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    for start in range(0, num_nodes, shard_size):
        yield start, min(start + shard_size, num_nodes)


def induced_shard(graph: GraphData, start: int, end: int) -> GraphData:
    mask = (
        (graph.edge_index[0] >= start)
        & (graph.edge_index[0] < end)
        & (graph.edge_index[1] >= start)
        & (graph.edge_index[1] < end)
    )
    edge_index = graph.edge_index[:, mask] - start
    return GraphData(
        x=graph.x[start:end],
        y=graph.y[start:end],
        edge_index=edge_index,
        edge_weight=graph.edge_weight[mask],
        record_ids=graph.record_ids[start:end],
        feature_columns=graph.feature_columns,
        metadata={**graph.metadata, "shard_start": start, "shard_end": end},
    )
