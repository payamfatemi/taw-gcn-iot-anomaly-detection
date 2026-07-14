import torch

from taw_gcn.constants import METADATA_COLUMNS
from taw_gcn.graph.builder import build_graph
from taw_gcn.preprocessing.feature_transformer import fit_transformer, transform_frame


def test_graph_is_sparse_and_finite(canonical_frame, config) -> None:
    fitted = fit_transformer(canonical_frame, config["preprocessing"])
    processed = transform_frame(canonical_frame, fitted, list(METADATA_COLUMNS))
    graph = build_graph(processed, fitted.output_feature_columns, config)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_weight.ndim == 1
    assert graph.num_nodes == len(canonical_frame)
    assert torch.isfinite(graph.edge_weight).all()
    assert (graph.edge_index[0] == graph.edge_index[1]).any()
