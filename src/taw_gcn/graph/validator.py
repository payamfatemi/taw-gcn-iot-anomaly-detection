from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from taw_gcn.exceptions import GraphValidationError


def graph_statistics(matrix: sp.spmatrix) -> dict[str, Any]:
    csr = matrix.tocsr()
    n = csr.shape[0]
    degree = np.diff(csr.indptr)
    non_self_edges = int(csr.nnz - np.count_nonzero(csr.diagonal()))
    return {
        "num_nodes": int(n),
        "num_stored_edges_including_directions_and_self_loops": int(csr.nnz),
        "num_non_self_directed_entries": non_self_edges,
        "average_stored_degree": float(csr.nnz / max(n, 1)),
        "isolated_ratio_before_self_loop_interpretation": float(np.mean(degree <= 1)) if n else 0.0,
        "density": float(csr.nnz / max(n * n, 1)),
        "minimum_weight": float(csr.data.min()) if csr.nnz else 0.0,
        "maximum_weight": float(csr.data.max()) if csr.nnz else 0.0,
    }


def validate_graph(stats: dict[str, Any], config: dict[str, Any]) -> None:
    average = float(stats["average_stored_degree"])
    isolated = float(stats["isolated_ratio_before_self_loop_interpretation"])
    if average < float(config.get("minimum_average_degree", 0.0)):
        raise GraphValidationError(f"Graph is too sparse: average degree {average:.3f}")
    if average > float(config.get("maximum_average_degree", float("inf"))):
        raise GraphValidationError(f"Graph is too dense: average degree {average:.3f}")
    if isolated > float(config.get("maximum_isolated_ratio", 1.0)):
        raise GraphValidationError(f"Too many isolated nodes: ratio {isolated:.3f}")
