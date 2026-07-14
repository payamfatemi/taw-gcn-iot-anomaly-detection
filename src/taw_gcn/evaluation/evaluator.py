from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from taw_gcn.evaluation.metrics import binary_metrics
from taw_gcn.types import GraphData


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    graph: GraphData,
    threshold: float,
    zero_division: int = 0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    if graph.x.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    logits = model(graph.x, graph.edge_index, graph.edge_weight)
    probabilities = torch.sigmoid(logits)
    if graph.x.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    probs = probabilities.detach().cpu().numpy()
    truth = graph.y.detach().cpu().numpy().astype(int)
    metrics = binary_metrics(truth, probs, threshold, zero_division)
    metrics.update(
        {
            "inference_seconds": float(elapsed),
            "inference_milliseconds_per_record": float(elapsed * 1000 / max(len(truth), 1)),
        }
    )
    return metrics, probs, (probs >= threshold).astype(int)
