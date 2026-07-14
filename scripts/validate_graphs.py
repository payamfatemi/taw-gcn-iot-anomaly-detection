#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from taw_gcn.graph.storage import load_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect saved graph artifacts.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    report = {}
    for split in ("train", "validation", "test"):
        graph = load_graph(run_dir / "graphs" / f"{split}_graph.pt")
        report[split] = {
            **graph.metadata,
            "x_shape": list(graph.x.shape),
            "y_shape": list(graph.y.shape),
            "edge_index_shape": list(graph.edge_index.shape),
            "finite_features": bool(graph.x.isfinite().all()),
            "finite_weights": bool(graph.edge_weight.isfinite().all()),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
