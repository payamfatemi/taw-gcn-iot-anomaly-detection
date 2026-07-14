#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from taw_gcn.config.loader import load_and_merge_configs
from taw_gcn.workflows import evaluate_from_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved TAW-GCN checkpoint.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    metrics = evaluate_from_graphs(load_and_merge_configs(args.config), args.run_dir, args.checkpoint)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
