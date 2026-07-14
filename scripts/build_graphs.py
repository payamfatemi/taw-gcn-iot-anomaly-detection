#!/usr/bin/env python
from __future__ import annotations

import argparse

from taw_gcn.config.loader import load_and_merge_configs
from taw_gcn.workflows import build_graphs_from_processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sparse TAW graphs from processed splits.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    outputs = build_graphs_from_processed(load_and_merge_configs(args.config), args.run_dir)
    for split, path in outputs.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
