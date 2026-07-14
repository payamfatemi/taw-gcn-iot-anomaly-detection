#!/usr/bin/env python
from __future__ import annotations

import argparse

from taw_gcn.config.loader import load_and_merge_configs
from taw_gcn.workflows import train_from_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TAW-GCN from saved train/validation graphs.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    checkpoint = train_from_graphs(load_and_merge_configs(args.config), args.run_dir)
    print(checkpoint)


if __name__ == "__main__":
    main()
