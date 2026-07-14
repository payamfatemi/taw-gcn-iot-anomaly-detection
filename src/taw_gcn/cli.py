from __future__ import annotations

import argparse

from taw_gcn.config.loader import load_and_merge_configs
from taw_gcn.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete TAW-GCN pipeline.")
    parser.add_argument("--config", action="append", required=True, help="YAML config; repeat to merge.")
    parser.add_argument("--raw", required=True, help="Raw dataset file or directory.")
    parser.add_argument("--run-name", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_and_merge_configs(args.config)
    run_dir = run_pipeline(config, args.raw, args.run_name)
    print(f"Completed: {run_dir}")


if __name__ == "__main__":
    main()
