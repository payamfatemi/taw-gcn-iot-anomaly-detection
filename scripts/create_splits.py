#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from taw_gcn.config.loader import load_and_merge_configs
from taw_gcn.preprocessing.pipeline import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare processed train/validation/test splits.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    prepared = prepare_dataset(load_and_merge_configs(args.config), args.raw, Path(args.run_dir))
    print(f"features={len(prepared.feature_columns)} output={prepared.output_dir}")


if __name__ == "__main__":
    main()
