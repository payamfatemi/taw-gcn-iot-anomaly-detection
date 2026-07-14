#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from taw_gcn.ablation import ABLATION_VARIANTS
from taw_gcn.config.loader import load_and_merge_configs
from taw_gcn.experiments.ablation import run_ablation
from taw_gcn.experiments.scenarios import run_data_volume_experiment, run_network_scale_experiment
from taw_gcn.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standard and requested internal TAW-GCN experiments.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--include-ablation", action="store_true")
    parser.add_argument("--include-data-volume", action="store_true")
    parser.add_argument("--include-network-scale", action="store_true")
    args = parser.parse_args()
    config = load_and_merge_configs(args.config)
    result = {"standard": str(run_pipeline(config, args.raw, "standard"))}
    if args.include_ablation:
        result["ablation"] = run_ablation(config, args.raw, ABLATION_VARIANTS)
    if args.include_data_volume:
        result["data_volume"] = run_data_volume_experiment(
            config, args.raw, [100000, 500000, 1000000, 5000000, 10000000, 20000000]
        )
    if args.include_network_scale:
        result["network_scale"] = run_network_scale_experiment(
            config, args.raw, [50, 200, 1000, 5000, 10000]
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
