#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from taw_gcn.config.loader import load_and_merge_configs
from taw_gcn.experiments.scenarios import run_data_volume_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--scenario-config", default="configs/experiments/data_volume.yaml")
    parser.add_argument("--raw", required=True)
    args = parser.parse_args()
    config = load_and_merge_configs(args.config)
    scenario = load_and_merge_configs([*args.config, args.scenario_config])["experiment"]
    result = run_data_volume_experiment(config, args.raw, scenario["values"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
