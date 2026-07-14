#!/usr/bin/env python
from __future__ import annotations

import argparse

from taw_gcn.config.loader import load_and_merge_configs
from taw_gcn.experiments.ablation import run_ablation

parser = argparse.ArgumentParser()
parser.add_argument('--config', action='append', required=True)
parser.add_argument('--ablation-config', default='configs/experiments/ablation.yaml')
parser.add_argument('--raw', required=True)
parser.add_argument('--prefix', default='ablation')
args = parser.parse_args()
config = load_and_merge_configs(args.config)
ablation = load_and_merge_configs([*args.config, args.ablation_config])
outputs = run_ablation(config, args.raw, ablation['experiment']['variants'], args.prefix)
for name, output in outputs.items():
    print(name, output)
