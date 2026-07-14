#!/usr/bin/env python
import argparse
from pathlib import Path

from taw_gcn.config.loader import load_and_merge_configs
from taw_gcn.preprocessing.pipeline import prepare_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--raw', required=True)
parser.add_argument('--output', default='outputs/prepare_ton_iot')
args = parser.parse_args()
config = load_and_merge_configs(['configs/base.yaml', 'configs/datasets/ton_iot.yaml'])
prepare_dataset(config, args.raw, Path(args.output))
print(args.output)
