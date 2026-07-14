#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate test metrics from experiment directories.")
    parser.add_argument("--root", default="outputs/experiments")
    parser.add_argument("--output", default="outputs/experiment_summary.csv")
    args = parser.parse_args()
    rows = []
    for metric_file in sorted(Path(args.root).glob("*/test_metrics.json")):
        metrics = json.loads(metric_file.read_text(encoding="utf-8"))
        rows.append({"run": metric_file.parent.name, **metrics})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
