#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from taw_gcn.config.loader import load_and_merge_configs
from taw_gcn.data.factory import create_adapter
from taw_gcn.data.integrity_checker import validate_canonical_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate a raw IoT dataset.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--raw", required=True)
    args = parser.parse_args()
    config = load_and_merge_configs(args.config)
    frame = create_adapter(config).adapt(args.raw)
    report = validate_canonical_frame(frame)
    report["columns"] = frame.columns.tolist()
    report["label_values"] = frame["raw_label"].value_counts().head(30).to_dict()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
