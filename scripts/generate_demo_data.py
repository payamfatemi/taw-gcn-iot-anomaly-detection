#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from taw_gcn.data.demo_generator import generate_demo_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_frame(args.rows, args.seed).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
