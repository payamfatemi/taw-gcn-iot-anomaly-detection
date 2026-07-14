#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset acquisition guidance.")
    parser.add_argument("dataset", choices=["iot23", "ton_iot"])
    args = parser.parse_args()
    destinations = {"iot23": "data/raw/iot23/", "ton_iot": "data/raw/ton_iot/"}
    print(
        f"Automatic redistribution is intentionally disabled. Obtain {args.dataset} from its official "
        f"provider under the applicable license, verify its checksum, and place the authorized files "
        f"under {destinations[args.dataset]}. Then run scripts/inspect_dataset.py."
    )


if __name__ == "__main__":
    main()
