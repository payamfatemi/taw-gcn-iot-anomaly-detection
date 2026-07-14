#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from taw_gcn.utils.hashing import sha256_file

parser = argparse.ArgumentParser()
parser.add_argument('run_dir')
args = parser.parse_args()
run_dir = Path(args.run_dir)
checksum_file = run_dir / 'checksums.sha256'
if not checksum_file.exists():
    raise SystemExit('checksums.sha256 not found')
failures = []
for line in checksum_file.read_text(encoding='utf-8').splitlines():
    expected, relative = line.split('  ', 1)
    path = run_dir / relative
    observed = sha256_file(path)
    if observed != expected:
        failures.append({'path': relative, 'expected': expected, 'observed': observed})
print(json.dumps({'verified': not failures, 'failures': failures}, indent=2))
raise SystemExit(1 if failures else 0)
