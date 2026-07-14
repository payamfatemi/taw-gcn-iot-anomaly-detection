from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from taw_gcn.config.loader import load_and_merge_configs


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def config(project_root: Path) -> dict:
    cfg = load_and_merge_configs(
        [project_root / "configs/base.yaml", project_root / "configs/datasets/demo.yaml"]
    )
    cfg["training"]["maximum_epochs"] = 3
    cfg["training"]["early_stopping"]["patience"] = 2
    cfg["graph"]["minimum_average_degree"] = 0.5
    return cfg


@pytest.fixture
def canonical_frame() -> pd.DataFrame:
    rows = 60
    return pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(rows)],
            "timestamp": [float(i) for i in range(rows)],
            "src_ip": [f"10.0.0.{i % 6}" for i in range(rows)],
            "dst_ip": [f"192.168.0.{i % 3}" for i in range(rows)],
            "src_port": [1000 + i % 10 for i in range(rows)],
            "dst_port": [80 if i % 2 else 443 for i in range(rows)],
            "protocol": ["tcp" if i % 3 else "udp" for i in range(rows)],
            "service": ["http" if i % 2 else "mqtt" for i in range(rows)],
            "conn_state": ["SF" if i % 4 else "S0" for i in range(rows)],
            "capture_id": [f"c{i // 10}" for i in range(rows)],
            "scenario_id": [f"s{i // 20}" for i in range(rows)],
            "raw_label": ["normal" if i % 3 else "attack" for i in range(rows)],
            "label": [0 if i % 3 else 1 for i in range(rows)],
            "duration": [i / 10 for i in range(rows)],
            "bytes": [100 + i * 3 for i in range(rows)],
        }
    )
