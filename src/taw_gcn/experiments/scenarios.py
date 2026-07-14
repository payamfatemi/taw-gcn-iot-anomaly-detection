from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from taw_gcn.data.factory import create_adapter
from taw_gcn.pipeline import run_pipeline
from taw_gcn.utils.file_io import ensure_dir, write_json


def _balanced_sample(frame: pd.DataFrame, records: int, seed: int) -> pd.DataFrame:
    if records > len(frame):
        raise ValueError("Insufficient unique records")
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("label"):
        count = max(1, round(records * len(group) / len(frame)))
        parts.append(group.sample(n=min(count, len(group)), random_state=seed))
    sample = pd.concat(parts).drop_duplicates(subset=["record_id"])
    if len(sample) < records:
        remainder = frame.loc[~frame["record_id"].isin(sample["record_id"])]
        sample = pd.concat([sample, remainder.sample(n=records - len(sample), random_state=seed)])
    return sample.sample(n=records, random_state=seed).reset_index(drop=True)


def run_data_volume_experiment(
    config: dict[str, Any],
    raw: str | Path,
    values: list[int],
    prefix: str = "data_volume",
) -> dict[str, Any]:
    canonical = create_adapter(config).adapt(raw)
    seed = int(config["project"].get("seed", 42))
    staging = ensure_dir(Path(config["paths"]["output_root"]) / "_scenario_inputs" / prefix)
    manifest: dict[str, Any] = {"scenario": "data_volume", "runs": {}}
    for value in values:
        if value > len(canonical):
            manifest["runs"][str(value)] = {
                "status": "skipped",
                "reason": "insufficient_unique_records",
                "available": len(canonical),
            }
            continue
        sample = _balanced_sample(canonical, value, seed)
        path = staging / f"records_{value}.csv"
        sample.to_csv(path, index=False)
        run_dir = run_pipeline(config, path, f"{prefix}_{value}")
        manifest["runs"][str(value)] = {"status": "completed", "run_dir": str(run_dir)}
    write_json(manifest, Path(config["paths"]["output_root"]) / f"{prefix}_manifest.json")
    return manifest


def run_network_scale_experiment(
    config: dict[str, Any],
    raw: str | Path,
    device_counts: list[int],
    device_column: str = "src_ip",
    prefix: str = "network_scale",
) -> dict[str, Any]:
    canonical = create_adapter(config).adapt(raw)
    devices = sorted(canonical[device_column].astype(str).unique().tolist())
    staging = ensure_dir(Path(config["paths"]["output_root"]) / "_scenario_inputs" / prefix)
    manifest: dict[str, Any] = {"scenario": "network_scale", "device_column": device_column, "runs": {}}
    for count in device_counts:
        if count > len(devices):
            manifest["runs"][str(count)] = {
                "status": "skipped",
                "reason": "insufficient_unique_devices",
                "available": len(devices),
            }
            continue
        selected = set(devices[:count])
        sample = canonical[canonical[device_column].astype(str).isin(selected)].reset_index(drop=True)
        path = staging / f"devices_{count}.csv"
        sample.to_csv(path, index=False)
        run_dir = run_pipeline(config, path, f"{prefix}_{count}")
        manifest["runs"][str(count)] = {
            "status": "completed",
            "records": len(sample),
            "run_dir": str(run_dir),
        }
    write_json(manifest, Path(config["paths"]["output_root"]) / f"{prefix}_manifest.json")
    return manifest
