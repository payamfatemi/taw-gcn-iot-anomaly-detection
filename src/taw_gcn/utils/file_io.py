from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def write_yaml(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".tsv", ".labeled"} or path.name.endswith(".log.labeled"):
        return pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(frame: pd.DataFrame, path: str | Path, output_format: str = "csv") -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if output_format == "parquet":
        actual = path.with_suffix(".parquet")
        frame.to_parquet(actual, index=False)
    elif output_format == "csv":
        actual = path.with_suffix(".csv")
        frame.to_csv(actual, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return actual
