from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from taw_gcn.config.validator import validate_config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return content


def load_and_merge_configs(paths: list[str | Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one configuration path is required.")
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml(path))
    validate_config(merged)
    return merged


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value
