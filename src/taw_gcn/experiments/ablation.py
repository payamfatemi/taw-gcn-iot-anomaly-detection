from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from taw_gcn.config.loader import set_dotted
from taw_gcn.pipeline import run_pipeline


def run_ablation(
    base_config: dict[str, Any],
    raw: str | Path,
    variants: dict[str, dict[str, Any]],
    prefix: str = "ablation",
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for name, overrides in variants.items():
        config = deepcopy(base_config)
        for dotted, value in overrides.items():
            set_dotted(config, dotted, value)
        run_dir = run_pipeline(config, raw, f"{prefix}_{name}")
        outputs[name] = str(run_dir)
    return outputs
