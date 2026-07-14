from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from taw_gcn.utils.hashing import sha256_file


def environment_manifest() -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_commit = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git_commit": git_commit,
    }


def write_predictions(
    record_ids: list[str],
    truth: Any,
    probabilities: Any,
    predictions: Any,
    path: str | Path,
) -> None:
    frame = pd.DataFrame(
        {
            "record_id": record_ids,
            "true_label": truth,
            "anomaly_probability": probabilities,
            "predicted_label": predictions,
        }
    )
    frame.to_csv(path, index=False)


def write_checksums(directory: str | Path) -> None:
    directory = Path(directory)
    rows: list[str] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(directory)}")
    (directory / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
