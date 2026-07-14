from __future__ import annotations

from pathlib import Path
from typing import Any


def export_metric_markdown(metrics: dict[str, Any], path: str | Path) -> None:
    lines = ["| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in metrics.items())
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
