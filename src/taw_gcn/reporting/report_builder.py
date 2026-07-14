from __future__ import annotations

from pathlib import Path
from typing import Any

from taw_gcn.reporting.markdown_exporter import export_metric_markdown


def build_metric_report(metrics: dict[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "metrics.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    export_metric_markdown(metrics, path)
    return path
