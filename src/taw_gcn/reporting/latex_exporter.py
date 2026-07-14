from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_latex(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).write_text(frame.to_latex(index=False), encoding="utf-8")
