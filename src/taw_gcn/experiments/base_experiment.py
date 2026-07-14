from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentContext:
    config: dict[str, Any]
    raw: Path
    name: str
