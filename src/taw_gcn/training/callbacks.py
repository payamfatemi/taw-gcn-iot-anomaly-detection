from __future__ import annotations

from collections.abc import Callable
from typing import Any

EpochCallback = Callable[[int, dict[str, Any]], None]
