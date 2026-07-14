from __future__ import annotations

from typing import Any

from taw_gcn.models.taw_gcn import TAWGCN


def create_model(input_dim: int, config: dict[str, Any]) -> TAWGCN:
    return TAWGCN.from_config(input_dim, config)
