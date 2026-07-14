from __future__ import annotations

import os

import psutil
import torch


def memory_snapshot() -> dict[str, float]:
    process = psutil.Process(os.getpid())
    result = {"rss_bytes": float(process.memory_info().rss)}
    if torch.cuda.is_available():
        result["cuda_allocated_bytes"] = float(torch.cuda.memory_allocated())
        result["cuda_reserved_bytes"] = float(torch.cuda.memory_reserved())
    return result
