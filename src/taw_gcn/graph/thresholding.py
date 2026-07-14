from __future__ import annotations


def retain_edge(weight: float, delta: float, i: int, j: int) -> bool:
    return i != j and weight >= delta
