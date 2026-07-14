from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def timed(store: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - start
