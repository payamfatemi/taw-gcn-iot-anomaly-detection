from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def profile_runtime(result: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    result[key] = time.perf_counter() - start
