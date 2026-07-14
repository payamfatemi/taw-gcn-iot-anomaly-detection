from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    minimum_delta: float = 0.0
    mode: str = "max"
    best: float | None = None
    stale_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = value > self.best + self.minimum_delta
        else:
            improved = value < self.best - self.minimum_delta
        if improved:
            self.best = value
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return improved, self.stale_epochs >= self.patience
