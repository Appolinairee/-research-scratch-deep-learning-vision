from __future__ import annotations

from dataclasses import dataclass

from src.optim.sgd import SGD


@dataclass
class ReduceLROnPlateau:
    factor: float = 0.5
    patience: int = 10
    min_lr: float = 1e-6
    _best: float | None = None
    _bad_epochs: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.factor < 1:
            raise ValueError("factor must be in (0, 1)")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.min_lr < 0:
            raise ValueError("min_lr must be non-negative")

    def step(self, metric: float, optimizer: SGD) -> None:
        if self._best is None or metric < self._best:
            self._best = metric
            self._bad_epochs = 0
            return

        self._bad_epochs += 1
        if self._bad_epochs >= self.patience:
            optimizer.lr = max(optimizer.lr * self.factor, self.min_lr)
            self._bad_epochs = 0
