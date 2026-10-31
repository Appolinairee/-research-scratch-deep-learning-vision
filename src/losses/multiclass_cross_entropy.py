from __future__ import annotations

from dataclasses import dataclass, field

from src.core.backend import NumpyBackend
from src.core.types import Array


@dataclass
class MulticlassCrossEntropy:
    eps: float = 1e-12
    backend: NumpyBackend = field(default_factory=NumpyBackend, repr=False)
    _cached_y_true: Array | None = field(init=False, default=None, repr=False)
    _cached_y_pred: Array | None = field(init=False, default=None, repr=False)

    def forward(self, y_true: Array, y_pred: Array) -> float:
        y_true = self.backend.asarray(y_true)
        y_pred = self.backend.asarray(y_pred)
        self._cached_y_true = y_true
        self._cached_y_pred = self.backend.clip(
            y_pred, self.eps, 1.0 - self.eps
        )
        loss = -self.backend.sum(
            y_true * self.backend.log(self._cached_y_pred), axis=1
        )
        return self.backend.scalar(self.backend.mean(loss))

    def backward(self) -> Array:
        if self._cached_y_true is None or self._cached_y_pred is None:
            raise ValueError("forward must be called before backward")
        batch_size = self._cached_y_true.shape[0]
        return -(self._cached_y_true / self._cached_y_pred) / batch_size
