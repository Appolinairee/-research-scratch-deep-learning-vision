from __future__ import annotations

from dataclasses import dataclass, field

from src.core.backend import NumpyBackend
from src.core.types import Array


@dataclass
class MulticlassFocalLoss:
    gamma: float = 2.0
    alpha: float = 0.25
    eps: float = 1e-12
    backend: NumpyBackend = field(default_factory=NumpyBackend, repr=False)
    _cached_y_true: Array | None = field(init=False, default=None, repr=False)
    _cached_y_pred: Array | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

    def forward(self, y_true: Array, y_pred: Array) -> float:
        y_true = self.backend.asarray(y_true)
        y_pred = self.backend.asarray(y_pred)
        clipped = self.backend.clip(y_pred, self.eps, 1.0 - self.eps)
        self._cached_y_true = y_true
        self._cached_y_pred = clipped
        true_probability = self.backend.sum(y_true * clipped, axis=1)
        loss = -self.alpha * (1.0 - true_probability) ** self.gamma * self.backend.log(
            true_probability
        )
        return self.backend.scalar(self.backend.mean(loss))

    def backward(self) -> Array:
        if self._cached_y_true is None or self._cached_y_pred is None:
            raise ValueError("forward must be called before backward")
        true_probability = self.backend.sum(
            self._cached_y_true * self._cached_y_pred, axis=1
        )
        focusing = (1.0 - true_probability) ** self.gamma
        derivative = self.alpha * (
            self.gamma
            * (1.0 - true_probability) ** (self.gamma - 1.0)
            * self.backend.log(true_probability)
            - focusing / true_probability
        )
        batch_size = self._cached_y_true.shape[0]
        return self._cached_y_true * derivative[:, None] / batch_size
