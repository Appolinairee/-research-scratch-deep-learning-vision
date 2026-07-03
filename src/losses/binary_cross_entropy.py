from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.types import Array


def bernoulli_nll(y: float, p: float, eps: float = 1e-12) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def dataset_binary_cross_entropy(y: Array, p: Array, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(loss))


@dataclass
class BinaryCrossEntropy:
    eps: float = 1e-12
    _cached_y_true: Array | None = field(init=False, default=None, repr=False)
    _cached_y_pred: Array | None = field(init=False, default=None, repr=False)

    def forward(self, y_true: Array, y_pred: Array) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        self._cached_y_true = y_true
        self._cached_y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return dataset_binary_cross_entropy(y_true, self._cached_y_pred, eps=self.eps)

    def backward(self) -> Array:
        if self._cached_y_true is None or self._cached_y_pred is None:
            raise ValueError("forward must be called before backward")
        batch_size = self._cached_y_true.shape[0]
        return -(
            (self._cached_y_true / self._cached_y_pred)
            - ((1.0 - self._cached_y_true) / (1.0 - self._cached_y_pred))
        ) / batch_size
