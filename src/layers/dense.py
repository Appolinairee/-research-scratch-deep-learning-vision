from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.types import Array


@dataclass
class Dense:
    input_dim: int
    output_dim: int
    seed: int | None = None
    W: Array = field(init=False)
    b: Array = field(init=False)
    dW: Array = field(init=False)
    db: Array = field(init=False)
    _cached_X: Array | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = np.sqrt(2.0 / max(1, self.input_dim))
        self.W = rng.normal(loc=0.0, scale=scale, size=(self.input_dim, self.output_dim))
        self.b = np.zeros(self.output_dim, dtype=float)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        self._cached_X = X
        return X @ self.W + self.b

    def backward(self, dZ: Array) -> Array:
        if self._cached_X is None:
            raise ValueError("forward must be called before backward")

        X = self._cached_X
        dZ = np.asarray(dZ, dtype=float)
        batch_size = X.shape[0]

        self.dW = (X.T @ dZ) / batch_size
        self.db = np.sum(dZ, axis=0) / batch_size
        dX = dZ @ self.W.T
        return dX

    def apply_gradients(self, lr: float) -> None:
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db
