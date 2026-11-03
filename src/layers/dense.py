from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.backend import NumpyBackend
from src.core.types import Array
from src.optim import SGD

@dataclass
class Dense:
    input_dim: int
    output_dim: int
    seed: int | None = None
    backend: NumpyBackend = field(default_factory=NumpyBackend, repr=False)
    W: Array = field(init=False)
    b: Array = field(init=False)
    dW: Array = field(init=False)
    db: Array = field(init=False)
    _cached_X: Array | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        scale = np.sqrt(2.0 / max(1, self.input_dim))
        self.W = self.backend.random_normal(
            self.seed, scale, (self.input_dim, self.output_dim)
        )
        self.b = self.backend.zeros(self.output_dim)
        self.dW = self.backend.zeros_like(self.W)
        self.db = self.backend.zeros_like(self.b)

    def forward(self, X: Array) -> Array:
        X = self.backend.asarray(X)
        self._cached_X = X
        return X @ self.W + self.b

    def backward(self, dZ: Array) -> Array:
        if self._cached_X is None:
            raise ValueError("forward must be called before backward")

        X = self._cached_X
        dZ = self.backend.asarray(dZ)

        self.dW = X.T @ dZ
        self.db = self.backend.sum(dZ, axis=0)
        dX = dZ @ self.W.T
        return dX

    def apply_gradients(
        self,
        optimizer: SGD | float | None = None,
        *,
        lr: float | None = None,
    ) -> None:
        if optimizer is None:
            if lr is None:
                raise TypeError("provide optimizer or lr")
            optimizer = lr
        elif lr is not None:
            raise TypeError("provide optimizer or lr, not both")
        if isinstance(optimizer, SGD):
            self.W = optimizer.step(self.W, self.dW, state_key=(id(self), "W"))
            self.b = optimizer.step(self.b, self.db, state_key=(id(self), "b"))
            return
        self.W = self.W - optimizer * self.dW
        self.b = self.b - optimizer * self.db
