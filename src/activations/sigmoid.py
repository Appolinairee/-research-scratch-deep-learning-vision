from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.types import Array


def sigmoid(z: Array | float) -> Array | float:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Sigmoid:
    _cached_output: Array | None = field(init=False, default=None, repr=False)

    def forward(self, Z: Array) -> Array:
        A = np.asarray(sigmoid(Z), dtype=float)
        self._cached_output = A
        return A

    def backward(self, dA: Array) -> Array:
        if self._cached_output is None:
            raise ValueError("forward must be called before backward")
        dA = np.asarray(dA, dtype=float)
        return dA * self._cached_output * (1.0 - self._cached_output)

    def apply_gradients(self, lr: float) -> None:
        return None
