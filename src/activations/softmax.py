from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.types import Array


def softmax(z: Array) -> Array:
    z = np.asarray(z, dtype=float)
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


@dataclass
class Softmax:
    _cached_output: Array | None = field(init=False, default=None, repr=False)

    def forward(self, Z: Array) -> Array:
        A = np.asarray(softmax(Z), dtype=float)
        self._cached_output = A
        return A

    def backward(self, dA: Array) -> Array:
        if self._cached_output is None:
            raise ValueError("forward must be called before backward")
        dA = np.asarray(dA, dtype=float)
        dot = np.sum(dA * self._cached_output, axis=1, keepdims=True)
        return self._cached_output * (dA - dot)

    def apply_gradients(self, lr: float) -> None:
        return None
