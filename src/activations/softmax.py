from __future__ import annotations

from dataclasses import dataclass, field

from src.core.backend import NumpyBackend
from src.core.types import Array


def softmax(z: Array, backend: NumpyBackend | None = None) -> Array:
    backend = backend or NumpyBackend()
    z = backend.asarray(z)
    shifted = z - backend.max(z, axis=1, keepdims=True)
    exp = backend.exp(shifted)
    return exp / backend.sum(exp, axis=1, keepdims=True)


@dataclass
class Softmax:
    backend: NumpyBackend = field(default_factory=NumpyBackend, repr=False)
    _cached_output: Array | None = field(init=False, default=None, repr=False)

    def forward(self, Z: Array) -> Array:
        A = self.backend.asarray(softmax(Z, self.backend))
        self._cached_output = A
        return A

    def backward(self, dA: Array) -> Array:
        if self._cached_output is None:
            raise ValueError("forward must be called before backward")
        dA = self.backend.asarray(dA)
        dot = self.backend.sum(dA * self._cached_output, axis=1, keepdims=True)
        return self._cached_output * (dA - dot)

    def apply_gradients(self, lr: float) -> None:
        return None
