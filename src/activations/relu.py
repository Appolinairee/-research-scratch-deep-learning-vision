from __future__ import annotations

from dataclasses import dataclass, field

from src.core.backend import NumpyBackend
from src.core.types import Array


@dataclass
class ReLU:
    backend: NumpyBackend = field(default_factory=NumpyBackend, repr=False)
    _cached_input: Array | None = field(init=False, default=None, repr=False)

    def forward(self, Z: Array) -> Array:
        Z = self.backend.asarray(Z)
        self._cached_input = Z
        return self.backend.maximum(0.0, Z)

    def backward(self, dA: Array) -> Array:
        if self._cached_input is None:
            raise ValueError("forward must be called before backward")
        dA = self.backend.asarray(dA)
        mask = (self._cached_input > 0.0).astype(float)
        return dA * mask

    def zero_fraction(self) -> float:
        if self._cached_input is None:
            raise ValueError("forward must be called before reading diagnostics")
        return self.backend.scalar(self.backend.mean(self._cached_input <= 0.0))

    def apply_gradients(self, lr: float) -> None:
        return None
