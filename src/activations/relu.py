from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.types import Array


@dataclass
class ReLU:
    _cached_input: Array | None = field(init=False, default=None, repr=False)

    def forward(self, Z: Array) -> Array:
        Z = np.asarray(Z, dtype=float)
        self._cached_input = Z
        return np.maximum(0.0, Z)

    def backward(self, dA: Array) -> Array:
        if self._cached_input is None:
            raise ValueError("forward must be called before backward")
        dA = np.asarray(dA, dtype=float)
        mask = (self._cached_input > 0.0).astype(float)
        return dA * mask

    def zero_fraction(self) -> float:
        if self._cached_input is None:
            raise ValueError("forward must be called before reading diagnostics")
        return float(np.mean(self._cached_input <= 0.0))

    def apply_gradients(self, lr: float) -> None:
        return None
