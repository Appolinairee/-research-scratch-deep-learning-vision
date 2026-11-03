from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.backend import NumpyBackend
from src.core.types import Array


@dataclass
class Dropout:
    rate: float
    seed: int | None = None
    backend: NumpyBackend = field(default_factory=NumpyBackend, repr=False)
    training: bool = True
    _mask: Array | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.rate < 1:
            raise ValueError("rate must be in [0, 1)")
        self._rng = np.random.default_rng(self.seed)

    def set_training(self, training: bool) -> None:
        self.training = training

    def forward(self, X: Array, training: bool | None = None) -> Array:
        X = self.backend.asarray(X)
        active = self.training if training is None else training
        if not active:
            self._mask = None
            return X
        mask = self._rng.random(X.shape) >= self.rate
        self._mask = self.backend.asarray(mask, dtype=float) / (1 - self.rate)
        return X * self._mask

    def backward(self, dA: Array) -> Array:
        if self._mask is None:
            return self.backend.asarray(dA)
        return self.backend.asarray(dA) * self._mask

    def apply_gradients(self, _lr: float) -> None:
        return None
