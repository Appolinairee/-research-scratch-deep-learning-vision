from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import Array


def sgd_step(param: Array | float, grad: Array | float, lr: float) -> Array | float:
    if isinstance(param, np.ndarray) or isinstance(grad, np.ndarray):
        return np.asarray(param, dtype=float) - lr * np.asarray(grad, dtype=float)
    return float(param) - lr * float(grad)


@dataclass(frozen=True)
class SGD:
    lr: float

    def step(self, param: Array | float, grad: Array | float) -> Array | float:
        return sgd_step(param, grad, self.lr)
