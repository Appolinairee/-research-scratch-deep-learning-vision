from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from src.core.types import Array


def sgd_step(param: Array | float, grad: Array | float, lr: float) -> Array | float:
    if isinstance(param, np.ndarray) or isinstance(grad, np.ndarray):
        return np.asarray(param, dtype=float) - lr * np.asarray(grad, dtype=float)
    return float(param) - lr * float(grad)


def _array_module(value: Array) -> object:
    if isinstance(value, np.ndarray):
        return np
    module_name = type(value).__module__.split(".", 1)[0]
    if module_name == "cupy":
        import cupy

        return cupy
    raise TypeError(f"unsupported array type: {type(value)!r}")


@dataclass
class SGD:
    lr: float
    momentum: float = 0.0
    _velocities: dict[int, Array | float] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")

    def step(
        self,
        param: Array | float,
        grad: Array | float,
        state_key: Hashable | None = None,
    ) -> Array | float:
        if self.momentum == 0:
            return sgd_step(param, grad, self.lr)

        key = state_key if state_key is not None else id(param)
        previous = self._velocities.get(key)
        xp = _array_module(param)
        if previous is None:
            velocity = xp.asarray(grad, dtype=float) * (1 - self.momentum)
        else:
            velocity = (
                self.momentum * previous
                + (1 - self.momentum) * xp.asarray(grad, dtype=float)
            )
        next_param = xp.asarray(param, dtype=float) - self.lr * velocity
        self._velocities[key] = velocity
        if state_key is None:
            self._velocities[id(next_param)] = velocity
        return next_param
