from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from src.core.types import Array

from .sgd import _array_module


@dataclass
class Adam:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    decay: float = 0.0
    _states: dict[int | Hashable, tuple[Array, Array, int]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= self.beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.decay < 0:
            raise ValueError("decay must be non-negative")

    def step(
        self,
        param: Array | float,
        grad: Array | float,
        state_key: Hashable | None = None,
    ) -> Array | float:
        key = state_key if state_key is not None else id(param)
        if isinstance(param, np.ndarray):
            xp = np
        else:
            try:
                xp = _array_module(param)
            except TypeError:
                xp = np
        parameter = xp.asarray(param, dtype=float)
        gradient = xp.asarray(grad, dtype=float)
        previous = self._states.get(key)
        if previous is None:
            first_moment = xp.zeros_like(parameter)
            second_moment = xp.zeros_like(parameter)
            step_number = 0
        else:
            first_moment, second_moment, step_number = previous

        step_number += 1
        first_moment = self.beta1 * first_moment + (1 - self.beta1) * gradient
        second_moment = self.beta2 * second_moment + (1 - self.beta2) * gradient**2
        corrected_first = first_moment / (1 - self.beta1**step_number)
        corrected_second = second_moment / (1 - self.beta2**step_number)
        learning_rate = self.lr / (1.0 + self.decay * (step_number - 1))
        updated = parameter - learning_rate * corrected_first / (
            xp.sqrt(corrected_second) + self.epsilon
        )
        self._states[key] = (first_moment, second_moment, step_number)
        if state_key is None:
            self._states[id(updated)] = self._states.pop(key)
        if isinstance(param, np.ndarray) or not np.isscalar(updated):
            return updated
        return float(updated)
