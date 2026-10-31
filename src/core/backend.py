from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NumpyBackend:
    xp: Any = np
    is_gpu: bool = False

    def asarray(self, value: Any, dtype: Any = float) -> Any:
        return self.xp.asarray(value, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = float) -> Any:
        return self.xp.zeros(shape, dtype=dtype)

    def zeros_like(self, value: Any) -> Any:
        return self.xp.zeros_like(value)

    def maximum(self, left: Any, right: Any) -> Any:
        return self.xp.maximum(left, right)

    def clip(self, value: Any, lower: float, upper: float) -> Any:
        return self.xp.clip(value, lower, upper)

    def sum(
        self, value: Any, axis: Any = None, keepdims: bool = False
    ) -> Any:
        return self.xp.sum(value, axis=axis, keepdims=keepdims)

    def mean(self, value: Any) -> Any:
        return self.xp.mean(value)

    def log(self, value: Any) -> Any:
        return self.xp.log(value)

    def exp(self, value: Any) -> Any:
        return self.xp.exp(value)

    def max(self, value: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return self.xp.max(value, axis=axis, keepdims=keepdims)

    def argmax(self, value: Any, axis: int) -> Any:
        return self.xp.argmax(value, axis=axis)

    def sqrt(self, value: Any) -> Any:
        return self.xp.sqrt(value)

    def random_normal(self, seed: int | None, scale: float, shape: Any) -> Any:
        rng = self.xp.random.default_rng(seed)
        return rng.normal(loc=0.0, scale=scale, size=shape)

    def is_array(self, value: Any) -> bool:
        return isinstance(value, self.xp.ndarray)

    def to_host(self, value: Any) -> np.ndarray:
        if self.is_gpu:
            return self.xp.asnumpy(value)
        return np.asarray(value)

    def scalar(self, value: Any) -> float:
        return float(self.to_host(value))

    def synchronize(self) -> None:
        if self.is_gpu:
            self.xp.cuda.get_current_stream().synchronize()


def get_backend(name: str = "cpu") -> NumpyBackend:
    if name == "cpu":
        return NumpyBackend()
    if name != "gpu":
        raise ValueError(f"unknown backend: {name}")
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("GPU backend requires CuPy") from exc
    if not cp.cuda.is_available():
        raise RuntimeError("GPU backend requires an available CUDA device")
    return NumpyBackend(xp=cp, is_gpu=True)
