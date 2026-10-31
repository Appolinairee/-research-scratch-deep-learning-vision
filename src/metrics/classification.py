from __future__ import annotations

import numpy as np

from src.core.backend import NumpyBackend
from src.core.types import Array


def binary_accuracy(
    y_true: Array,
    y_pred: Array,
    threshold: float = 0.5,
    backend: NumpyBackend | None = None,
) -> float:
    backend = backend or NumpyBackend()
    y_true = backend.asarray(y_true)
    y_pred = backend.asarray(y_pred)
    predicted = (y_pred >= threshold).astype(float)
    return backend.scalar(backend.mean(predicted == y_true))


def multiclass_accuracy(
    y_true: Array,
    y_pred: Array,
    backend: NumpyBackend | None = None,
) -> float:
    backend = backend or NumpyBackend()
    y_true = backend.asarray(y_true)
    y_pred = backend.asarray(y_pred)
    true_labels = backend.argmax(y_true, axis=1)
    predicted_labels = backend.argmax(y_pred, axis=1)
    return backend.scalar(backend.mean(predicted_labels == true_labels))
