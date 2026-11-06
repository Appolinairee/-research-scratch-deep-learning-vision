from __future__ import annotations

from dataclasses import dataclass

from src.core.types import Array
from src.losses.multiclass_cross_entropy import MulticlassCrossEntropy
from src.metrics.classification import multiclass_accuracy
from src.models.sequential import Sequential


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float


def evaluate_multiclass(
    model: Sequential,
    X: Array,
    y: Array,
    loss: MulticlassCrossEntropy,
    batch_size: int | None = None,
) -> ClassificationMetrics:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")

    X = model.backend.asarray(X)
    y = model.backend.asarray(y)
    batch_size = len(X) if batch_size is None else batch_size
    total_loss = 0.0
    total_correct = 0.0

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        predictions = model.predict_proba(X[start:end])
        current_size = min(end, len(X)) - start
        total_loss += loss.forward(
            y[start:end], predictions
        ) * current_size
        total_correct += (
            multiclass_accuracy(
                y[start:end], predictions, backend=model.backend
            )
            * current_size
        )

    sample_count = len(X)
    return ClassificationMetrics(
        loss=total_loss / sample_count,
        accuracy=total_correct / sample_count,
    )
