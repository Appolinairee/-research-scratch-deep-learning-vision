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
) -> ClassificationMetrics:
    predictions = model.predict_proba(model.backend.asarray(X))
    return ClassificationMetrics(
        loss=loss.forward(y, predictions),
        accuracy=multiclass_accuracy(y, predictions, backend=model.backend),
    )
