from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import Array
from src.losses.multiclass_cross_entropy import MulticlassCrossEntropy
from src.metrics.classification import multiclass_accuracy
from src.models.sequential import Sequential


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    confusion_matrix: np.ndarray


def classification_report(
    model: Sequential,
    X: Array,
    y: Array,
    class_count: int,
    batch_size: int | None = None,
) -> ClassificationReport:
    if class_count <= 0:
        raise ValueError("class_count must be positive")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(X) == 0:
        raise ValueError("classification report requires at least one sample")
    X = model.backend.asarray(X)
    y_true = np.asarray(model.backend.to_host(y)).argmax(axis=1)
    batch_size = len(X) if batch_size is None else batch_size
    predictions = []
    for start in range(0, len(X), batch_size):
        predictions.append(
            model.backend.to_host(model.predict_proba(X[start : start + batch_size]))
        )
    y_pred = np.concatenate(predictions).argmax(axis=1)
    matrix = np.zeros((class_count, class_count), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    true_positive = np.diag(matrix).astype(float)
    predicted_positive = matrix.sum(axis=0)
    actual_positive = matrix.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive),
        where=predicted_positive != 0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros_like(true_positive),
        where=actual_positive != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return ClassificationReport(
        accuracy=float(np.mean(y_true == y_pred)),
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=matrix,
    )


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
