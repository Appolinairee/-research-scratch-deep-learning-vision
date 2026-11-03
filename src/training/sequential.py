from __future__ import annotations

import numpy as np

from src.core.types import Array
from src.losses.binary_cross_entropy import BinaryCrossEntropy
from src.losses.multiclass_cross_entropy import MulticlassCrossEntropy
from src.metrics.classification import binary_accuracy, multiclass_accuracy
from src.models.sequential import Sequential
from src.optim import ReduceLROnPlateau, SGD
from src.training.single_neuron import build_batches
from src.evaluation.classification import evaluate_multiclass
from src.training.history import EpochMetrics, TrainingHistory
from src.training.diagnostics import collect_diagnostics
from src.regularization import L2Regularization


def train_binary_classifier(
    model: Sequential,
    X: Array,
    y: Array,
    loss: BinaryCrossEntropy,
    optimizer: SGD,
    epochs: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> TrainingHistory:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    history = TrainingHistory()

    for epoch in range(epochs):
        batches = build_batches(X, y, batch_size=batch_size, rng=rng, shuffle=shuffle)

        for X_batch, y_batch in batches:
            y_pred = model.forward(X_batch)
            loss.forward(y_batch, y_pred)
            dA = loss.backward()
            model.backward(dA)
            model.apply_gradients(optimizer)

        full_pred = model.forward(X)
        history.append(
            EpochMetrics(
                epoch=epoch,
                loss=loss.forward(y, full_pred),
                accuracy=binary_accuracy(y, full_pred),
            )
        )

    return history


def train_multiclass_classifier(
    model: Sequential,
    X: Array,
    y: Array,
    loss: MulticlassCrossEntropy,
    optimizer: SGD,
    epochs: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
    validation_data: tuple[Array, Array] | None = None,
    regularizer: L2Regularization | None = None,
    scheduler: ReduceLROnPlateau | None = None,
) -> TrainingHistory:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    backend = model.backend
    X = backend.asarray(X)
    y = backend.asarray(y)
    if validation_data is not None:
        X_validation = backend.asarray(validation_data[0])
        y_validation = backend.asarray(validation_data[1])
    else:
        X_validation = y_validation = None
    rng = np.random.default_rng(seed)
    history = TrainingHistory()

    for epoch in range(epochs):
        model.train()
        sample_count = len(X)
        indices = np.arange(sample_count)
        if shuffle:
            indices = rng.permutation(sample_count)
        batches = []
        for start in range(0, sample_count, batch_size):
            batch_indices = indices[start:start + batch_size]
            batches.append((X[batch_indices], y[batch_indices]))

        for X_batch, y_batch in batches:
            y_pred = model.forward(X_batch)
            loss.forward(y_batch, y_pred)
            dA = loss.backward()
            model.backward(dA)
            if regularizer is not None:
                regularizer.add_to_gradients(model)
            model.apply_gradients(optimizer)

        model.eval()
        train_metrics = evaluate_multiclass(model, X, y, loss)
        diagnostics = collect_diagnostics(model)
        train_loss = train_metrics.loss
        if regularizer is not None:
            train_loss += regularizer.penalty(model)
        validation_metrics = None
        if X_validation is not None and y_validation is not None:
            validation_metrics = evaluate_multiclass(
                model, X_validation, y_validation, loss
            )
            if scheduler is not None:
                scheduler.step(validation_metrics.loss, optimizer)
        history.append(
            EpochMetrics(
                epoch=epoch,
                loss=train_loss,
                accuracy=train_metrics.accuracy,
                validation_loss=(
                    validation_metrics.loss if validation_metrics else None
                ),
                validation_accuracy=(
                    validation_metrics.accuracy if validation_metrics else None
                ),
                weight_norm=diagnostics.weight_norm,
                gradient_norm=diagnostics.gradient_norm,
                relu_zero_fraction=diagnostics.relu_zero_fraction,
            )
        )

    return history
