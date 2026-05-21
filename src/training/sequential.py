from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import Array
from src.losses.binary_cross_entropy import BinaryCrossEntropy
from src.losses.multiclass_cross_entropy import MulticlassCrossEntropy
from src.metrics.classification import binary_accuracy, multiclass_accuracy
from src.models.sequential import Sequential
from src.optim.sgd import SGD
from src.training.single_neuron import build_batches


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    accuracy: float


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
) -> list[EpochMetrics]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    history: list[EpochMetrics] = []

    for epoch in range(epochs):
        batches = build_batches(X, y, batch_size=batch_size, rng=rng, shuffle=shuffle)

        for X_batch, y_batch in batches:
            y_pred = model.forward(X_batch)
            loss.forward(y_batch, y_pred)
            dA = loss.backward()
            model.backward(dA)
            model.apply_gradients(optimizer.lr)

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
) -> list[EpochMetrics]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    history: list[EpochMetrics] = []

    for epoch in range(epochs):
        batches = build_batches(X, y, batch_size=batch_size, rng=rng, shuffle=shuffle)

        for X_batch, y_batch in batches:
            y_pred = model.forward(X_batch)
            loss.forward(y_batch, y_pred)
            dA = loss.backward()
            model.backward(dA)
            model.apply_gradients(optimizer.lr)

        full_pred = model.forward(X)
        history.append(
            EpochMetrics(
                epoch=epoch,
                loss=loss.forward(y, full_pred),
                accuracy=multiclass_accuracy(y, full_pred),
            )
        )

    return history
