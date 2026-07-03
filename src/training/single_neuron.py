from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import Array
from src.layers.single_neuron import SingleNeuron, gradients, predict_proba
from src.optim.sgd import SGD


@dataclass(frozen=True)
class TrainingStep:
    optimizer_name: str
    epoch: int
    update: int
    batch_size: int
    w: float
    b: float
    loss: float
    grad_norm: float
    proba_at_zero: float


def build_batches(
    X: Array,
    y: Array,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> list[tuple[Array, Array]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    sample_count = len(X)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size >= sample_count:
        return [(X, y)]

    indices = np.arange(sample_count)
    if shuffle:
        indices = rng.permutation(sample_count)

    batches: list[tuple[Array, Array]] = []
    for start in range(0, sample_count, batch_size):
        batch_indices = indices[start:start + batch_size]
        batches.append((X[batch_indices], y[batch_indices]))
    return batches


def run_single_neuron_training(
    X: Array,
    y: Array,
    model: SingleNeuron,
    optimizer: SGD,
    epochs: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[TrainingStep]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    rng = np.random.default_rng(seed)
    history: list[TrainingStep] = []
    current_model = model
    update_count = 0

    for epoch in range(epochs):
        batches = build_batches(X, y, batch_size=batch_size, rng=rng, shuffle=shuffle)

        for X_batch, y_batch in batches:
            grad_w, grad_b = gradients(X_batch, y_batch, current_model.w, current_model.b)
            next_w = optimizer.step(current_model.w, grad_w)
            next_b = optimizer.step(current_model.b, grad_b)
            current_model = SingleNeuron(w=np.asarray(next_w, dtype=float), b=float(next_b))

            history.append(
                TrainingStep(
                    optimizer_name=type(optimizer).__name__,
                    epoch=epoch,
                    update=update_count,
                    batch_size=len(X_batch),
                    w=float(current_model.w[0]),
                    b=current_model.b,
                    loss=current_model.cost(X, y),
                    grad_norm=float(np.sqrt(np.sum(grad_w ** 2) + grad_b ** 2)),
                    proba_at_zero=float(
                        predict_proba(np.zeros(current_model.w.shape, dtype=float), current_model.w, current_model.b)
                    ),
                )
            )
            update_count += 1

    return history
