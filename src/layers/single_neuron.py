from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.activations.sigmoid import sigmoid
from src.core.types import Array
from src.losses.binary_cross_entropy import dataset_binary_cross_entropy
from src.optim.sgd import SGD


def linear_score(x: Array, w: Array, b: float) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(x @ w + b)


def predict_proba(x: Array, w: Array, b: float) -> float:
    return float(sigmoid(linear_score(x, w, b)))


def predict_label(x: Array, w: Array, b: float, threshold: float = 0.5) -> int:
    return int(predict_proba(x, w, b) >= threshold)


def dataset_cost(X: Array, y: Array, w: Array, b: float) -> float:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(sigmoid(X @ w + b), dtype=float)
    return dataset_binary_cross_entropy(y, p)


def gradients(X: Array, y: Array, w: Array, b: float) -> tuple[Array, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m = X.shape[0]
    p = sigmoid(X @ w + b)
    error = p - y
    grad_w = (X.T @ error) / m
    grad_b = float(np.sum(error) / m)
    return grad_w, grad_b


def gradient_step(X: Array, y: Array, w: Array, b: float, lr: float) -> tuple[Array, float]:
    grad_w, grad_b = gradients(X, y, w, b)
    next_w = np.asarray(w, dtype=float) - lr * grad_w
    next_b = float(b - lr * grad_b)
    return next_w, next_b


@dataclass(frozen=True)
class SingleNeuron:
    w: Array
    b: float = 0.0

    def score(self, x: Array) -> float:
        return linear_score(x, self.w, self.b)

    def proba(self, x: Array) -> float:
        return predict_proba(x, self.w, self.b)

    def label(self, x: Array, threshold: float = 0.5) -> int:
        return predict_label(x, self.w, self.b, threshold=threshold)

    def cost(self, X: Array, y: Array) -> float:
        return dataset_cost(X, y, self.w, self.b)

    def train_step(self, X: Array, y: Array, lr: float) -> "SingleNeuron":
        next_w, next_b = gradient_step(X, y, self.w, self.b, lr=lr)
        return SingleNeuron(w=next_w, b=next_b)

    def train_step_with(self, X: Array, y: Array, optimizer: SGD) -> "SingleNeuron":
        grad_w, grad_b = gradients(X, y, self.w, self.b)
        next_w = optimizer.step(self.w, grad_w)
        next_b = optimizer.step(self.b, grad_b)
        return SingleNeuron(w=np.asarray(next_w, dtype=float), b=float(next_b))
