from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.backend import NumpyBackend
from src.core.types import Array
from src.optim import SGD


@dataclass
class Sequential:
    layers: list
    training: bool = True

    @property
    def backend(self) -> NumpyBackend:
        for layer in self.layers:
            if hasattr(layer, "backend"):
                return layer.backend
        return NumpyBackend()

    @property
    def trainable_layers(self) -> list:
        return [
            layer
            for layer in self.layers
            if hasattr(layer, "W") and hasattr(layer, "dW")
        ]

    def forward(self, X: Array) -> Array:
        output = self.backend.asarray(X)
        for layer in self.layers:
            if hasattr(layer, "set_training"):
                layer.set_training(self.training)
            output = layer.forward(output)
        return output

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def backward(self, dA: Array) -> Array:
        grad = self.backend.asarray(dA)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def apply_gradients(self, optimizer: SGD | float) -> None:
        for layer in self.layers:
            if hasattr(layer, "apply_gradients"):
                layer.apply_gradients(optimizer)

    def predict_proba(self, X: Array) -> Array:
        self.eval()
        return self.forward(X)

    def predict(self, X: Array, threshold: float = 0.5) -> Array:
        return (self.predict_proba(X) >= threshold).astype(float)
