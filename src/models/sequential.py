from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import Array


@dataclass
class Sequential:
    layers: list

    @property
    def trainable_layers(self) -> list:
        return [
            layer
            for layer in self.layers
            if hasattr(layer, "W") and hasattr(layer, "dW")
        ]

    def forward(self, X: Array) -> Array:
        output = np.asarray(X, dtype=float)
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, dA: Array) -> Array:
        grad = np.asarray(dA, dtype=float)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def apply_gradients(self, lr: float) -> None:
        for layer in self.layers:
            if hasattr(layer, "apply_gradients"):
                layer.apply_gradients(lr)

    def predict_proba(self, X: Array) -> Array:
        return self.forward(X)

    def predict(self, X: Array, threshold: float = 0.5) -> Array:
        return (self.predict_proba(X) >= threshold).astype(float)
