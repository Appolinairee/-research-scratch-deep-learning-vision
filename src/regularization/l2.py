from __future__ import annotations

import numpy as np

from src.models.sequential import Sequential


class L2Regularization:
    def __init__(self, strength: float) -> None:
        if strength < 0:
            raise ValueError("strength must be non-negative")
        self.strength = float(strength)

    def penalty(self, model: Sequential) -> float:
        return float(
            0.5
            * self.strength
            * sum(np.sum(layer.W**2) for layer in model.trainable_layers)
        )

    def add_to_gradients(self, model: Sequential) -> None:
        for layer in model.trainable_layers:
            layer.dW += self.strength * layer.W
