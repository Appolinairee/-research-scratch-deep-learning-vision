from __future__ import annotations

from dataclasses import dataclass

from src.activations.relu import ReLU
from src.models.sequential import Sequential


@dataclass(frozen=True)
class TrainingDiagnostics:
    weight_norm: float
    gradient_norm: float
    relu_zero_fraction: float


def collect_diagnostics(model: Sequential) -> TrainingDiagnostics:
    trainable_layers = model.trainable_layers
    weight_norm = model.backend.scalar(
        model.backend.sqrt(
            sum(model.backend.sum(layer.W**2) for layer in trainable_layers)
        )
    )
    gradient_norm = model.backend.scalar(
        model.backend.sqrt(
            sum(model.backend.sum(layer.dW**2) for layer in trainable_layers)
        )
    )

    relu_layers = [
        layer for layer in model.layers if isinstance(layer, ReLU)
    ]
    if relu_layers:
        relu_zero_fraction = sum(
            layer.zero_fraction() for layer in relu_layers
        ) / len(relu_layers)
    else:
        relu_zero_fraction = 0.0

    return TrainingDiagnostics(
        weight_norm=weight_norm,
        gradient_norm=gradient_norm,
        relu_zero_fraction=relu_zero_fraction,
    )
