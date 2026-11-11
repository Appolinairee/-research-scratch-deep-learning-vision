from __future__ import annotations

from dataclasses import dataclass

from src.activations import Dropout, ReLU, Softmax
from src.core.backend import NumpyBackend
from src.layers import Dense
from src.losses import MulticlassCrossEntropy, MulticlassFocalLoss
from src.models import Sequential


@dataclass(frozen=True)
class NSLKDDModelConfig:
    hidden_units: tuple[int, ...] = (512, 256, 128)
    dropout_rate: float = 0.01
    learning_rate: float = 1e-4
    momentum: float = 0.9
    l2_strength: float = 1e-4
    epochs: int = 250
    batch_size: int = 64
    gamma: float = 2.0
    alpha: float = 0.25

    def __post_init__(self) -> None:
        if not self.hidden_units or any(units <= 0 for units in self.hidden_units):
            raise ValueError("hidden_units must contain positive sizes")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")


def build_nsl_kdd_model(
    config: NSLKDDModelConfig,
    input_dim: int,
    backend: NumpyBackend | None = None,
) -> Sequential:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    backend = backend or NumpyBackend()
    layers = []
    previous_dim = input_dim
    for index, hidden_dim in enumerate(config.hidden_units):
        layers.extend(
            [
                Dense(
                    input_dim=previous_dim,
                    output_dim=hidden_dim,
                    seed=index,
                    backend=backend,
                ),
                ReLU(backend=backend),
                Dropout(
                    rate=config.dropout_rate,
                    seed=10 + index,
                    backend=backend,
                ),
            ]
        )
        previous_dim = hidden_dim
    layers.extend(
        [
            Dense(
                input_dim=previous_dim,
                output_dim=5,
                seed=100,
                backend=backend,
            ),
            Softmax(backend=backend),
        ]
    )
    return Sequential(layers)


def build_nsl_kdd_loss(
    config: NSLKDDModelConfig,
    backend: NumpyBackend | None = None,
) -> MulticlassFocalLoss:
    return MulticlassFocalLoss(
        gamma=config.gamma,
        alpha=config.alpha,
        backend=backend or NumpyBackend(),
    )


def build_nsl_kdd_cross_entropy(
    backend: NumpyBackend | None = None,
) -> MulticlassCrossEntropy:
    return MulticlassCrossEntropy(backend=backend or NumpyBackend())
