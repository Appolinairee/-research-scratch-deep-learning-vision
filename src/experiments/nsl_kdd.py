from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from src.core.types import Array
from src.data.nsl_kdd import NSLKDD_CLASSES, NSLKDDData
from src.evaluation.classification import ClassificationReport, classification_report
from src.optim import Adam
from src.training.history import TrainingHistory
from src.training.sequential import train_multiclass_classifier

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


@dataclass(frozen=True)
class NSLKDDComparisonResult:
    name: str
    model: Sequential
    history: TrainingHistory
    report: ClassificationReport
    probabilities: np.ndarray
    hidden_activations: np.ndarray
    elapsed_seconds: float


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


def build_nsl_kdd_losses(
    config: NSLKDDModelConfig,
    backend: NumpyBackend | None = None,
) -> dict[str, MulticlassCrossEntropy | MulticlassFocalLoss]:
    backend = backend or NumpyBackend()
    return {
        "cross_entropy": build_nsl_kdd_cross_entropy(backend=backend),
        "focal_loss": build_nsl_kdd_loss(config=config, backend=backend),
    }


def extract_penultimate_activation(model: Sequential, X: Array) -> np.ndarray:
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    if len(dense_layers) < 2:
        raise ValueError("penultimate activation requires at least two dense layers")

    model.eval()
    activation = model.backend.asarray(X)
    last_dense = dense_layers[-1]
    penultimate = None
    for layer in model.layers:
        if hasattr(layer, "set_training"):
            layer.set_training(False)
        if layer is last_dense:
            penultimate = activation
        activation = layer.forward(activation)
    if penultimate is None:
        raise RuntimeError("penultimate activation could not be captured")
    return np.asarray(model.backend.to_host(penultimate))


def run_nsl_kdd_comparison(
    data: NSLKDDData,
    config: NSLKDDModelConfig,
    backend: NumpyBackend | None = None,
    seed: int = 42,
) -> dict[str, NSLKDDComparisonResult]:
    backend = backend or NumpyBackend()
    results = {}
    losses = build_nsl_kdd_losses(config=config, backend=backend)

    for name, loss in losses.items():
        model = build_nsl_kdd_model(
            config=config,
            input_dim=data.X_train.shape[1],
            backend=backend,
        )
        optimizer = Adam(lr=config.learning_rate, decay=0.004)
        start_time = time.perf_counter()
        history = train_multiclass_classifier(
            model=model,
            X=data.X_train,
            y=data.y_train,
            loss=loss,
            optimizer=optimizer,
            epochs=config.epochs,
            batch_size=config.batch_size,
            shuffle=True,
            seed=seed,
            validation_data=(data.X_validation, data.y_validation),
            early_stopping_patience=25,
        )
        backend.synchronize()
        elapsed_seconds = time.perf_counter() - start_time
        report = classification_report(
            model=model,
            X=data.X_test,
            y=data.y_test,
            class_count=len(NSLKDD_CLASSES),
            batch_size=config.batch_size,
        )
        probabilities = np.asarray(model.backend.to_host(
            model.predict_proba(data.X_test)
        ))
        hidden_activations = extract_penultimate_activation(model, data.X_test)
        results[name] = NSLKDDComparisonResult(
            name=name,
            model=model,
            history=history,
            report=report,
            probabilities=probabilities,
            hidden_activations=hidden_activations,
            elapsed_seconds=elapsed_seconds,
        )
    return results
