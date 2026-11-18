from src.activations import Dropout, ReLU, Softmax
from src.experiments.nsl_kdd import (
    NSLKDDModelConfig,
    build_nsl_kdd_cross_entropy,
    build_nsl_kdd_loss,
    build_nsl_kdd_model,
)
from src.losses import MulticlassCrossEntropy
from src.layers import Dense
from src.experiments.nsl_kdd import build_nsl_kdd_losses
from src.losses import MulticlassFocalLoss
from src.experiments.nsl_kdd import extract_penultimate_activation
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.models import Sequential
import numpy as np


def test_build_nsl_kdd_model_uses_article_architecture():
    model = build_nsl_kdd_model(NSLKDDModelConfig(), input_dim=41)
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]

    assert [
        dense_layers[0].input_dim,
        *(layer.output_dim for layer in dense_layers),
    ] == [41, 512, 256, 128, 5]
    assert sum(isinstance(layer, ReLU) for layer in model.layers) == 3
    assert sum(isinstance(layer, Dropout) for layer in model.layers) == 3
    assert isinstance(model.layers[-1], Softmax)


def test_build_nsl_kdd_loss_uses_article_focal_defaults():
    loss = build_nsl_kdd_loss(NSLKDDModelConfig())

    assert loss.gamma == 2.0
    assert loss.alpha == 0.25


def test_build_nsl_kdd_cross_entropy_returns_reference_loss():
    assert isinstance(build_nsl_kdd_cross_entropy(), MulticlassCrossEntropy)


def test_build_nsl_kdd_losses_returns_both_comparison_losses():
    losses = build_nsl_kdd_losses(NSLKDDModelConfig())

    assert tuple(losses) == ("cross_entropy", "focal_loss")
    assert isinstance(losses["cross_entropy"], MulticlassCrossEntropy)
    assert isinstance(losses["focal_loss"], MulticlassFocalLoss)


def test_extract_penultimate_activation_returns_output_before_last_dense():
    model = Sequential([
        Dense(input_dim=3, output_dim=4, seed=1),
        ReLU(),
        Dense(input_dim=4, output_dim=2, seed=2),
        Softmax(),
    ])
    X = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])

    activation = extract_penultimate_activation(model, X)

    assert activation.shape == (2, 4)
    assert np.isfinite(activation).all()
