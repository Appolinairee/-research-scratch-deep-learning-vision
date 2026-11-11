from src.activations import Dropout, ReLU, Softmax
from src.experiments.nsl_kdd import (
    NSLKDDModelConfig,
    build_nsl_kdd_cross_entropy,
    build_nsl_kdd_loss,
    build_nsl_kdd_model,
)
from src.losses import MulticlassCrossEntropy
from src.layers import Dense


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
