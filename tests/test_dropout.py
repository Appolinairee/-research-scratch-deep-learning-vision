import numpy as np

from src.activations import Dropout


def test_dropout_scales_surviving_activations_during_training():
    layer = Dropout(rate=0.5, seed=0)
    values = np.ones((1000, 4))

    output = layer.forward(values, training=True)

    assert set(np.unique(output)).issubset({0.0, 2.0})
    assert 0.9 < output.mean() < 1.1


def test_dropout_is_identity_during_evaluation():
    layer = Dropout(rate=0.5, seed=0)
    values = np.ones((2, 3))

    np.testing.assert_array_equal(layer.forward(values, training=False), values)
