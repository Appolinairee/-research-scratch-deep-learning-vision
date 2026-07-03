import numpy as np

from src.activations import sigmoid


def test_sigmoid_maps_zero_to_half():
    assert sigmoid(0.0) == 0.5


def test_sigmoid_is_bounded_between_zero_and_one():
    values = sigmoid(np.array([-100.0, 0.0, 100.0]))
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
