import numpy as np

from src.activations import Sigmoid


def test_sigmoid_layer_forward_returns_probabilities():
    activation = Sigmoid()
    Z = np.array([[-2.0, 0.0, 2.0]])

    out = activation.forward(Z)

    assert out.shape == Z.shape
    assert np.all(out > 0.0)
    assert np.all(out < 1.0)


def test_sigmoid_layer_backward_matches_closed_form_derivative():
    activation = Sigmoid()
    Z = np.array([[-1.0, 0.0, 1.0]])
    dA = np.array([[0.5, 2.0, -3.0]])

    A = activation.forward(Z)
    dZ = activation.backward(dA)

    np.testing.assert_allclose(dZ, dA * A * (1.0 - A))
