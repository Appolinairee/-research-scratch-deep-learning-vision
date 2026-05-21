import numpy as np

from src.activations import Softmax


def test_softmax_forward_returns_probabilities_per_row():
    activation = Softmax()
    Z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])

    out = activation.forward(Z)

    assert out.shape == Z.shape
    np.testing.assert_allclose(np.sum(out, axis=1), np.ones(Z.shape[0]))
    assert np.all(out > 0.0)


def test_softmax_backward_matches_jacobian_vector_product():
    activation = Softmax()
    Z = np.array([[1.0, 2.0, 3.0]])
    dA = np.array([[0.1, -0.2, 0.3]])

    out = activation.forward(Z)
    dZ = activation.backward(dA)

    expected = out * (dA - np.sum(dA * out, axis=1, keepdims=True))
    np.testing.assert_allclose(dZ, expected)
