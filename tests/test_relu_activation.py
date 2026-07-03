import numpy as np

from src.activations import ReLU


def test_relu_forward_zeroes_negative_values():
    activation = ReLU()
    X = np.array([[-2.0, 0.0, 3.5], [4.0, -1.0, -0.2]])

    out = activation.forward(X)

    np.testing.assert_allclose(out, np.array([[0.0, 0.0, 3.5], [4.0, 0.0, 0.0]]))


def test_relu_backward_passes_gradient_only_on_positive_entries():
    activation = ReLU()
    X = np.array([[-2.0, 0.0, 3.5], [4.0, -1.0, -0.2]])
    dA = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    activation.forward(X)
    dZ = activation.backward(dA)

    np.testing.assert_allclose(dZ, np.array([[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]))
