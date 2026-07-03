import numpy as np

from src.losses import BinaryCrossEntropy


def test_binary_cross_entropy_forward_matches_known_value():
    loss = BinaryCrossEntropy()
    y_true = np.array([[1.0], [0.0]])
    y_pred = np.array([[0.8], [0.3]])

    value = loss.forward(y_true, y_pred)

    expected = -0.5 * (np.log(0.8) + np.log(0.7))
    assert np.isclose(value, expected)


def test_binary_cross_entropy_backward_matches_closed_form():
    loss = BinaryCrossEntropy()
    y_true = np.array([[1.0], [0.0]])
    y_pred = np.array([[0.8], [0.3]])

    loss.forward(y_true, y_pred)
    grad = loss.backward()

    expected = -0.5 * ((y_true / y_pred) - ((1.0 - y_true) / (1.0 - y_pred)))
    np.testing.assert_allclose(grad, expected)
