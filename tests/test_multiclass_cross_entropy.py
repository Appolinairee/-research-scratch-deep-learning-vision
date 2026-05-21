import numpy as np

from src.losses import MulticlassCrossEntropy


def test_multiclass_cross_entropy_forward_matches_known_value():
    loss = MulticlassCrossEntropy()
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

    value = loss.forward(y_true, y_pred)

    expected = -0.5 * (np.log(0.7) + np.log(0.8))
    assert np.isclose(value, expected)


def test_multiclass_cross_entropy_backward_matches_closed_form():
    loss = MulticlassCrossEntropy()
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

    loss.forward(y_true, y_pred)
    grad = loss.backward()

    expected = -(y_true / y_pred) / len(y_true)
    np.testing.assert_allclose(grad, expected)
