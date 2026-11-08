import numpy as np

from src.losses import MulticlassFocalLoss


def test_multiclass_focal_loss_reduces_the_weight_of_easy_examples():
    loss = MulticlassFocalLoss(gamma=2.0, alpha=0.25)
    y_true = np.array([[1.0, 0.0], [1.0, 0.0]])
    y_pred = np.array([[0.99, 0.01], [0.6, 0.4]])

    value = loss.forward(y_true, y_pred)

    expected = np.mean(
        [
            -0.25 * (1.0 - 0.99) ** 2 * np.log(0.99),
            -0.25 * (1.0 - 0.6) ** 2 * np.log(0.6),
        ]
    )
    assert np.isclose(value, expected)


def test_multiclass_focal_loss_backward_matches_finite_difference():
    loss = MulticlassFocalLoss(gamma=2.0, alpha=0.25)
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.7, 0.3], [0.4, 0.6]])
    loss.forward(y_true, y_pred)
    gradient = loss.backward()

    epsilon = 1e-6
    numerical = np.zeros_like(y_pred)
    for row in range(y_pred.shape[0]):
        for column in range(y_pred.shape[1]):
            plus = y_pred.copy()
            minus = y_pred.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical[row, column] = (
                loss.forward(y_true, plus) - loss.forward(y_true, minus)
            ) / (2 * epsilon)

    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6)
