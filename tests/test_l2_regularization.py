import numpy as np

from src.layers import Dense
from src.models import Sequential
from src.regularization import L2Regularization


def test_l2_regularization_calculates_penalty_and_updates_weight_gradients():
    first = Dense(input_dim=2, output_dim=2, seed=0)
    second = Dense(input_dim=2, output_dim=1, seed=1)
    first.W = np.array([[1.0, -2.0], [0.5, 0.0]])
    second.W = np.array([[3.0], [-1.0]])
    first.dW = np.zeros_like(first.W)
    second.dW = np.zeros_like(second.W)
    model = Sequential([first, second])
    regularizer = L2Regularization(strength=0.1)

    penalty = regularizer.penalty(model)
    regularizer.add_to_gradients(model)

    np.testing.assert_allclose(penalty, 0.7625)
    np.testing.assert_allclose(first.dW, 0.1 * first.W)
    np.testing.assert_allclose(second.dW, 0.1 * second.W)
