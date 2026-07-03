import numpy as np

from src.optim import SGD, sgd_step


def test_sgd_step_updates_vector_parameter():
    param = np.array([1.0, -2.0])
    grad = np.array([0.5, -1.0])

    next_param = sgd_step(param, grad, lr=0.1)

    np.testing.assert_allclose(next_param, np.array([0.95, -1.9]))


def test_sgd_step_updates_scalar_parameter():
    next_param = sgd_step(1.0, 0.25, lr=0.4)
    assert next_param == 0.9


def test_sgd_object_delegates_update_rule():
    optim = SGD(lr=0.1)
    param = np.array([2.0, 4.0])
    grad = np.array([1.0, 3.0])

    next_param = optim.step(param, grad)

    np.testing.assert_allclose(next_param, np.array([1.9, 3.7]))
