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


def test_sgd_momentum_accumulates_per_parameter_updates():
    optim = SGD(lr=0.1, momentum=0.9)
    param = np.array([2.0, 4.0])
    grad = np.array([1.0, 3.0])

    first = optim.step(param, grad)
    second = optim.step(first, grad)

    np.testing.assert_allclose(first, np.array([1.99, 3.97]))
    np.testing.assert_allclose(second, np.array([1.971, 3.913]))


def test_sgd_rejects_invalid_momentum():
    try:
        SGD(lr=0.1, momentum=1.0)
    except ValueError:
        return
    raise AssertionError("momentum >= 1 must be rejected")
