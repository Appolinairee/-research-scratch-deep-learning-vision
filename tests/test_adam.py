import numpy as np

from src.optim import Adam


def test_adam_applies_bias_corrected_update():
    optim = Adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    param = np.array([1.0, -2.0])
    grad = np.array([0.5, -1.0])

    updated = optim.step(param, grad)

    np.testing.assert_allclose(
        updated,
        np.array([0.999, -1.999]),
        rtol=1e-6,
        atol=1e-8,
    )


def test_adam_keeps_state_separate_for_each_parameter():
    optim = Adam(lr=0.001)
    first_param = np.array([1.0])
    second_param = np.array([1.0])

    first = optim.step(first_param, np.array([1.0]))
    second = optim.step(second_param, np.array([2.0]))

    np.testing.assert_allclose(first, np.array([0.999]))
    np.testing.assert_allclose(second, np.array([0.999]))


def test_adam_rejects_invalid_hyperparameters():
    for kwargs in (
        {"lr": -1.0},
        {"lr": 0.1, "beta1": 1.0},
        {"lr": 0.1, "beta2": 1.0},
        {"lr": 0.1, "epsilon": 0.0},
    ):
        try:
            Adam(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid parameters accepted: {kwargs}")


def test_adam_applies_learning_rate_decay():
    optim = Adam(lr=0.1, decay=0.1)
    param = np.array([1.0])
    gradient = np.array([1.0])

    first = optim.step(param, gradient)
    second = optim.step(first, gradient)

    np.testing.assert_allclose(first, np.array([0.9]), rtol=1e-6)
    np.testing.assert_allclose(second, np.array([0.8090909]), rtol=1e-6)
