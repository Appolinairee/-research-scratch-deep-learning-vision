import numpy as np

from src.layers.dense import Dense
from src.optim import SGD


def _finite_difference_parameter_gradients(layer, X, dZ, epsilon=1e-6):
    numeric_dW = np.zeros_like(layer.W)
    numeric_db = np.zeros_like(layer.b)

    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            plus = layer.W.copy()
            minus = layer.W.copy()
            plus[i, j] += epsilon
            minus[i, j] -= epsilon

            out_plus = X @ plus + layer.b
            out_minus = X @ minus + layer.b
            loss_plus = np.sum(out_plus * dZ)
            loss_minus = np.sum(out_minus * dZ)
            numeric_dW[i, j] = (loss_plus - loss_minus) / (2.0 * epsilon)

    for j in range(layer.b.shape[0]):
        plus = layer.b.copy()
        minus = layer.b.copy()
        plus[j] += epsilon
        minus[j] -= epsilon

        out_plus = X @ layer.W + plus
        out_minus = X @ layer.W + minus
        loss_plus = np.sum(out_plus * dZ)
        loss_minus = np.sum(out_minus * dZ)
        numeric_db[j] = (loss_plus - loss_minus) / (2.0 * epsilon)

    return numeric_dW, numeric_db


def test_dense_forward_returns_expected_shape():
    layer = Dense(input_dim=3, output_dim=2, seed=0)
    X = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]])

    Z = layer.forward(X)

    assert Z.shape == (2, 2)


def test_dense_backward_returns_expected_input_gradient_shape():
    layer = Dense(input_dim=3, output_dim=2, seed=0)
    X = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]])
    dZ = np.array([[1.0, -2.0], [0.5, 3.0]])

    layer.forward(X)
    dX = layer.backward(dZ)

    assert dX.shape == X.shape
    assert layer.dW.shape == layer.W.shape
    assert layer.db.shape == layer.b.shape


def test_dense_backward_matches_closed_form_for_affine_layer():
    layer = Dense(input_dim=2, output_dim=2, seed=0)
    layer.W = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.b = np.array([0.5, -1.0])

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    dZ = np.array([[2.0, -1.0], [3.0, 4.0]])

    layer.forward(X)
    dX = layer.backward(dZ)

    expected_dW = X.T @ dZ / len(X)
    expected_db = np.sum(dZ, axis=0) / len(X)
    expected_dX = dZ @ layer.W.T

    np.testing.assert_allclose(layer.dW, expected_dW)
    np.testing.assert_allclose(layer.db, expected_db)
    np.testing.assert_allclose(dX, expected_dX)


def test_dense_parameter_gradients_match_finite_differences():
    layer = Dense(input_dim=2, output_dim=3, seed=0)
    layer.W = np.array([[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]])
    layer.b = np.array([0.1, -0.2, 0.5])

    X = np.array([[1.0, 2.0], [-1.0, 0.5], [0.3, -0.7]])
    dZ = np.array([[0.5, -1.0, 0.2], [1.5, 0.1, -0.8], [-0.3, 0.4, 2.0]])

    layer.forward(X)
    layer.backward(dZ)
    numeric_dW, numeric_db = _finite_difference_parameter_gradients(layer, X, dZ)

    np.testing.assert_allclose(layer.dW, numeric_dW / len(X), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(layer.db, numeric_db / len(X), rtol=1e-5, atol=1e-6)


def test_dense_update_step_changes_parameters():
    layer = Dense(input_dim=2, output_dim=1, seed=0)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    dZ = np.array([[1.0], [2.0]])

    original_W = layer.W.copy()
    original_b = layer.b.copy()

    layer.forward(X)
    layer.backward(dZ)
    layer.apply_gradients(lr=0.1)

    assert not np.allclose(layer.W, original_W)
    assert not np.allclose(layer.b, original_b)


def test_dense_can_apply_optimizer_momentum():
    layer = Dense(input_dim=1, output_dim=1, seed=0)
    layer.dW = np.array([[1.0]])
    layer.db = np.array([1.0])
    optimizer = SGD(lr=0.1, momentum=0.9)
    initial_W = layer.W.copy()
    initial_b = layer.b.copy()

    layer.apply_gradients(optimizer)

    np.testing.assert_allclose(layer.W, initial_W - 0.01)
    np.testing.assert_allclose(layer.b, initial_b - 0.01)
