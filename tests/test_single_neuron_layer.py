import numpy as np

from src.layers.single_neuron import (
    SingleNeuron,
    dataset_cost,
    gradient_step,
    gradients,
    linear_score,
    predict_label,
    predict_proba,
)
from src.optim import SGD


def _finite_difference_gradients(X, y, w, b, epsilon=1e-6):
    grad_w = np.zeros_like(w, dtype=float)

    for idx in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[idx] += epsilon
        w_minus[idx] -= epsilon
        cost_plus = dataset_cost(X, y, w_plus, b)
        cost_minus = dataset_cost(X, y, w_minus, b)
        grad_w[idx] = (cost_plus - cost_minus) / (2.0 * epsilon)

    cost_plus_b = dataset_cost(X, y, w, b + epsilon)
    cost_minus_b = dataset_cost(X, y, w, b - epsilon)
    grad_b = (cost_plus_b - cost_minus_b) / (2.0 * epsilon)

    return grad_w, grad_b


def test_linear_score_matches_affine_form():
    x = np.array([1.0, 2.0])
    w = np.array([3.0, -1.0])
    b = 0.5
    assert linear_score(x, w, b) == 1.5


def test_predict_proba_returns_probability():
    x = np.array([0.5, -0.25])
    w = np.array([2.0, 1.0])
    b = -0.1
    p = predict_proba(x, w, b)
    assert 0.0 < p < 1.0


def test_gradients_match_closed_form_on_tiny_dataset():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([1.0, 0.0])
    w = np.zeros(2)
    b = 0.0
    grad_w, grad_b = gradients(X, y, w, b)
    np.testing.assert_allclose(grad_w, np.array([-0.25, 0.25]))
    assert grad_b == 0.0


def test_gradients_match_finite_differences():
    X = np.array([
        [0.2, -1.1],
        [1.4, 0.3],
        [-0.7, 0.8],
        [0.5, 1.2],
    ])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    w = np.array([0.4, -0.6])
    b = 0.2

    analytic_w, analytic_b = gradients(X, y, w, b)
    numeric_w, numeric_b = _finite_difference_gradients(X, y, w, b)

    np.testing.assert_allclose(analytic_w, numeric_w, rtol=1e-5, atol=1e-6)
    assert np.isclose(analytic_b, numeric_b, rtol=1e-5, atol=1e-6)


def test_gradient_step_reduces_cost_on_simple_dataset():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    w = np.array([0.0])
    b = 0.0
    before = dataset_cost(X, y, w, b)
    next_w, next_b = gradient_step(X, y, w, b, lr=0.5)
    after = dataset_cost(X, y, next_w, next_b)
    assert after < before


def test_predict_label_applies_threshold():
    x = np.array([1.0])
    w = np.array([10.0])
    b = -2.0
    assert predict_label(x, w, b) == 1


def test_single_neuron_train_step_returns_new_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = SingleNeuron(w=np.array([0.0]), b=0.0)
    next_model = model.train_step(X, y, lr=0.5)
    assert isinstance(next_model, SingleNeuron)
    assert next_model.cost(X, y) < model.cost(X, y)


def test_single_neuron_train_step_with_optimizer_reduces_cost():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = SingleNeuron(w=np.array([0.0]), b=0.0)
    optimizer = SGD(lr=0.5)

    next_model = model.train_step_with(X, y, optimizer)

    assert next_model.cost(X, y) < model.cost(X, y)
