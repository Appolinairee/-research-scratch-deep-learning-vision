import numpy as np

from src.activations import sigmoid
from src.layers import SingleNeuron
from src.optim import SGD
from src.training.single_neuron import (
    TrainingStep,
    build_batches,
    run_single_neuron_training,
)


def test_build_batches_returns_full_dataset_for_batch_gradient_descent():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)

    batches = build_batches(X, y, batch_size=6, rng=np.random.default_rng(0), shuffle=True)

    assert len(batches) == 1
    np.testing.assert_allclose(batches[0][0], X)
    np.testing.assert_allclose(batches[0][1], y)


def test_build_batches_splits_dataset_into_minibatches():
    X = np.arange(18, dtype=float).reshape(9, 2)
    y = np.arange(9, dtype=float)

    batches = build_batches(X, y, batch_size=4, rng=np.random.default_rng(0), shuffle=False)

    assert [len(batch_y) for _, batch_y in batches] == [4, 4, 1]


def test_run_single_neuron_training_reduces_cost_for_full_batch():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = SingleNeuron(w=np.array([0.0]), b=0.0)

    history = run_single_neuron_training(
        X,
        y,
        model=model,
        optimizer=SGD(lr=0.5),
        epochs=12,
        batch_size=len(X),
        shuffle=False,
        seed=0,
    )

    assert history[0].loss > history[-1].loss


def test_run_single_neuron_training_returns_structured_history():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = SingleNeuron(w=np.array([-2.5]), b=2.0)

    history = run_single_neuron_training(
        X,
        y,
        model=model,
        optimizer=SGD(lr=0.1),
        epochs=2,
        batch_size=2,
        shuffle=False,
        seed=0,
    )

    assert len(history) == 4
    assert isinstance(history[0], TrainingStep)
    assert history[0].optimizer_name == "SGD"
    assert history[0].epoch == 0
    assert history[-1].epoch == 1


def test_run_single_neuron_training_matches_manual_probability_progression():
    X = np.linspace(-3.0, 3.0, 20).reshape(-1, 1)
    y = (sigmoid(2.0 * X[:, 0] - 0.3) > 0.5).astype(float)
    model = SingleNeuron(w=np.array([-2.5]), b=2.0)

    history = run_single_neuron_training(
        X,
        y,
        model=model,
        optimizer=SGD(lr=0.2),
        epochs=4,
        batch_size=5,
        shuffle=True,
        seed=3,
    )

    assert all(0.0 <= step.proba_at_zero <= 1.0 for step in history)
