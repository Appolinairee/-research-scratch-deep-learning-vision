import numpy as np

from src.activations import ReLU, Sigmoid
from src.layers import Dense
from src.losses import BinaryCrossEntropy
from src.models import Sequential
from src.optim import SGD
from src.training.sequential import train_binary_classifier


def test_sequential_forward_preserves_batch_shape():
    model = Sequential([Dense(input_dim=2, output_dim=3, seed=0), ReLU(), Dense(input_dim=3, output_dim=1, seed=1), Sigmoid()])
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    out = model.forward(X)

    assert out.shape == (3, 1)


def test_sequential_backward_propagates_without_shape_errors():
    model = Sequential([Dense(input_dim=2, output_dim=3, seed=0), ReLU(), Dense(input_dim=3, output_dim=1, seed=1), Sigmoid()])
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0]])
    loss = BinaryCrossEntropy()

    y_pred = model.forward(X)
    loss.forward(y, y_pred)
    dA = loss.backward()
    dX = model.backward(dA)

    assert dX.shape == X.shape


def test_binary_classifier_training_reduces_loss():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [0.0], [0.0], [1.0]])
    model = Sequential([Dense(input_dim=2, output_dim=4, seed=0), ReLU(), Dense(input_dim=4, output_dim=1, seed=1), Sigmoid()])

    history = train_binary_classifier(
        model=model,
        X=X,
        y=y,
        loss=BinaryCrossEntropy(),
        optimizer=SGD(lr=0.5),
        epochs=300,
        batch_size=4,
        shuffle=True,
        seed=0,
    )

    assert history[0].loss > history[-1].loss
    assert history[-1].accuracy >= 0.75


def test_binary_classifier_learns_xor():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    model = Sequential(
        [
            Dense(input_dim=2, output_dim=8, seed=0),
            ReLU(),
            Dense(input_dim=8, output_dim=1, seed=1),
            Sigmoid(),
        ]
    )

    history = train_binary_classifier(
        model=model,
        X=X,
        y=y,
        loss=BinaryCrossEntropy(),
        optimizer=SGD(lr=0.5),
        epochs=12000,
        batch_size=4,
        shuffle=True,
        seed=0,
    )

    predictions = model.predict(X)

    assert history[0].loss > history[-1].loss
    assert history[-1].accuracy >= 0.75
    assert np.sum(predictions == y) >= 3
