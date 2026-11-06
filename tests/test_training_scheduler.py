import numpy as np

from src.activations import Softmax
from src.layers import Dense
from src.losses import MulticlassCrossEntropy
from src.models import Sequential
from src.optim import ReduceLROnPlateau, SGD
from src.training.sequential import train_multiclass_classifier


def test_training_applies_scheduler_to_validation_loss():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    model = Sequential([Dense(2, 2, seed=0), Softmax()])
    optimizer = SGD(lr=0.1)

    train_multiclass_classifier(
        model=model,
        X=X,
        y=y,
        loss=MulticlassCrossEntropy(),
        optimizer=optimizer,
        epochs=2,
        batch_size=2,
        validation_data=(X, y),
        scheduler=ReduceLROnPlateau(patience=1, factor=0.5),
    )

    assert optimizer.lr <= 0.1


def test_training_uses_optimizer_state_for_momentum():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    model = Sequential([Dense(2, 2, seed=0), Softmax()])
    optimizer = SGD(lr=0.1, momentum=0.9)

    train_multiclass_classifier(
        model=model,
        X=X,
        y=y,
        loss=MulticlassCrossEntropy(),
        optimizer=optimizer,
        epochs=1,
        batch_size=2,
    )

    assert len(optimizer._velocities) == 2


def test_training_evaluates_in_batches():
    X = np.eye(4)
    y = np.eye(4)
    model = Sequential([Dense(4, 4, seed=0), Softmax()])

    history = train_multiclass_classifier(
        model=model,
        X=X,
        y=y,
        loss=MulticlassCrossEntropy(),
        optimizer=SGD(lr=0.1),
        epochs=1,
        batch_size=2,
        validation_data=(X, y),
    )

    assert len(history) == 1
