import numpy as np

from src.activations import Softmax
from src.layers import Dense
from src.losses import MulticlassCrossEntropy
from src.models import Sequential
from src.optim import SGD
from src.training.sequential import train_multiclass_classifier


def test_training_stops_after_validation_patience():
    X = np.eye(2)
    y = np.eye(2)
    model = Sequential([Dense(2, 2, seed=0), Softmax()])

    history = train_multiclass_classifier(
        model=model,
        X=X,
        y=y,
        loss=MulticlassCrossEntropy(),
        optimizer=SGD(lr=0.0),
        epochs=10,
        batch_size=2,
        validation_data=(X, y),
        early_stopping_patience=2,
    )

    assert len(history) == 3
