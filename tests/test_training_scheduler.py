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
