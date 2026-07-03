import numpy as np

from src.activations import ReLU, Softmax
from src.layers import Dense
from src.losses import MulticlassCrossEntropy
from src.models import Sequential
from src.optim import SGD
from src.training.sequential import train_multiclass_classifier


def test_multiclass_sequential_forward_preserves_shape():
    model = Sequential(
        [
            Dense(input_dim=2, output_dim=5, seed=0),
            ReLU(),
            Dense(input_dim=5, output_dim=3, seed=1),
            Softmax(),
        ]
    )
    X = np.array([[0.0, 1.0], [1.0, 0.0]])

    out = model.forward(X)

    assert out.shape == (2, 3)
    np.testing.assert_allclose(np.sum(out, axis=1), np.ones(len(X)))


def test_multiclass_classifier_training_reduces_loss():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -0.5],
            [1.0, 2.0],
            [1.5, 1.0],
            [2.0, -2.0],
            [2.5, -1.5],
        ]
    )
    y = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    model = Sequential(
        [
            Dense(input_dim=2, output_dim=8, seed=0),
            ReLU(),
            Dense(input_dim=8, output_dim=3, seed=1),
            Softmax(),
        ]
    )

    history = train_multiclass_classifier(
        model=model,
        X=X,
        y=y,
        loss=MulticlassCrossEntropy(),
        optimizer=SGD(lr=0.2),
        epochs=600,
        batch_size=6,
        shuffle=True,
        seed=0,
    )

    assert history[0].loss > history[-1].loss
    assert history[-1].accuracy >= 0.95
