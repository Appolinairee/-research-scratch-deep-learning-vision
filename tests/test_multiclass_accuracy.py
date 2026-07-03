import numpy as np

from src.metrics import multiclass_accuracy


def test_multiclass_accuracy_uses_argmax():
    y_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    y_pred = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.4, 0.3, 0.3],
        ]
    )

    accuracy = multiclass_accuracy(y_true, y_pred)

    assert np.isclose(accuracy, 2.0 / 3.0)
