import numpy as np

from src.activations import Softmax
from src.evaluation.classification import classification_report
from src.layers import Dense
from src.models import Sequential


def test_classification_report_returns_macro_metrics_and_confusion_matrix():
    model = Sequential([Dense(input_dim=2, output_dim=2, seed=0), Softmax()])
    X = np.eye(2)
    y = np.eye(2)

    report = classification_report(model, X, y, class_count=2)

    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.macro_f1 <= 1.0
    assert report.confusion_matrix.shape == (2, 2)
    assert report.precision.shape == (2,)
    assert report.recall.shape == (2,)
    assert report.f1.shape == (2,)
