import numpy as np

from src.activations import ReLU, Softmax
from src.evaluation.classification import evaluate_multiclass
from src.layers import Dense
from src.losses import MulticlassCrossEntropy
from src.models import Sequential


def test_evaluate_multiclass_returns_loss_and_accuracy():
    model = Sequential(
        [Dense(input_dim=2, output_dim=2, seed=0), Softmax()]
    )
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([[1.0, 0.0], [0.0, 1.0]])

    metrics = evaluate_multiclass(model, X, y, MulticlassCrossEntropy())

    assert metrics.loss > 0.0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_history_best_epoch_uses_validation_loss():
    from src.training.history import TrainingHistory, EpochMetrics

    history = TrainingHistory(
        [
            EpochMetrics(0, 1.0, 0.5, 0.9, 0.6),
            EpochMetrics(1, 0.8, 0.6, 0.8, 0.5),
        ]
    )

    assert history.best_epoch == 1


def test_training_loss_includes_regularization_penalty():
    from src.optim import SGD
    from src.regularization import L2Regularization
    from src.training.sequential import train_multiclass_classifier

    model = Sequential([Dense(input_dim=2, output_dim=2, seed=0), Softmax()])
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([[1.0, 0.0], [0.0, 1.0]])

    history = train_multiclass_classifier(
        model=model,
        X=X,
        y=y,
        loss=MulticlassCrossEntropy(),
        optimizer=SGD(lr=0.0),
        epochs=1,
        batch_size=2,
        regularizer=L2Regularization(strength=1.0),
    )

    data_loss = MulticlassCrossEntropy().forward(y, model.predict_proba(X))
    assert history[0].loss > data_loss
