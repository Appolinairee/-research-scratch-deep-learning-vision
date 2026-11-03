import pytest

from src.optim import SGD
from src.optim.scheduler import ReduceLROnPlateau


def test_reduce_lr_on_plateau_waits_for_patience():
    optimizer = SGD(lr=0.1)
    scheduler = ReduceLROnPlateau(factor=0.5, patience=2, min_lr=0.01)

    scheduler.step(1.0, optimizer)
    scheduler.step(1.1, optimizer)
    assert optimizer.lr == 0.1

    scheduler.step(1.2, optimizer)

    assert optimizer.lr == 0.05


def test_reduce_lr_on_plateau_respects_minimum_learning_rate():
    optimizer = SGD(lr=0.02)
    scheduler = ReduceLROnPlateau(factor=0.5, patience=1, min_lr=0.01)

    scheduler.step(1.0, optimizer)
    scheduler.step(1.1, optimizer)
    scheduler.step(1.2, optimizer)

    assert optimizer.lr == 0.01


def test_reduce_lr_on_plateau_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        ReduceLROnPlateau(factor=1.0, patience=2, min_lr=0.0)
