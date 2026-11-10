import csv
from pathlib import Path

import numpy as np

from src.data.nsl_kdd import load_nsl_kdd, read_nsl_kdd


def _row(label: str, index: int) -> list[str]:
    numeric_before = [str(float(index))]
    numeric_after = [str(float(index + column)) for column in range(1, 38)]
    return numeric_before + ["tcp", "http", "SF"] + numeric_after + [label, "0"]


def _write_dataset(path: Path, labels: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(_row(label, index) for index, label in enumerate(labels))


def _write_balanced_fixture(tmp_path: Path) -> tuple[Path, Path]:
    labels = ["normal.", "neptune.", "ipsweep.", "guess_passwd.", "rootkit."]
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    _write_dataset(train, labels * 4)
    _write_dataset(test, labels)
    return train, test


def test_read_nsl_kdd_maps_attack_labels_to_five_classes(tmp_path):
    path = tmp_path / "train.csv"
    _write_dataset(
        path, ["normal.", "neptune.", "ipsweep.", "guess_passwd.", "rootkit."]
    )

    _, labels = read_nsl_kdd(path)

    np.testing.assert_array_equal(labels, [0, 1, 2, 3, 4])


def test_load_nsl_kdd_returns_disjoint_preprocessed_splits(tmp_path):
    train_path, test_path = _write_balanced_fixture(tmp_path)

    data = load_nsl_kdd(
        train_path, test_path, validation_fraction=0.25, seed=42
    )

    assert len(data.X_train) + len(data.X_validation) == 20
    assert data.X_train.shape[1] == data.X_validation.shape[1] == data.X_test.shape[1]
    assert data.y_train.shape[1] == data.y_validation.shape[1] == data.y_test.shape[1] == 5
    assert set(data.y_train_idx.tolist()).issubset(range(5))
    assert set(data.y_validation_idx.tolist()).issubset(range(5))
