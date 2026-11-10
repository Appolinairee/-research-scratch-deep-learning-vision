from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NSLKDD_CLASSES = ("normal", "dos", "probe", "r2l", "u2r")
_CATEGORICAL_COLUMNS = (1, 2, 3)
_ATTACK_GROUPS = {
    "normal": "normal",
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "mailbomb": "dos",
    "apache2": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    "worm": "dos",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "satan": "probe",
    "mscan": "probe",
    "saint": "probe",
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "httptunnel": "r2l",
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
    "ps": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
}
_CLASS_TO_INDEX = {name: index for index, name in enumerate(NSLKDD_CLASSES)}


@dataclass(frozen=True)
class NSLKDDData:
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    y_train_idx: np.ndarray
    y_validation_idx: np.ndarray
    y_test_idx: np.ndarray
    feature_names: tuple[str, ...]


def read_nsl_kdd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[str]] = []
    labels: list[int] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 42:
                raise ValueError(f"NSL-KDD row has fewer than 42 columns: {path}")
            raw_label = row[41].rstrip(".")
            try:
                class_name = _ATTACK_GROUPS[raw_label]
            except KeyError as exc:
                raise ValueError(f"Unknown NSL-KDD label: {raw_label}") from exc
            features.append(row[:41])
            labels.append(_CLASS_TO_INDEX[class_name])
    return np.asarray(features, dtype=object), np.asarray(labels, dtype=int)


def load_nsl_kdd(
    train_path: Path,
    test_path: Path,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> NSLKDDData:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    raw_train, labels = read_nsl_kdd(train_path)
    raw_test, test_labels = read_nsl_kdd(test_path)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(raw_train))
    validation_size = int(len(raw_train) * validation_fraction)
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]

    category_maps = {
        column: {
            value: index
            for index, value in enumerate(np.unique(raw_train[train_indices, column]))
        }
        for column in _CATEGORICAL_COLUMNS
    }

    def encode(raw_features: np.ndarray) -> np.ndarray:
        columns: list[np.ndarray] = []
        for column in range(raw_features.shape[1]):
            values = raw_features[:, column]
            if column not in _CATEGORICAL_COLUMNS:
                columns.append(values.astype(np.float32)[:, None])
                continue
            mapping = category_maps[column]
            encoded = np.zeros(
                (len(values), len(mapping) + 1), dtype=np.float32
            )
            for row, value in enumerate(values):
                encoded[row, mapping.get(value, len(mapping))] = 1.0
            columns.append(encoded)
        return np.concatenate(columns, axis=1)

    encoded_train = encode(raw_train)
    encoded_test = encode(raw_test)
    X_train = encoded_train[train_indices]
    X_validation = encoded_train[validation_indices]
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8

    def normalize(features: np.ndarray) -> np.ndarray:
        return ((features - mean) / std).astype(np.float32)

    train_labels = labels[train_indices]
    validation_labels = labels[validation_indices]
    return NSLKDDData(
        X_train=normalize(X_train),
        X_validation=normalize(X_validation),
        X_test=normalize(encoded_test),
        y_train=np.eye(len(NSLKDD_CLASSES), dtype=np.float32)[train_labels],
        y_validation=np.eye(len(NSLKDD_CLASSES), dtype=np.float32)[validation_labels],
        y_test=np.eye(len(NSLKDD_CLASSES), dtype=np.float32)[test_labels],
        y_train_idx=train_labels,
        y_validation_idx=validation_labels,
        y_test_idx=test_labels,
        feature_names=tuple(
            f"feature_{index}" for index in range(encoded_train.shape[1])
        ),
    )
