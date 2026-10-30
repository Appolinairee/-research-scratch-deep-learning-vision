from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    accuracy: float
    validation_loss: float | None = None
    validation_accuracy: float | None = None
    weight_norm: float | None = None
    gradient_norm: float | None = None
    relu_zero_fraction: float | None = None


class TrainingHistory(list[EpochMetrics]):
    @property
    def best_epoch(self) -> int:
        candidates = [
            row for row in self
            if row.validation_loss is not None
        ]
        if not candidates:
            candidates = list(self)
        if not candidates:
            raise ValueError("history cannot be empty")
        return min(candidates, key=lambda row: row.validation_loss
                   if row.validation_loss is not None else row.loss).epoch
