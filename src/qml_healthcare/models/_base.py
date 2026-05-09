"""Shared result container for all trained models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FittedModel:
    y_pred: np.ndarray
    y_proba: np.ndarray
    train_seconds: float
    name: str = ""
    loss_history: list[float] = field(default_factory=list)
