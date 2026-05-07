"""QSVM training using ``qiskit_machine_learning.algorithms.QSVC``."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import BaseKernel


@dataclass
class FittedQSVM:
    name: str
    model: QSVC
    y_pred: np.ndarray
    y_proba: np.ndarray
    train_seconds: float


def _decision_to_proba(scores: np.ndarray) -> np.ndarray:
    """Map raw SVM decision scores to a [0, 1] pseudo-probability via min-max."""
    scores = np.asarray(scores, dtype=float)
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - lo) / (hi - lo)


def train_qsvm(
    kernel: BaseKernel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    name: str = "qsvm",
    C: float = 1.0,
) -> FittedQSVM:
    """Fit a ``QSVC`` with the supplied quantum kernel and return predictions on ``X_test``."""
    model = QSVC(quantum_kernel=kernel, C=C, probability=True)
    t0 = time.perf_counter()
    model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train).astype(int))
    elapsed = time.perf_counter() - t0

    X_test_a = np.asarray(X_test, dtype=float)
    y_pred = model.predict(X_test_a).astype(int)
    try:
        y_proba = model.predict_proba(X_test_a)[:, 1]
    except Exception:
        y_proba = _decision_to_proba(model.decision_function(X_test_a))
    return FittedQSVM(
        name=name,
        model=model,
        y_pred=y_pred,
        y_proba=np.asarray(y_proba, dtype=float),
        train_seconds=float(elapsed),
    )
