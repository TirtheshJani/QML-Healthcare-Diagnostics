"""Quantum Support Vector Machine using FidelityQuantumKernel."""

from __future__ import annotations

import time

import numpy as np
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from qml_healthcare.models._base import FittedModel


def train_qsvm(
    kernel: FidelityQuantumKernel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    name: str = "qsvm",
) -> FittedModel:
    """Fit a QSVC with the given quantum kernel; return predictions and decision scores."""
    qsvc = QSVC(quantum_kernel=kernel)
    t0 = time.perf_counter()
    qsvc.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    y_pred = qsvc.predict(X_test)
    # SVC without probability=True exposes decision_function; convert to [0,1] via sigmoid
    df = qsvc.decision_function(X_test)
    y_proba = 1.0 / (1.0 + np.exp(-df))

    return FittedModel(
        y_pred=y_pred,
        y_proba=y_proba,
        train_seconds=train_seconds,
        name=name,
    )
