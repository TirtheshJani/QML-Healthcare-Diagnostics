"""Variational Quantum Classifier trainer."""

from __future__ import annotations

import time

import numpy as np
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from scipy.optimize import minimize

from qml_healthcare.models._base import FittedModel


def train_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
    reps: int = 2,
    maxiter: int = 60,
) -> FittedModel:
    """Train a VQC (ZZFeatureMap + RealAmplitudes) via COBYLA; return predictions."""
    loss_history: list[float] = []

    def callback(weights: np.ndarray, loss: float) -> None:
        loss_history.append(float(loss))

    def cobyla_optimizer(fun, x0, jac=None, bounds=None):  # noqa: ARG001
        return minimize(fun, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})

    vqc = VQC(
        feature_map=ZZFeatureMap(feature_dimension=n_features, reps=reps),
        ansatz=RealAmplitudes(n_features, reps=reps),
        optimizer=cobyla_optimizer,
        callback=callback,
        sampler=StatevectorSampler(),
    )

    t0 = time.perf_counter()
    vqc.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    y_pred = vqc.predict(X_test)
    y_proba = vqc.predict_proba(X_test)[:, 1]

    return FittedModel(
        y_pred=y_pred,
        y_proba=y_proba,
        train_seconds=train_seconds,
        loss_history=loss_history,
    )
