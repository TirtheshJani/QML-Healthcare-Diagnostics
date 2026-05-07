"""Variational Quantum Classifier (``ZZFeatureMap`` + ``RealAmplitudes`` + COBYLA)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

from qml_healthcare.config import RANDOM_SEED


@dataclass
class FittedVQC:
    name: str
    model: VQC
    y_pred: np.ndarray
    y_proba: np.ndarray
    train_seconds: float
    loss_history: list[float] = field(default_factory=list)


def _one_hot(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    onehot = np.zeros((len(y), 2), dtype=float)
    onehot[np.arange(len(y)), y] = 1.0
    return onehot


def train_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
    reps: int = 1,
    maxiter: int = 60,
    name: str = "vqc",
) -> FittedVQC:
    """Train a Qiskit ML ``VQC`` on the supplied quantum-ready features."""
    np.random.seed(RANDOM_SEED)
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=reps, entanglement="linear")
    ansatz = RealAmplitudes(num_qubits=n_features, reps=max(1, reps))

    loss_history: list[float] = []

    def _cb(_weights: np.ndarray, value: float) -> None:
        loss_history.append(float(value))

    optimizer = COBYLA(maxiter=maxiter)
    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=_cb,
    )

    X_train_a = np.asarray(X_train, dtype=float)
    X_test_a = np.asarray(X_test, dtype=float)
    y_train_oh = _one_hot(y_train)

    t0 = time.perf_counter()
    model.fit(X_train_a, y_train_oh)
    elapsed = time.perf_counter() - t0

    raw = np.asarray(model.predict(X_test_a))
    if raw.ndim == 2:
        y_pred = raw.argmax(axis=1).astype(int)
    else:
        y_pred = raw.astype(int)

    try:
        proba = np.asarray(model.predict_proba(X_test_a), dtype=float)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_proba = proba[:, 1]
        else:
            y_proba = proba.ravel().astype(float)
    except Exception:
        y_proba = y_pred.astype(float)

    return FittedVQC(
        name=name,
        model=model,
        y_pred=y_pred,
        y_proba=np.asarray(y_proba, dtype=float),
        train_seconds=float(elapsed),
        loss_history=loss_history,
    )
