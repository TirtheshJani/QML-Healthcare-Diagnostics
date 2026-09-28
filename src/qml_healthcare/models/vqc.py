"""Variational Quantum Classifier trainer."""

from __future__ import annotations

import time

import numpy as np
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from scipy.optimize import minimize

from qml_healthcare.config import RANDOM_SEED
from qml_healthcare.models._base import FittedModel


def train_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
    reps: int = 2,
    maxiter: int = 60,
    seed: int = RANDOM_SEED,
) -> FittedModel:
    """Train a VQC (ZZFeatureMap + RealAmplitudes) via COBYLA; return predictions.

    ``seed`` fixes the initial weights and the sampler's shot sampling, so reruns match.
    """
    loss_history: list[float] = []

    def cobyla_optimizer(fun, x0, jac=None, bounds=None):  # noqa: ARG001
        # qiskit-machine-learning only invokes `callback` for its own SciPyOptimizer classes,
        # not for a plain callable like this one, so record each loss evaluation here.
        def logged_fun(weights: np.ndarray) -> float:
            loss = fun(weights)
            loss_history.append(float(loss))
            return loss

        options = {"maxiter": maxiter, "rhobeg": 0.5}
        return minimize(logged_fun, x0, method="COBYLA", options=options)

    ansatz = RealAmplitudes(n_features, reps=reps)
    vqc = VQC(
        feature_map=ZZFeatureMap(feature_dimension=n_features, reps=reps),
        ansatz=ansatz,
        optimizer=cobyla_optimizer,
        sampler=StatevectorSampler(seed=seed),
        initial_point=np.random.default_rng(seed).random(ansatz.num_parameters),
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
