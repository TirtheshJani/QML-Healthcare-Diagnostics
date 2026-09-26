"""Quantum Neural Network classifier (SamplerQNN + NeuralNetworkClassifier)."""

from __future__ import annotations

import time

import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from scipy.optimize import minimize

from qml_healthcare.config import RANDOM_SEED
from qml_healthcare.models._base import FittedModel
from qml_healthcare.models.quantum_kernels import build_feature_map


def train_qnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
    reps: int = 2,
    maxiter: int = 60,
    seed: int = RANDOM_SEED,
) -> FittedModel:
    """Train a SamplerQNN classifier (PauliFeatureMap + RealAmplitudes); return predictions.

    ``seed`` fixes the initial weights and the sampler's shot sampling, so reruns match.
    """
    loss_history: list[float] = []

    def callback(weights: np.ndarray, loss: float) -> None:
        loss_history.append(float(loss))

    def cobyla_optimizer(fun, x0, jac=None, bounds=None):  # noqa: ARG001
        return minimize(fun, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})

    feature_map = build_feature_map("pauli", n_features=n_features, reps=1)
    ansatz = RealAmplitudes(n_features, reps=reps)

    qc = feature_map.compose(ansatz)

    qnn = SamplerQNN(
        circuit=qc,
        input_params=list(feature_map.parameters),
        weight_params=list(ansatz.parameters),
        interpret=lambda x: int(x) % 2,
        output_shape=2,
        sampler=StatevectorSampler(seed=seed),
    )

    classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        loss="cross_entropy",
        one_hot=True,
        optimizer=cobyla_optimizer,
        callback=callback,
        initial_point=np.random.default_rng(seed).random(qnn.num_weights),
    )

    t0 = time.perf_counter()
    classifier.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    return FittedModel(
        y_pred=y_pred,
        y_proba=y_proba,
        train_seconds=train_seconds,
        loss_history=loss_history,
    )
