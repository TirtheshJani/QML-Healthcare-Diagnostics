"""Estimator-based Quantum Neural Network classifier.

The circuit is a ``ZZFeatureMap`` (input) composed with a ``RealAmplitudes`` ansatz
(weights), measured against a single-qubit ``Z`` observable on the last qubit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA

from qml_healthcare.config import RANDOM_SEED


@dataclass
class FittedQNN:
    name: str
    model: NeuralNetworkClassifier
    y_pred: np.ndarray
    y_proba: np.ndarray
    train_seconds: float
    loss_history: list[float] = field(default_factory=list)


def _build_observable(num_qubits: int) -> SparsePauliOp:
    """Z on the last qubit, identity elsewhere → ``Z ⊗ I ⊗ ... ⊗ I`` (Qiskit little-endian)."""
    label = "Z" + "I" * (num_qubits - 1)
    return SparsePauliOp.from_list([(label, 1.0)])


def train_qnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
    reps: int = 1,
    maxiter: int = 60,
    name: str = "qnn",
) -> FittedQNN:
    """Train an ``EstimatorQNN`` wrapped in ``NeuralNetworkClassifier`` (binary, ±1 output)."""
    np.random.seed(RANDOM_SEED)
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=reps, entanglement="linear")
    ansatz = RealAmplitudes(num_qubits=n_features, reps=max(1, reps))

    circuit = QuantumCircuit(n_features)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    qnn = EstimatorQNN(
        circuit=circuit,
        observables=_build_observable(n_features),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

    loss_history: list[float] = []

    def _cb(_weights: np.ndarray, value: float) -> None:
        loss_history.append(float(value))

    optimizer = COBYLA(maxiter=maxiter)
    initial_point = np.random.uniform(-np.pi, np.pi, size=qnn.num_weights)

    classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        loss="squared_error",
        optimizer=optimizer,
        callback=_cb,
        initial_point=initial_point,
    )

    X_train_a = np.asarray(X_train, dtype=float)
    X_test_a = np.asarray(X_test, dtype=float)
    # NeuralNetworkClassifier with a 1-D output expects ±1 labels for binary classification.
    y_train_pm = np.where(np.asarray(y_train).astype(int) == 1, 1, -1)

    t0 = time.perf_counter()
    classifier.fit(X_train_a, y_train_pm)
    elapsed = time.perf_counter() - t0

    raw = np.asarray(classifier.predict(X_test_a)).ravel()
    y_pred = (raw > 0).astype(int)
    # Map signed predictions in [-1, +1] to a [0, 1] pseudo-probability.
    y_proba = (np.clip(raw, -1.0, 1.0) + 1.0) / 2.0

    return FittedQNN(
        name=name,
        model=classifier,
        y_pred=y_pred,
        y_proba=np.asarray(y_proba, dtype=float),
        train_seconds=float(elapsed),
        loss_history=loss_history,
    )
