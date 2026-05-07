"""Quantum feature maps + ``FidelityQuantumKernel`` factory."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

FEATURE_MAP_NAMES: tuple[str, ...] = ("zz", "pauli", "custom")


def _custom_ring_feature_map(n_features: int, reps: int = 1) -> QuantumCircuit:
    """Hadamard + RY(x) + RZ(x) per qubit, ring-entangled with pairwise RZ(x_i * x_{i+1})."""
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    x = ParameterVector("x", length=n_features)
    qc = QuantumCircuit(n_features, name="custom_ring")
    for _ in range(max(1, reps)):
        for i in range(n_features):
            qc.h(i)
            qc.ry(x[i], i)
            qc.rz(x[i], i)
        if n_features > 1:
            for i in range(n_features):
                j = (i + 1) % n_features
                qc.cx(i, j)
                qc.rz(x[i] * x[j], j)
                qc.cx(i, j)
    return qc


def build_feature_map(name: str, n_features: int, reps: int = 1) -> QuantumCircuit:
    """Construct one of ``FEATURE_MAP_NAMES`` with ``n_features`` qubits."""
    name = name.lower()
    if name == "zz":
        return ZZFeatureMap(feature_dimension=n_features, reps=reps, entanglement="linear")
    if name == "pauli":
        return PauliFeatureMap(
            feature_dimension=n_features,
            reps=reps,
            paulis=["Z", "ZZ", "ZZZ"] if n_features >= 3 else ["Z", "ZZ"],
            entanglement="linear",
        )
    if name == "custom":
        return _custom_ring_feature_map(n_features, reps=reps)
    raise ValueError(f"Unknown feature map '{name}'. Expected one of {FEATURE_MAP_NAMES}.")


def make_quantum_kernel(feature_map: QuantumCircuit) -> FidelityQuantumKernel:
    """Wrap a feature map in a ``FidelityQuantumKernel`` (ComputeUncompute fidelity)."""
    return FidelityQuantumKernel(feature_map=feature_map)


def compute_kernel_matrix(
    kernel: FidelityQuantumKernel,
    X: np.ndarray,
    Y: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate K(X, X) when ``Y`` is None, else K(X, Y)."""
    X = np.asarray(X, dtype=float)
    if Y is None:
        K = kernel.evaluate(x_vec=X)
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
        return K
    Y = np.asarray(Y, dtype=float)
    return kernel.evaluate(x_vec=X, y_vec=Y)
