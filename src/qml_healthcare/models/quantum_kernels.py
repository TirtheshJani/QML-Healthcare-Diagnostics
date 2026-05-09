"""Quantum feature maps and the FidelityQuantumKernel wrapper."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

FEATURE_MAP_NAMES: tuple[str, ...] = ("zz", "pauli", "custom")


def build_feature_map(name: str, n_features: int, reps: int = 2) -> QuantumCircuit:
    """Return a Qiskit QuantumCircuit encoding ``n_features`` classical inputs."""
    name = name.lower()
    if name == "zz":
        return ZZFeatureMap(feature_dimension=n_features, reps=reps)
    if name == "pauli":
        return PauliFeatureMap(feature_dimension=n_features, reps=reps, paulis=["Z", "ZZ"])
    if name == "custom":
        return _custom_feature_map(n_features=n_features, reps=reps)
    raise ValueError(f"Unknown feature map '{name}'. Choose from {FEATURE_MAP_NAMES}.")


def _custom_feature_map(n_features: int, reps: int) -> QuantumCircuit:
    """H + RZ(2x) encoding with CZ entanglement — explicit alternative to ZZFeatureMap."""
    params = ParameterVector("x", n_features)
    qc = QuantumCircuit(n_features)
    for _ in range(reps):
        qc.h(range(n_features))
        for i, p in enumerate(params):
            qc.rz(2.0 * p, i)
        for i in range(n_features - 1):
            qc.cz(i, i + 1)
    return qc


def make_quantum_kernel(feature_map: QuantumCircuit) -> FidelityQuantumKernel:
    """Wrap a feature map in a FidelityQuantumKernel (statevector-based)."""
    return FidelityQuantumKernel(feature_map=feature_map)


def compute_kernel_matrix(
    kernel: FidelityQuantumKernel,
    X_a: np.ndarray,
    X_b: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the kernel matrix K[i,j] = ⟨φ(xᵢ)|φ(xⱼ)⟩²."""
    if X_b is None:
        return kernel.evaluate(X_a)
    return kernel.evaluate(X_a, X_b)
