"""Tests for quantum feature maps and the FidelityQuantumKernel."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qml_healthcare.models.quantum_kernels import (
    FEATURE_MAP_NAMES,
    build_feature_map,
    compute_kernel_matrix,
    make_quantum_kernel,
)


@pytest.mark.parametrize("name", FEATURE_MAP_NAMES)
def test_build_feature_map_qubit_count(name):
    qc = build_feature_map(name, n_features=4, reps=1)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 4


def test_build_feature_map_unknown_raises():
    with pytest.raises(ValueError):
        build_feature_map("nonsense", n_features=2)


@pytest.mark.parametrize("name", FEATURE_MAP_NAMES)
def test_kernel_matrix_is_symmetric_and_psd(name):
    qc = build_feature_map(name, n_features=3, reps=1)
    kernel = make_quantum_kernel(qc)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 3))
    K = compute_kernel_matrix(kernel, X)
    assert K.shape == (6, 6)
    assert np.allclose(K, K.T, atol=1e-6), "kernel should be symmetric"
    assert np.allclose(np.diag(K), 1.0, atol=1e-6), "self-fidelity should be 1"
    eig_min = np.linalg.eigvalsh(K).min()
    assert eig_min > -1e-6, f"kernel must be PSD, got eig_min={eig_min}"


def test_kernel_matrix_cross_shape():
    qc = build_feature_map("zz", n_features=2, reps=1)
    kernel = make_quantum_kernel(qc)
    rng = np.random.default_rng(0)
    X_a = rng.normal(size=(4, 2))
    X_b = rng.normal(size=(3, 2))
    K = compute_kernel_matrix(kernel, X_a, X_b)
    assert K.shape == (4, 3)
    assert (K >= -1e-6).all() and (K <= 1.0 + 1e-6).all()
