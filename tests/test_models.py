"""Tests for classical baselines, QSVM, VQC, and QNN — all end-to-end on tiny data."""

from __future__ import annotations

import numpy as np

from qml_healthcare.models.classical import train_baseline
from qml_healthcare.models.qnn import train_qnn
from qml_healthcare.models.qsvm import train_qsvm
from qml_healthcare.models.quantum_kernels import build_feature_map, make_quantum_kernel
from qml_healthcare.models.vqc import train_vqc


def test_classical_baselines_produce_expected_shapes(small_quantum_data):
    X_train, y_train, X_test, _ = small_quantum_data
    fitted = train_baseline(X_train, y_train, X_test)
    assert set(fitted.keys()) == {"svm_rbf", "logreg", "random_forest"}
    for f in fitted.values():
        assert f.y_pred.shape == (X_test.shape[0],)
        assert f.y_proba.shape == (X_test.shape[0],)
        assert f.train_seconds >= 0.0
        # Probabilities lie in [0, 1]
        assert (f.y_proba >= 0).all() and (f.y_proba <= 1).all()


def test_qsvm_trains_on_tiny_quantum_data(small_quantum_data):
    X_train, y_train, X_test, _ = small_quantum_data
    fm = build_feature_map("zz", n_features=X_train.shape[1], reps=1)
    kernel = make_quantum_kernel(fm)
    fitted = train_qsvm(kernel, X_train, y_train, X_test)
    assert fitted.y_pred.shape == (X_test.shape[0],)
    assert fitted.y_proba.shape == (X_test.shape[0],)
    assert set(np.unique(fitted.y_pred)).issubset({0, 1})


def test_vqc_trains_on_tiny_quantum_data(small_quantum_data):
    X_train, y_train, X_test, _ = small_quantum_data
    fitted = train_vqc(X_train, y_train, X_test, n_features=X_train.shape[1], reps=1, maxiter=3)
    assert fitted.y_pred.shape == (X_test.shape[0],)
    assert fitted.y_proba.shape == (X_test.shape[0],)
    assert set(np.unique(fitted.y_pred)).issubset({0, 1})
    assert (fitted.y_proba >= 0).all() and (fitted.y_proba <= 1).all()
    assert len(fitted.loss_history) >= 1
    assert fitted.train_seconds >= 0.0


def test_qnn_trains_on_tiny_quantum_data(small_quantum_data):
    X_train, y_train, X_test, _ = small_quantum_data
    fitted = train_qnn(X_train, y_train, X_test, n_features=X_train.shape[1], reps=1, maxiter=3)
    assert fitted.y_pred.shape == (X_test.shape[0],)
    assert fitted.y_proba.shape == (X_test.shape[0],)
    assert set(np.unique(fitted.y_pred)).issubset({0, 1})
    assert (fitted.y_proba >= 0).all() and (fitted.y_proba <= 1).all()
    assert len(fitted.loss_history) >= 1
    assert fitted.train_seconds >= 0.0
