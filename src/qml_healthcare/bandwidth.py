"""Helpers for the kernel bandwidth ablation (``scripts/ablate_kernel_bandwidth.py``).

The ablation multiplies the quantum inputs by a scale factor ``s`` before encoding (the kernel
"bandwidth") and fits the same SVC the pipeline's QSVC fits, on exact fidelity kernels.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from qml_healthcare.models.quantum_kernels import exact_fidelity_kernel


def offdiag_mean(K: np.ndarray) -> float:
    """Mean of the off-diagonal entries of a square kernel matrix."""
    return float(K[~np.eye(len(K), dtype=bool)].mean())


def svc_on_kernel(
    K_train: np.ndarray, y_train: np.ndarray, K_eval: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``SVC(kernel="precomputed")`` with QSVC's defaults (C=1.0) on ``K_train``.

    Returns predictions and scores for the rows of ``K_eval`` (shape n_eval x n_train). The
    scores go through the same sigmoid of the decision function as ``models.qsvm.train_qsvm``.
    """
    clf = SVC(kernel="precomputed").fit(K_train, y_train)
    y_pred = clf.predict(K_eval)
    y_proba = 1.0 / (1.0 + np.exp(-clf.decision_function(K_eval)))
    return y_pred, y_proba


def select_scale(val_roc_auc: dict[float, float]) -> float:
    """Pick the scale with the highest validation ROC-AUC; ties go to the larger scale."""
    return max(val_roc_auc, key=lambda s: (val_roc_auc[s], s))


def sweep_scales(
    feature_map: QuantumCircuit,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scales: tuple[float, ...],
) -> list[dict[str, float]]:
    """For each scale, the training-kernel off-diagonal mean and the validation ROC-AUC.

    Only training and validation rows go in, so the test rows cannot influence selection.
    """
    rows = []
    for s in scales:
        K_train = exact_fidelity_kernel(feature_map, s * X_train)
        K_val = exact_fidelity_kernel(feature_map, s * X_val, s * X_train)
        _, val_scores = svc_on_kernel(K_train, y_train, K_val)
        rows.append(
            {
                "scale": s,
                "offdiag_mean": offdiag_mean(K_train),
                "val_roc_auc": float(roc_auc_score(y_val, val_scores)),
            }
        )
    return rows
