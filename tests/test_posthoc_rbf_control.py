"""Tests for the post-hoc tuned RBF control (scripts/posthoc_tuned_rbf_control.py).

The control was added after the bandwidth ablation's results were committed, so it is post-hoc.
These tests check that it tunes the RBF bandwidth the way the ablation tunes the QSVM input scale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from qml_healthcare.bandwidth import select_scale
from qml_healthcare.config import REPORTS_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from posthoc_tuned_rbf_control import (  # noqa: E402
    KEY,
    gamma_scale,
    offdiag_corr,
    rbf_scores,
    sweep_rbf_scales,
)


def test_gamma_scale_matches_sklearn_scale(small_quantum_data):
    X_train, y_train, X_test, _ = small_quantum_data
    ours = SVC(kernel="rbf", gamma=gamma_scale(X_train)).fit(X_train, y_train)
    sk = SVC(kernel="rbf", gamma="scale").fit(X_train, y_train)
    np.testing.assert_allclose(ours.decision_function(X_test), sk.decision_function(X_test))


def test_rbf_scores_at_scale_s_equal_rbf_on_scaled_inputs(small_quantum_data):
    """gamma = gamma_scale * s**2 on X is the RBF kernel on s * X with gamma fixed: the same
    intervention as the QSVM ablation, which multiplies the inputs by s."""
    X_train, y_train, X_test, _ = small_quantum_data
    g, s = gamma_scale(X_train), 0.2
    _, ours = rbf_scores(X_train, y_train, X_test, g * s**2)
    clf = SVC(kernel="rbf", gamma=g).fit(s * X_train, y_train)
    expected = 1.0 / (1.0 + np.exp(-clf.decision_function(s * X_test)))
    np.testing.assert_allclose(ours, expected, atol=1e-8)


def test_sweep_rbf_scales_reports_every_scale(small_quantum_data):
    X_train, y_train, X_val, _ = small_quantum_data
    y_val = np.array([0, 1] * (len(X_val) // 2))
    g = gamma_scale(X_train)
    rows = sweep_rbf_scales(X_train, y_train, X_val, y_val, (0.1, 1.0), g)
    assert [r["scale"] for r in rows] == [0.1, 1.0]
    for r in rows:
        assert np.isclose(r["gamma"], g * r["scale"] ** 2)
        assert 0.0 <= r["val_roc_auc"] <= 1.0


def test_offdiag_corr_uses_only_off_diagonal_entries():
    A = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])
    B = np.array([[9.0, 0.3, 0.5], [0.3, -9.0, 0.7], [0.5, 0.7, 5.0]])  # diagonal ignored
    assert np.isclose(offdiag_corr(A, B), 1.0)
    assert np.isclose(offdiag_corr(A, -B), -1.0)


def test_committed_posthoc_control_follows_the_ablation_rule():
    """The committed post-hoc block uses the ablation's grid, bootstrap and selection rule, and
    leaves the ablation's own keys alone."""
    ablation = json.loads((REPORTS_DIR / "bandwidth_ablation.json").read_text(encoding="utf-8"))
    block = ablation[KEY]
    assert block["post_hoc"] is True
    assert [r["scale"] for r in block["sweep"]] == ablation["prespecified"]["scales"]
    val = {r["scale"]: r["val_roc_auc"] for r in block["sweep"]}
    assert block["selected_scale"] == select_scale(val)
    assert block["test"]["scale"] == block["selected_scale"]
    assert block["n_boot"] == ablation["prespecified"]["n_boot"]
    assert block["seed"] == ablation["prespecified"]["seed"]
    assert block["test"]["roc_auc_ci_low"] <= block["test"]["roc_auc"]
    assert block["test"]["roc_auc"] <= block["test"]["roc_auc_ci_high"]
    assert set(block["kernel_offdiag_corr_with_tuned_rbf"]) == set(ablation["feature_maps"])
