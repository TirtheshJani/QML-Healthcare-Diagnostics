"""Tests for the kernel bandwidth ablation helpers (scripts/ablate_kernel_bandwidth.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from qml_healthcare.bandwidth import offdiag_mean, select_scale, svc_on_kernel, sweep_scales
from qml_healthcare.config import REPORTS_DIR
from qml_healthcare.evaluation import load_results
from qml_healthcare.models.qsvm import train_qsvm
from qml_healthcare.models.quantum_kernels import (
    build_feature_map,
    exact_fidelity_kernel,
    make_quantum_kernel,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ablate_kernel_bandwidth import DESCRIPTION  # noqa: E402


def test_svc_on_exact_kernel_matches_pipeline_qsvc(small_quantum_data):
    """At s = 1 the ablation must reproduce the pipeline's QSVC, predictions and scores."""
    X_train, y_train, X_test, _ = small_quantum_data
    fm = build_feature_map("zz", n_features=X_train.shape[1], reps=2)
    qsvc = train_qsvm(make_quantum_kernel(fm), X_train, y_train, X_test)
    y_pred, y_proba = svc_on_kernel(
        exact_fidelity_kernel(fm, X_train), y_train, exact_fidelity_kernel(fm, X_test, X_train)
    )
    np.testing.assert_array_equal(y_pred, qsvc.y_pred)
    # The two kernels agree to ~1e-15 (tests/test_quantum_kernels.py checks them at 1e-8), but
    # libsvm stops at its default tolerance (tol=1e-3), so rounding-level kernel differences can
    # end the solver at slightly different dual solutions. That moves the scores by ~1e-5, and
    # which side of 1e-6 it lands on depends on the platform. 1e-4 still fails on a real mismatch:
    # changing C by 10 percent moves these scores by ~1.5e-2.
    np.testing.assert_allclose(y_proba, qsvc.y_proba, atol=1e-4)


def test_offdiag_mean_ignores_the_diagonal():
    K = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])
    assert np.isclose(offdiag_mean(K), 0.4)


def test_select_scale_picks_highest_validation_auc():
    assert select_scale({0.05: 0.61, 0.1: 0.80, 0.5: 0.70, 1.0: 0.52}) == 0.1


def test_select_scale_breaks_ties_toward_the_larger_scale():
    assert select_scale({0.1: 0.70, 0.5: 0.70, 1.0: 0.52}) == 0.5


def test_sweep_scales_reports_every_scale(small_quantum_data):
    X_train, y_train, X_val, _ = small_quantum_data
    y_val = np.array([0, 1] * (len(X_val) // 2))
    fm = build_feature_map("custom", n_features=X_train.shape[1], reps=1)
    rows = sweep_scales(fm, X_train, y_train, X_val, y_val, scales=(1e-6, 1.0))
    assert [r["scale"] for r in rows] == [1e-6, 1.0]
    for r in rows:
        assert 0.0 <= r["val_roc_auc"] <= 1.0
        assert 0.0 <= r["offdiag_mean"] <= 1.0 + 1e-9
    # Shrinking the inputs towards 0 maps every point to the same state, so K -> all ones.
    assert rows[0]["offdiag_mean"] > 0.999


def test_committed_ablation_agrees_with_committed_results():
    """reports/bandwidth_ablation.json must reproduce the committed QSVM rows at s = 1.0, and
    each selected scale must follow the selection rule of the design fixed before the first run,
    applied to the recorded validation AUCs."""
    ablation = json.loads((REPORTS_DIR / "bandwidth_ablation.json").read_text(encoding="utf-8"))
    committed = load_results()["qsvm"]
    for name, m in ablation["feature_maps"].items():
        for key in ("roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "accuracy"):
            assert np.isclose(m["test"]["s_1"][key], committed[f"qsvm_{name}"][key], atol=1e-9)
        val = {r["scale"]: r["val_roc_auc"] for r in m["sweep"]}
        assert m["selected_scale"] == select_scale(val)
        assert m["test"]["selected"]["scale"] == m["selected_scale"]


def test_committed_ablation_description_is_the_one_the_script_writes():
    """The description in reports/bandwidth_ablation.json must equal the string that
    scripts/ablate_kernel_bandwidth.py writes, so an edit to one without the other fails here."""
    ablation = json.loads((REPORTS_DIR / "bandwidth_ablation.json").read_text(encoding="utf-8"))
    assert ablation["description"] == DESCRIPTION
