"""Tests for the client-side demo model export.

The browser recomputes the logistic-regression sigmoid from the exported JSON. These tests
guarantee that math is faithful: the exported coef/intercept/scaler reproduce sklearn's
predict_proba to within 1e-9, and the per-feature metadata is internally consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from qml_healthcare.data.download import generate_synthetic_icu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_demo_model import (  # noqa: E402
    DEMO_FEATURES,
    _assemble_payload,
    _fit_demo,
    build_demo_payload,
)


def _payload_p(payload: dict, X_raw: np.ndarray) -> np.ndarray:
    """Reproduce the JavaScript sigmoid using only exported JSON fields."""
    mean = np.asarray(payload["scaler_mean"])
    scale = np.asarray(payload["scaler_scale"])
    coef = np.asarray(payload["coef"])
    z = (X_raw - mean) / scale
    logit = payload["intercept"] + z @ coef
    return 1.0 / (1.0 + np.exp(-logit))


def test_exported_math_matches_sklearn() -> None:
    df = generate_synthetic_icu(n=2000, seed=123)
    fit = _fit_demo(df, seed=123)
    payload = _assemble_payload(fit, seed=123)

    X_raw = fit["X_test"].to_numpy()
    js_p = _payload_p(payload, X_raw)
    sklearn_p = fit["model"].predict_proba(fit["scaler"].transform(fit["X_test"]))[:, 1]

    assert np.allclose(js_p, sklearn_p, atol=1e-9)


def test_payload_shape_and_metadata() -> None:
    df = generate_synthetic_icu(n=2000, seed=123)
    payload = build_demo_payload(df, seed=123)

    n = len(DEMO_FEATURES)
    assert payload["n_features"] == n
    assert len(payload["coef"]) == n
    assert len(payload["scaler_mean"]) == n
    assert len(payload["scaler_scale"]) == n
    assert len(payload["features"]) == n

    assert 0.0 <= payload["demo_test_roc_auc"] <= 1.0
    for feat in payload["features"]:
        assert feat["min"] <= feat["median"] <= feat["max"]
        assert feat["default"] == feat["median"]
        assert feat["label"] and feat["unit"]


def test_payload_is_deterministic() -> None:
    df = generate_synthetic_icu(n=1500, seed=123)
    a = build_demo_payload(df, seed=123)
    b = build_demo_payload(df, seed=123)
    assert a["coef"] == b["coef"]
    assert a["intercept"] == b["intercept"]
