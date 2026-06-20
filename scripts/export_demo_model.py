"""Export a small, interpretable logistic-regression model for the in-browser demo.

The showcase site is static (GitHub Pages), so there is no Python backend at request time.
A logistic regression is a linear model, so we can export its coefficients, intercept, and
the StandardScaler statistics to JSON and recompute the sigmoid in JavaScript. The visitor
enters a handful of ICU values and gets an instant risk estimate, fully client-side.

This is a deliberately SIMPLIFIED model: six interpretable raw inputs, retrained here so the
coefficients map one-to-one to labeled sliders. It is not the full 29-feature benchmark model
and is not for clinical use. Its own held-out ROC-AUC is exported so the site can state it
honestly next to the prediction.

The JavaScript in docs/assets/js/demo.js implements exactly this math:
    z_i   = (input_i - scaler_mean_i) / scaler_scale_i
    logit = intercept + sum_i coef_i * z_i
    p     = 1 / (1 + exp(-logit))
tests/test_demo_export.py asserts that formula reproduces sklearn's predict_proba.
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from qml_healthcare.config import PROJECT_ROOT, RANDOM_SEED
from qml_healthcare.data.download import ensure_dataset
from qml_healthcare.data.loader import load_raw
from qml_healthcare.data.preprocess import clean, make_splits, select_features
from qml_healthcare.evaluation import compute_metrics

DEMO_MODEL_PATH = PROJECT_ROOT / "docs" / "assets" / "data" / "demo_model.json"

# Six interpretable inputs a non-specialist can reason about. Order here is the order of the
# exported coef / scaler arrays, so the JavaScript can zip them positionally.
DEMO_FEATURES: list[dict[str, object]] = [
    {
        "key": "apache_4a_hospital_death_prob",
        "label": "APACHE-IVa baseline risk",
        "unit": "probability (an existing clinical score, not a quantum result)",
        "step": 0.01,
    },
    {"key": "age", "label": "Age", "unit": "years", "step": 1},
    {"key": "heart_rate_apache", "label": "Heart rate", "unit": "bpm", "step": 1},
    {"key": "map_apache", "label": "Mean arterial pressure", "unit": "mmHg", "step": 1},
    {"key": "creatinine_apache", "label": "Creatinine", "unit": "mg/dL", "step": 0.1},
    {"key": "gcs_motor_apache", "label": "Glasgow Coma motor score", "unit": "1 to 6", "step": 1},
]
DEMO_NOTE = (
    "Simplified demonstration model: logistic regression on six interpretable inputs, "
    "retrained for this widget. It is not the full benchmark model and is not for clinical use."
)


def _fit_demo(df: pd.DataFrame, seed: int = RANDOM_SEED) -> dict:
    """Fit the compact demo model and gather everything needed to assemble the payload."""
    keys = [f["key"] for f in DEMO_FEATURES]
    X_all, y_all = select_features(df)
    X_all, y_all = clean(X_all, y_all)
    missing = [k for k in keys if k not in X_all.columns]
    if missing:
        raise KeyError(f"Demo features missing from cleaned data: {missing}")
    X = X_all[keys].astype(float)

    X_train, _X_val, X_test, y_train, _y_val, y_test = make_splits(X, y_all, seed=seed)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(
        scaler.transform(X_train), y_train
    )

    y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    y_pred = model.predict(scaler.transform(X_test))
    metrics = compute_metrics(y_test.to_numpy(), y_pred, y_proba)

    stats = {
        k: {
            "min": round(float(X[k].min()), 2),
            "max": round(float(X[k].max()), 2),
            "median": round(float(X[k].median()), 2),
        }
        for k in keys
    }
    return {
        "keys": keys,
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "demo_test_roc_auc": float(metrics.get("roc_auc", float("nan"))),
        "stats": stats,
    }


def _assemble_payload(fit: dict, seed: int = RANDOM_SEED) -> dict:
    """Turn a fitted demo model into the JSON payload the browser consumes."""
    model: LogisticRegression = fit["model"]
    scaler: StandardScaler = fit["scaler"]
    stats: dict = fit["stats"]

    features = []
    for spec in DEMO_FEATURES:
        s = stats[spec["key"]]
        features.append(
            {
                **spec,
                "min": s["min"],
                "max": s["max"],
                "median": s["median"],
                "default": s["median"],
            }
        )
    return {
        "model": "logistic_regression",
        "note": DEMO_NOTE,
        "n_features": len(features),
        "positive_class_label": "In-hospital death",
        "demo_test_roc_auc": round(fit["demo_test_roc_auc"], 4),
        "intercept": float(model.intercept_[0]),
        "coef": [float(c) for c in model.coef_[0]],
        "scaler_mean": [float(m) for m in scaler.mean_],
        "scaler_scale": [float(s) for s in scaler.scale_],
        "features": features,
        "seed": seed,
    }


def build_demo_payload(df: pd.DataFrame | None = None, seed: int = RANDOM_SEED) -> dict:
    """Fit the compact demo model and return the JSON-ready payload.

    When ``df`` is None the WiDS-schema dataset is loaded (real if Kaggle credentials are
    present, else the synthetic fallback), matching the rest of the pipeline.
    """
    if df is None:
        df = load_raw(ensure_dataset())
    fit = _fit_demo(df, seed=seed)
    return _assemble_payload(fit, seed=seed)


def main() -> None:
    payload = build_demo_payload()
    DEMO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_MODEL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {DEMO_MODEL_PATH} "
        f"({payload['n_features']} features, demo ROC-AUC {payload['demo_test_roc_auc']:.3f})."
    )


if __name__ == "__main__":
    main()
