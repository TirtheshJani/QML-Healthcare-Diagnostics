"""Classical baseline models: SVM-RBF, Logistic Regression, Random Forest."""

from __future__ import annotations

import time

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

from qml_healthcare.config import RANDOM_SEED
from qml_healthcare.models._base import FittedModel


def _make_models() -> dict[str, ClassifierMixin]:
    """Construct the classical baseline estimators (one place, so CV and fit agree)."""
    return {
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
        "logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, FittedModel]:
    """Fit SVM-RBF, Logistic Regression, and Random Forest; return predictions."""
    models = _make_models()
    results: dict[str, FittedModel] = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t0
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = FittedModel(
            y_pred=y_pred,
            y_proba=y_proba,
            train_seconds=train_seconds,
            name=name,
        )
    return results


def cross_validate_baselines(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
    scoring: tuple[str, ...] = ("roc_auc", "f1", "balanced_accuracy"),
) -> dict[str, dict[str, float]]:
    """Stratified k-fold CV for the classical baselines.

    Returns ``{model: {cv_<metric>_mean, cv_<metric>_std}}``. Quantum models are too
    expensive (O(N^2) kernels) to cross-validate, so this is classical-only and the
    asymmetry is documented in the README.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out: dict[str, dict[str, float]] = {}
    for name, model in _make_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=list(scoring))
        stats: dict[str, float] = {}
        for s in scoring:
            arr = scores[f"test_{s}"]
            stats[f"cv_{s}_mean"] = float(np.mean(arr))
            stats[f"cv_{s}_std"] = float(np.std(arr))
        out[name] = stats
    return out
