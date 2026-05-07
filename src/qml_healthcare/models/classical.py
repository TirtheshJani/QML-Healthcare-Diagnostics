"""Classical baselines: RBF SVM, Logistic Regression, Random Forest."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from qml_healthcare.config import RANDOM_SEED


@dataclass
class FittedClassical:
    """Container for a fitted classical model + its predictions on a test set."""

    name: str
    model: object
    y_pred: np.ndarray
    y_proba: np.ndarray
    train_seconds: float


def _build_models() -> dict[str, object]:
    return {
        "svm_rbf": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "logreg": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
    }


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, FittedClassical]:
    """Fit the three classical baselines on ``X_train``, predict on ``X_test``."""
    fitted: dict[str, FittedClassical] = {}
    for name, model in _build_models().items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)
        fitted[name] = FittedClassical(
            name=name,
            model=model,
            y_pred=np.asarray(y_pred).astype(int),
            y_proba=np.asarray(y_proba, dtype=float),
            train_seconds=float(elapsed),
        )
    return fitted
