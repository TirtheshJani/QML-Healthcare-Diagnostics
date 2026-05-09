"""Classical baseline models: SVM-RBF, Logistic Regression, Random Forest."""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from qml_healthcare.config import RANDOM_SEED
from qml_healthcare.models._base import FittedModel


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, FittedModel]:
    """Fit SVM-RBF, Logistic Regression, and Random Forest; return predictions."""
    models: dict[str, object] = {
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
        "logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }
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
