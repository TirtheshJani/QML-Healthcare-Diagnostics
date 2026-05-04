"""Shared test fixtures."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from qml_healthcare.data.download import generate_synthetic_icu

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """A 300-row synthetic ICU dataframe with WiDS-compatible columns."""
    return generate_synthetic_icu(n=300, seed=123)


@pytest.fixture
def small_quantum_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tiny (X_train, y_train, X_test, y_test) for quantum-model tests."""
    rng = np.random.default_rng(7)
    n_train, n_test, k = 16, 8, 3
    X_train = rng.normal(size=(n_train, k))
    y_train = rng.integers(0, 2, size=n_train)
    if len(set(y_train.tolist())) < 2:
        y_train[0] = 0
        y_train[1] = 1
    X_test = rng.normal(size=(n_test, k))
    y_test = rng.integers(0, 2, size=n_test)
    return X_train, y_train, X_test, y_test
