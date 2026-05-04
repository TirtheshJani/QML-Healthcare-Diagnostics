"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import numpy as np

from qml_healthcare.config import TARGET
from qml_healthcare.data.preprocess import (
    clean,
    make_splits,
    scale,
    select_features,
    subsample_for_quantum,
    top_k_features,
)


def test_select_features_drops_target_and_ids(synthetic_df):
    X, y = select_features(synthetic_df)
    assert TARGET not in X.columns
    assert "encounter_id" not in X.columns
    assert "patient_id" not in X.columns
    assert len(X) == len(y)


def test_clean_eliminates_missing_and_one_hots(synthetic_df):
    X, y = select_features(synthetic_df)
    X_clean, y_clean = clean(X, y)
    assert X_clean.isna().sum().sum() == 0, "no NaNs should remain"
    # All columns end up numeric after one-hot encoding
    assert all(np.issubdtype(dt, np.number) for dt in X_clean.dtypes), X_clean.dtypes
    assert len(X_clean) == len(y_clean)


def test_make_splits_are_stratified_and_disjoint(synthetic_df):
    X, y = clean(*select_features(synthetic_df))
    X_tr, X_v, X_te, y_tr, y_v, y_te = make_splits(X, y, seed=1)
    n = len(X)
    assert len(X_tr) + len(X_v) + len(X_te) == n
    # Stratification: positive rate within each split is close to overall
    overall = y.mean()
    for s in (y_tr, y_v, y_te):
        assert abs(s.mean() - overall) < 0.10, (s.mean(), overall)


def test_scale_fits_on_train_only(synthetic_df):
    X, y = clean(*select_features(synthetic_df))
    X_tr, X_v, X_te, _, _, _ = make_splits(X, y, seed=1)
    X_tr_s, X_v_s, X_te_s, _ = scale(X_tr, X_v, X_te)
    # Train means should be ~0; val/test will not be (no leakage)
    assert np.allclose(X_tr_s.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_tr_s.std(axis=0), 1, atol=1e-1)


def test_top_k_features_returns_correct_shape(synthetic_df):
    X, y = clean(*select_features(synthetic_df))
    X_tr, _, _, y_tr, _, _ = make_splits(X, y, seed=1)
    X_tr_s, *_ = scale(X_tr, X_tr, X_tr)  # reuse for simplicity
    X_topk, names, sel = top_k_features(X_tr_s, y_tr.to_numpy(), list(X_tr.columns), k=5)
    assert X_topk.shape[1] == 5
    assert len(names) == 5
    assert sel.k == 5


def test_subsample_for_quantum_is_balanced():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 4))
    # 90% / 10% imbalance
    y = (rng.uniform(size=1000) < 0.1).astype(int)
    X_q, y_q = subsample_for_quantum(X, y, n=100, balanced=True, seed=0)
    counts = np.bincount(y_q)
    # Should be close to 50/50
    assert min(counts) / max(counts) > 0.5, counts
    assert len(X_q) == len(y_q)
