"""Preprocessing for ICU mortality prediction.

Pipeline:
    select_features → clean → make_splits → scale → top_k_features → subsample_for_quantum

`prepare_data()` is the high-level entry point used by notebooks and scripts.
Outputs are cached to ``data/processed/`` as parquet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qml_healthcare.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    RANDOM_SEED,
    TARGET,
    ensure_dirs,
)
from qml_healthcare.data.loader import load_raw


@dataclass
class DataBundle:
    """All splits + metadata produced by ``prepare_data``."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    # Quantum-ready (subsampled, top-k) splits
    X_train_q: np.ndarray
    X_test_q: np.ndarray
    y_train_q: np.ndarray
    y_test_q: np.ndarray
    quantum_feature_names: list[str]
    quantum_selector: SelectKBest

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_quantum_features(self) -> int:
        return self.X_train_q.shape[1]


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Pick the curated feature subset and split off the target."""
    cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in dataframe.")
    X = df[cols].copy()
    y = df[TARGET].copy()
    return X, y


def clean(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drop missing-target rows; impute numeric (median) and categorical (mode); one-hot encode."""
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int).copy()

    numeric_cols = [c for c in X.columns if c in NUMERIC_FEATURES]
    categorical_cols = [c for c in X.columns if c in CATEGORICAL_FEATURES]

    for c in numeric_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in categorical_cols:
        if X[c].dtype.kind not in "fi":
            X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "Unknown")
        else:
            X[c] = X[c].fillna(X[c].median())

    # In pandas 3.x string columns may use StringDtype rather than object.
    string_categoricals = [c for c in categorical_cols if not pd.api.types.is_numeric_dtype(X[c])]
    if string_categoricals:
        X = pd.get_dummies(X, columns=string_categoricals, drop_first=True)
    # Cast dummy bool columns to numeric
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X, y


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train/val/test split."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=rel_val, random_state=seed, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaler on train only; transform all three splits."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def top_k_features(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str], k: int
) -> tuple[np.ndarray, list[str], SelectKBest]:
    """Select the k features with highest f_classif score against the target."""
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_sel = selector.fit_transform(X_train, y_train)
    mask = selector.get_support()
    selected = [n for n, keep in zip(feature_names, mask, strict=False) if keep]
    return X_sel, selected, selector


def subsample_for_quantum(
    X: np.ndarray, y: np.ndarray, n: int, seed: int = RANDOM_SEED, balanced: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample. For QSVM the kernel is O(N^2) so we have to keep N small."""
    if n >= len(X):
        return X, y
    rng = np.random.default_rng(seed)
    if balanced:
        # Equal-size sampling per class so the quantum kernel sees both classes
        per_class = n // 2
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        take_pos = rng.choice(idx_pos, size=min(per_class, len(idx_pos)), replace=False)
        take_neg = rng.choice(idx_neg, size=min(n - len(take_pos), len(idx_neg)), replace=False)
        idx = np.concatenate([take_pos, take_neg])
        rng.shuffle(idx)
    else:
        idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def prepare_data(
    csv_path: Path | None = None,
    quantum_n: int = 800,
    quantum_k: int = 6,
    seed: int = RANDOM_SEED,
    cache: bool = True,
) -> DataBundle:
    """End-to-end preprocessing. Returns a DataBundle with all splits."""
    ensure_dirs()
    df = load_raw(csv_path)
    X, y = select_features(df)
    X, y = clean(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y, seed=seed)
    feature_names = list(X_train.columns)
    X_train_s, X_val_s, X_test_s, scaler = scale(X_train, X_val, X_test)

    # Quantum-ready subset
    X_train_topk, selected, selector = top_k_features(
        X_train_s, y_train.to_numpy(), feature_names, k=quantum_k
    )
    X_test_topk = selector.transform(X_test_s)
    X_train_q, y_train_q = subsample_for_quantum(
        X_train_topk, y_train.to_numpy(), n=quantum_n, seed=seed
    )
    # Use a smaller test set for quantum too (kernel cost is N_train * N_test)
    X_test_q, y_test_q = subsample_for_quantum(
        X_test_topk, y_test.to_numpy(), n=max(200, quantum_n // 4), seed=seed + 1
    )

    bundle = DataBundle(
        X_train=X_train_s,
        X_val=X_val_s,
        X_test=X_test_s,
        y_train=y_train.to_numpy(),
        y_val=y_val.to_numpy(),
        y_test=y_test.to_numpy(),
        feature_names=feature_names,
        scaler=scaler,
        X_train_q=X_train_q,
        X_test_q=X_test_q,
        y_train_q=y_train_q,
        y_test_q=y_test_q,
        quantum_feature_names=selected,
        quantum_selector=selector,
    )

    if cache:
        _cache_bundle(bundle)
    return bundle


def _cache_bundle(b: DataBundle) -> None:
    """Persist preprocessed arrays to parquet (for cheap re-loading by notebooks)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(b.X_train, columns=b.feature_names).to_parquet(PROCESSED_DIR / "X_train.parquet")
    pd.DataFrame(b.X_val, columns=b.feature_names).to_parquet(PROCESSED_DIR / "X_val.parquet")
    pd.DataFrame(b.X_test, columns=b.feature_names).to_parquet(PROCESSED_DIR / "X_test.parquet")
    pd.Series(b.y_train, name="y").to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet")
    pd.Series(b.y_val, name="y").to_frame().to_parquet(PROCESSED_DIR / "y_val.parquet")
    pd.Series(b.y_test, name="y").to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet")
    pd.DataFrame(b.X_train_q, columns=b.quantum_feature_names).to_parquet(
        PROCESSED_DIR / "X_train_q.parquet"
    )
    pd.DataFrame(b.X_test_q, columns=b.quantum_feature_names).to_parquet(
        PROCESSED_DIR / "X_test_q.parquet"
    )
    pd.Series(b.y_train_q, name="y").to_frame().to_parquet(PROCESSED_DIR / "y_train_q.parquet")
    pd.Series(b.y_test_q, name="y").to_frame().to_parquet(PROCESSED_DIR / "y_test_q.parquet")
