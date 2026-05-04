"""Tests for evaluation metrics + plot helpers."""

from __future__ import annotations

import numpy as np

from qml_healthcare.evaluation import (
    compute_metrics,
    dump_results,
    load_results,
    plot_class_balance,
    plot_confusion,
    plot_kernel_heatmap,
    plot_loss_curve,
    plot_metric_bars,
    plot_pr_curves,
    plot_roc_curves,
)


def test_compute_metrics_keys():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_proba = np.array([0.2, 0.8, 0.6, 0.7, 0.4, 0.1, 0.9, 0.3])
    m = compute_metrics(y_true, y_pred, y_proba, train_seconds=1.5)
    expected = {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "train_seconds",
    }
    assert expected.issubset(m.keys())
    for k in expected:
        assert isinstance(m[k], float)


def test_compute_metrics_without_proba_omits_auc():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    m = compute_metrics(y_true, y_pred, y_proba=None)
    assert "roc_auc" not in m
    assert "accuracy" in m


def test_plot_helpers_write_files(tmp_path):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 50)
    y_pred = rng.integers(0, 2, 50)
    y_proba = rng.uniform(0, 1, 50)

    paths = {
        "roc": tmp_path / "roc.png",
        "pr": tmp_path / "pr.png",
        "cm": tmp_path / "cm.png",
        "kh": tmp_path / "kh.png",
        "bars": tmp_path / "bars.png",
        "loss": tmp_path / "loss.png",
        "balance": tmp_path / "balance.png",
    }
    plot_roc_curves({"a": {"y_proba": y_proba}}, y_true, paths["roc"])
    plot_pr_curves({"a": {"y_proba": y_proba}}, y_true, paths["pr"])
    plot_confusion(y_true, y_pred, paths["cm"])
    K = np.eye(8) + rng.normal(0, 0.1, (8, 8))
    K = (K + K.T) / 2
    plot_kernel_heatmap(K, rng.integers(0, 2, 8), paths["kh"])
    plot_metric_bars({"m1": {"v": 0.5}, "m2": {"v": 0.7}}, "v", paths["bars"])
    plot_loss_curve([1.0, 0.5, 0.3, 0.2], paths["loss"])
    plot_class_balance(y_true, paths["balance"])
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 1000


def test_dump_and_load_results_roundtrip(tmp_path):
    path = tmp_path / "results.json"
    data = {
        "classical": {"svm_rbf": {"accuracy": 0.85, "f1": np.float64(0.8)}},
        "qsvm": {"qsvm_zz": {"roc_auc": np.float32(0.75)}},
    }
    dump_results(data, path=path)
    loaded = load_results(path=path)
    assert loaded["classical"]["svm_rbf"]["accuracy"] == 0.85
    assert isinstance(loaded["qsvm"]["qsvm_zz"]["roc_auc"], float)
