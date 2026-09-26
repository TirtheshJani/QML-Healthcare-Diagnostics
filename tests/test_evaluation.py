"""Tests for evaluation metrics + plot helpers."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from qml_healthcare.evaluation import (
    bootstrap_metric_ci,
    compute_metrics,
    compute_metrics_with_ci,
    dump_results,
    load_results,
    plot_class_balance,
    plot_confusion,
    plot_kernel_heatmap,
    plot_loss_curve,
    plot_metric_bars,
    plot_metric_bars_with_ci,
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


def test_bootstrap_metric_ci_is_deterministic_and_bracketed():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 80)
    y_proba = np.clip(y_true * 0.4 + rng.uniform(0, 0.6, 80), 0, 1)
    y_pred = (y_proba > 0.5).astype(int)

    ci_a = bootstrap_metric_ci(y_true, y_pred, y_proba, n_boot=200, seed=42)
    ci_b = bootstrap_metric_ci(y_true, y_pred, y_proba, n_boot=200, seed=42)
    assert ci_a == ci_b, "same seed must give identical CIs"

    point = compute_metrics(y_true, y_pred, y_proba)
    for metric in ("roc_auc", "f1", "balanced_accuracy", "accuracy"):
        lo, hi = ci_a[f"{metric}_ci_low"], ci_a[f"{metric}_ci_high"]
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= point[metric] <= hi


def test_bootstrap_metric_ci_skips_auc_without_proba():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    ci = bootstrap_metric_ci(y_true, y_pred, y_proba=None, n_boot=100, seed=0)
    assert "roc_auc_ci_low" not in ci
    assert "f1_ci_low" in ci


def test_compute_metrics_with_ci_includes_ci_keys():
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, 40)
    y_proba = rng.uniform(0, 1, 40)
    y_pred = (y_proba > 0.5).astype(int)
    m = compute_metrics_with_ci(y_true, y_pred, y_proba, train_seconds=2.0, n_boot=100)
    assert "roc_auc" in m and "roc_auc_ci_low" in m and "roc_auc_ci_high" in m
    assert m["train_seconds"] == 2.0


def test_plot_metric_bars_with_ci_writes_file(tmp_path):
    results = {
        "a": {"roc_auc": 0.7, "roc_auc_ci_low": 0.6, "roc_auc_ci_high": 0.8},
        "b": {"roc_auc": 0.5},  # no CI bounds -> zero-length error bar
    }
    path = tmp_path / "ci_bars.png"
    plot_metric_bars_with_ci(results, "roc_auc", path, ylim=(0, 1))
    assert path.exists() and path.stat().st_size > 1000


@pytest.mark.parametrize("plot", [plot_metric_bars_with_ci, plot_metric_bars])
def test_metric_bar_tick_labels_do_not_overlap(tmp_path, monkeypatch, plot):
    """The README's key figures used to print 'random_forest' on top of 'svm_rbf'."""
    boxes = []
    original_savefig = Figure.savefig

    def spy(self, *args, **kwargs):
        out = original_savefig(self, *args, **kwargs)  # draws the figure
        renderer = self.canvas.get_renderer()
        boxes.extend(t.get_window_extent(renderer) for t in self.axes[0].get_xticklabels())
        return out

    monkeypatch.setattr(Figure, "savefig", spy)
    names = [
        "logreg",
        "random_forest",
        "svm_rbf",
        "qsvm_custom",
        "qsvm_pauli",
        "qsvm_zz",
        "qnn",
        "vqc",
    ]
    results = {n: {"roc_auc": 0.6, "roc_auc_ci_low": 0.5, "roc_auc_ci_high": 0.7} for n in names}
    plot(results, "roc_auc", tmp_path / "bars.png", ylim=(0, 1))

    assert len(boxes) == len(names)
    for left, right in zip(boxes, boxes[1:], strict=False):
        assert not left.overlaps(right)


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
