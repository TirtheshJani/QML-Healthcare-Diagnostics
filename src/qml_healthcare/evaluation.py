"""Metrics + plotting helpers, all writing to ``reports/figures/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from qml_healthcare.config import FIGURES_DIR, RANDOM_SEED, RESULTS_PATH

sns.set_theme(style="whitegrid", context="talk")

# Metrics that get bootstrap confidence intervals. roc_auc / pr_auc need probabilities.
DEFAULT_CI_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "f1", "balanced_accuracy", "accuracy")
_PROBA_METRICS = frozenset({"roc_auc", "pr_auc"})


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    train_seconds: float | None = None,
) -> dict[str, float]:
    """Standard binary-classification metrics."""
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    if train_seconds is not None:
        out["train_seconds"] = float(train_seconds)
    return out


def _metric_value(
    metric: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None
) -> float:
    """Evaluate a single named metric for the bootstrap loop."""
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_proba))
    if metric == "pr_auc":
        return float(average_precision_score(y_true, y_proba))
    raise ValueError(f"Unknown metric '{metric}'.")


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    metrics: tuple[str, ...] = DEFAULT_CI_METRICS,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
    ci: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap confidence intervals for test-set metrics.

    Resamples the test predictions with replacement ``n_boot`` times (seeded, so the
    output is deterministic) and returns ``{<metric>_ci_low, <metric>_ci_high}`` for each
    metric. Probability-based metrics are skipped when ``y_proba`` is None. Bootstrap draws
    that contain only one class are discarded, since AUC/F1 are undefined there.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = None if y_proba is None else np.asarray(y_proba)
    wanted = [m for m in metrics if not (m in _PROBA_METRICS and y_proba is None)]
    n = len(y_true)
    rng = np.random.default_rng(seed)

    samples: dict[str, list[float]] = {m: [] for m in wanted}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if np.unique(yt).size < 2:
            continue
        yp = y_pred[idx]
        ypr = None if y_proba is None else y_proba[idx]
        for m in wanted:
            samples[m].append(_metric_value(m, yt, yp, ypr))

    alpha = (1.0 - ci) / 2.0
    out: dict[str, float] = {}
    for m, vals in samples.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[f"{m}_ci_low"] = float(np.quantile(arr, alpha))
        out[f"{m}_ci_high"] = float(np.quantile(arr, 1.0 - alpha))
    return out


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    train_seconds: float | None = None,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
    ci_metrics: tuple[str, ...] = DEFAULT_CI_METRICS,
) -> dict[str, float]:
    """Point metrics plus bootstrap CI bounds, in one flat dict ready for results.json."""
    out = compute_metrics(y_true, y_pred, y_proba, train_seconds=train_seconds)
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out.update(
            bootstrap_metric_ci(
                y_true, y_pred, y_proba, metrics=ci_metrics, n_boot=n_boot, seed=seed
            )
        )
    return out


def plot_roc_curves(
    results: dict[str, dict[str, np.ndarray]],
    y_true: np.ndarray,
    path: Path,
    title: str = "ROC curves",
) -> None:
    """Plot overlaid ROC curves; results maps name → {y_proba: ndarray}."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, payload in results.items():
        if "y_proba" not in payload or len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, payload["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pr_curves(
    results: dict[str, dict[str, np.ndarray]],
    y_true: np.ndarray,
    path: Path,
    title: str = "Precision-Recall curves",
) -> None:
    """Plot overlaid Precision-Recall curves; results maps name → {y_proba: ndarray}."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, payload in results.items():
        if "y_proba" not in payload or len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, payload["y_proba"])
        ap = average_precision_score(y_true, payload["y_proba"])
        ax.plot(recall, precision, lw=2, label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str = "Confusion matrix"
) -> None:
    """Save a seaborn confusion-matrix heatmap with Survived/Died axis labels."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Survived", "Died"],
        yticklabels=["Survived", "Died"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_kernel_heatmap(
    K: np.ndarray, y: np.ndarray, path: Path, title: str = "Quantum kernel matrix"
) -> None:
    """Heatmap of K, with rows/columns reordered so same-class points cluster."""
    order = np.argsort(y)
    K_sorted = K[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(K_sorted, cmap="viridis", square=True, cbar=True, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_metric_bars(
    results: dict[str, dict[str, float]],
    metric: str,
    path: Path,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Bar chart comparing one scalar metric across all models; annotates each bar."""
    names = [n for n, m in results.items() if metric in m]
    values = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names)), 5))
    palette = sns.color_palette("crest", n_colors=len(names))
    bars = ax.bar(names, values, color=palette)
    for bar, v in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel(metric)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title or f"{metric} comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_metric_bars_with_ci(
    results: dict[str, dict[str, float]],
    metric: str,
    path: Path,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Bar chart for one metric with 95% bootstrap CI error bars.

    Reads ``<metric>_ci_low`` / ``<metric>_ci_high`` per model when present; models without
    CI bounds are drawn with no error bar (zero-length).
    """
    names = [n for n, m in results.items() if metric in m]
    values = np.array([results[n][metric] for n in names], dtype=float)
    lows = np.array([results[n].get(f"{metric}_ci_low", results[n][metric]) for n in names])
    highs = np.array([results[n].get(f"{metric}_ci_high", results[n][metric]) for n in names])
    yerr = np.clip(np.vstack([values - lows, highs - values]), 0.0, None)

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names)), 5))
    palette = sns.color_palette("crest", n_colors=len(names))
    ax.bar(names, values, color=palette, yerr=yerr, capsize=4, ecolor="0.3")
    ax.set_ylabel(metric)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title or f"{metric} (95% bootstrap CI)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_loss_curve(loss: list[float], path: Path, title: str = "Training loss") -> None:
    """Line plot of per-iteration training loss for variational models (VQC, QNN)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss, lw=2, color="tab:red")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_class_balance(y: np.ndarray, path: Path, title: str = "Class balance") -> None:
    """Bar chart showing Survived vs. Died counts and percentages."""
    counts = np.bincount(y.astype(int))
    labels = ["Survived", "Died"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, counts, color=sns.color_palette("crest", 2))
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c}\n({c/len(y):.1%})", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def dump_results(results: dict[str, Any], path: Path | None = None) -> Path:
    """Write a JSON-serializable metrics dump."""
    path = path or RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with path.open("w", encoding="utf-8") as f:
        json.dump(_convert(results), f, indent=2, sort_keys=True)
    return path


def load_results(path: Path | None = None) -> dict[str, Any]:
    """Load metrics from results.json; raise FileNotFoundError if not yet generated."""
    path = path or RESULTS_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def figure_path(name: str) -> Path:
    """Return ``reports/figures/<name>``, ensuring the directory exists."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name
