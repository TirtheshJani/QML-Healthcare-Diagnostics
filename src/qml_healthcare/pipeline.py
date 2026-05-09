"""End-to-end pipeline: data → baseline → quantum kernels → QSVM → bonus → reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qml_healthcare.config import (
    DEFAULT_QUANTUM_SUBSAMPLE,
    DEFAULT_QUBITS,
    DEFAULT_REPS,
    RANDOM_SEED,
    ensure_dirs,
)
from qml_healthcare.data.download import ensure_dataset
from qml_healthcare.data.preprocess import DataBundle, prepare_data
from qml_healthcare.evaluation import (
    compute_metrics,
    dump_results,
    figure_path,
    load_results,
    plot_class_balance,
    plot_confusion,
    plot_kernel_heatmap,
    plot_loss_curve,
    plot_metric_bars,
    plot_pr_curves,
    plot_roc_curves,
)
from qml_healthcare.models.classical import train_baseline
from qml_healthcare.models.qnn import train_qnn
from qml_healthcare.models.qsvm import train_qsvm
from qml_healthcare.models.quantum_kernels import (
    FEATURE_MAP_NAMES,
    build_feature_map,
    compute_kernel_matrix,
    make_quantum_kernel,
)
from qml_healthcare.models.vqc import train_vqc


def _merge_into_results(updates: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Read existing results.json (if any), merge updates, write back."""
    try:
        existing = load_results()
    except FileNotFoundError:
        existing = {}
    existing.update(updates)
    dump_results(existing)
    return existing


def run_data(force_synthetic: bool = False) -> Path:
    """Stage 1 — make sure ``data/raw/training_v2.csv`` exists."""
    ensure_dirs()
    path = ensure_dataset(force_synthetic=force_synthetic)
    return path


def _ensure_bundle(quantum_n: int, quantum_k: int) -> DataBundle:
    run_data()
    return prepare_data(quantum_n=quantum_n, quantum_k=quantum_k)


def run_baseline(
    quantum_n: int = DEFAULT_QUANTUM_SUBSAMPLE, quantum_k: int = DEFAULT_QUBITS
) -> dict:
    """Stage 2 — classical baselines on the full preprocessed train/test."""
    bundle = _ensure_bundle(quantum_n, quantum_k)
    plot_class_balance(bundle.y_train, figure_path("class_balance.png"), "Train-set class balance")
    print(f"Training classical baselines on N_train={len(bundle.X_train)}...")
    fitted = train_baseline(bundle.X_train, bundle.y_train, bundle.X_test)

    metrics = {
        name: compute_metrics(bundle.y_test, f.y_pred, f.y_proba, train_seconds=f.train_seconds)
        for name, f in fitted.items()
    }
    proba_dict = {name: {"y_proba": f.y_proba} for name, f in fitted.items()}
    plot_roc_curves(
        proba_dict, bundle.y_test, figure_path("classical_roc.png"), "Classical baselines — ROC"
    )
    plot_pr_curves(
        proba_dict, bundle.y_test, figure_path("classical_pr.png"), "Classical baselines — PR"
    )
    for name, f in fitted.items():
        plot_confusion(
            bundle.y_test,
            f.y_pred,
            figure_path(f"classical_confusion_{name}.png"),
            f"{name} — Confusion matrix",
        )
    _merge_into_results({"classical": metrics})
    return metrics


def run_qsvm(
    quantum_n: int = DEFAULT_QUANTUM_SUBSAMPLE,
    quantum_k: int = DEFAULT_QUBITS,
    reps: int = DEFAULT_REPS,
    feature_maps: tuple[str, ...] = FEATURE_MAP_NAMES,
) -> dict:
    """Stage 3 — QSVM with each feature map on the subsampled top-k splits."""
    bundle = _ensure_bundle(quantum_n, quantum_k)
    n_features = bundle.n_quantum_features
    print(
        f"QSVM stage: N_train_q={len(bundle.X_train_q)}, N_test_q={len(bundle.X_test_q)}, "
        f"K={n_features}, reps={reps}"
    )

    qsvm_metrics: dict[str, dict[str, float]] = {}
    proba_dict: dict[str, dict[str, np.ndarray]] = {}

    for fm_name in feature_maps:
        print(f"\n--- Feature map: {fm_name} ---")
        fm = build_feature_map(fm_name, n_features=n_features, reps=reps)
        kernel = make_quantum_kernel(fm)

        # Visualize kernel structure on a small subset (first 60 points)
        n_viz = min(60, len(bundle.X_train_q))
        K_viz = compute_kernel_matrix(kernel, bundle.X_train_q[:n_viz])
        plot_kernel_heatmap(
            K_viz,
            bundle.y_train_q[:n_viz],
            figure_path(f"kernel_heatmap_{fm_name}.png"),
            f"Quantum kernel ({fm_name}) — first {n_viz} train points",
        )

        fitted = train_qsvm(
            kernel,
            bundle.X_train_q,
            bundle.y_train_q,
            bundle.X_test_q,
            name=f"qsvm_{fm_name}",
        )
        m = compute_metrics(
            bundle.y_test_q, fitted.y_pred, fitted.y_proba, train_seconds=fitted.train_seconds
        )
        qsvm_metrics[fitted.name] = m
        proba_dict[fitted.name] = {"y_proba": fitted.y_proba}
        plot_confusion(
            bundle.y_test_q,
            fitted.y_pred,
            figure_path(f"confusion_qsvm_{fm_name}.png"),
            f"QSVM ({fm_name}) — Confusion matrix",
        )
        print(f"  metrics: {m}")

    plot_roc_curves(
        proba_dict, bundle.y_test_q, figure_path("qsvm_roc_overlay.png"), "QSVM ROC by feature map"
    )
    _merge_into_results({"qsvm": qsvm_metrics})
    return qsvm_metrics


def run_bonus(
    quantum_n: int = DEFAULT_QUANTUM_SUBSAMPLE,
    quantum_k: int = DEFAULT_QUBITS,
    reps: int = DEFAULT_REPS,
    maxiter: int = 60,
) -> dict:
    """Stage 4 — VQC + QNN bonus models."""
    bundle = _ensure_bundle(quantum_n, quantum_k)
    n_features = bundle.n_quantum_features

    print(f"VQC: N_train={len(bundle.X_train_q)}, K={n_features}, maxiter={maxiter}")
    vqc = train_vqc(
        bundle.X_train_q,
        bundle.y_train_q,
        bundle.X_test_q,
        n_features=n_features,
        reps=reps,
        maxiter=maxiter,
    )
    plot_loss_curve(vqc.loss_history, figure_path("vqc_loss.png"), "VQC training loss")
    plot_confusion(
        bundle.y_test_q, vqc.y_pred, figure_path("confusion_vqc.png"), "VQC — Confusion matrix"
    )
    vqc_m = compute_metrics(
        bundle.y_test_q, vqc.y_pred, vqc.y_proba, train_seconds=vqc.train_seconds
    )

    print(f"QNN: N_train={len(bundle.X_train_q)}, K={n_features}, maxiter={maxiter}")
    qnn = train_qnn(
        bundle.X_train_q,
        bundle.y_train_q,
        bundle.X_test_q,
        n_features=n_features,
        reps=reps,
        maxiter=maxiter,
    )
    plot_loss_curve(qnn.loss_history, figure_path("qnn_loss.png"), "QNN training loss")
    plot_confusion(
        bundle.y_test_q, qnn.y_pred, figure_path("confusion_qnn.png"), "QNN — Confusion matrix"
    )
    qnn_m = compute_metrics(
        bundle.y_test_q, qnn.y_pred, qnn.y_proba, train_seconds=qnn.train_seconds
    )

    _merge_into_results({"bonus": {"vqc": vqc_m, "qnn": qnn_m}})
    return {"vqc": vqc_m, "qnn": qnn_m}


def run_reports() -> dict:
    """Stage 5 — final consolidated comparison plots from results.json."""
    results = load_results()
    flat: dict[str, dict[str, float]] = {}
    flat.update(results.get("classical", {}))
    flat.update(results.get("qsvm", {}))
    bonus = results.get("bonus", {})
    flat.update(bonus)

    if not flat:
        raise RuntimeError("results.json is empty — run baseline and qsvm first.")

    plot_metric_bars(
        flat, "roc_auc", figure_path("final_comparison.png"), "ROC-AUC across models", ylim=(0, 1)
    )
    plot_metric_bars(flat, "f1", figure_path("f1_comparison.png"), "F1 across models", ylim=(0, 1))
    plot_metric_bars(
        flat,
        "balanced_accuracy",
        figure_path("balanced_accuracy_comparison.png"),
        "Balanced accuracy across models",
        ylim=(0, 1),
    )
    plot_metric_bars(
        flat, "train_seconds", figure_path("runtime_comparison.png"), "Training time (s)"
    )
    return flat


def run_all(
    quantum_n: int = DEFAULT_QUANTUM_SUBSAMPLE,
    quantum_k: int = DEFAULT_QUBITS,
    reps: int = DEFAULT_REPS,
    bonus_maxiter: int = 60,
    seed: int = RANDOM_SEED,  # noqa: ARG001
) -> dict:
    """Run all five pipeline stages in sequence and return the final metrics dict."""
    run_data()
    run_baseline(quantum_n=quantum_n, quantum_k=quantum_k)
    run_qsvm(quantum_n=quantum_n, quantum_k=quantum_k, reps=reps)
    run_bonus(quantum_n=quantum_n, quantum_k=quantum_k, reps=reps, maxiter=bonus_maxiter)
    return run_reports()
