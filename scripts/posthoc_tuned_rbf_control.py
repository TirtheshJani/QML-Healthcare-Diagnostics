"""POST-HOC control for the kernel bandwidth ablation: an RBF SVM tuned the same way.

This analysis is post-hoc. It was designed after scripts/ablate_kernel_bandwidth.py had been run
and its results committed, in response to a review that pointed out an asymmetry: each QSVM
had its input scale chosen on the validation split, while the ablation's classical controls use
the pipeline's default hyperparameters. It does not change scripts/ablate_kernel_bandwidth.py or
any key that script writes; it adds one key, "posthoc_tuned_rbf", to
reports/bandwidth_ablation.json.

- Rows: the same 200-row, 6-feature quantum training split, 500-row validation split and
  200-row quantum test split as the ablation (prepare_data with the pipeline defaults, seed 42).
  The script stops if they do not match the ablation's recorded "data" block.
- Model: SVC(kernel="rbf", C=1.0). Score: a sigmoid of the decision function, as for the QSVMs.
- Bandwidth grid: gamma = gamma_scale * s**2 for s in the ablation's grid (read from the
  ablation's "prespecified" block), where gamma_scale = 1 / (n_features * X_train.var()) is
  scikit-learn's gamma="scale" on the unscaled training rows. This is the RBF kernel on s * X
  with gamma fixed at gamma_scale, i.e. the ablation's intervention (multiply the inputs by s).
  s = 1 is the default gamma.
- Selection: qml_healthcare.bandwidth.select_scale on validation ROC-AUC (highest wins, ties go
  to the larger s). Test rows are not used for selection.
- Test evaluation: once, at the selected s, with compute_metrics_with_ci and the ablation's
  bootstrap settings (1000 draws, seed 42).
- Kernel similarity: for each feature map, the Pearson correlation between the off-diagonal
  entries of its training kernel at the ablation's selected s and those of the selected RBF
  training kernel.

Usage (after scripts/ablate_kernel_bandwidth.py, which rewrites the JSON without this key):
    python scripts/posthoc_tuned_rbf_control.py
"""

from __future__ import annotations

import json

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

from qml_healthcare.bandwidth import select_scale
from qml_healthcare.config import DEFAULT_QUANTUM_SUBSAMPLE, DEFAULT_QUBITS, REPORTS_DIR
from qml_healthcare.data.download import ensure_dataset
from qml_healthcare.data.preprocess import prepare_data
from qml_healthcare.evaluation import compute_metrics_with_ci
from qml_healthcare.models.quantum_kernels import build_feature_map, exact_fidelity_kernel

KEY = "posthoc_tuned_rbf"
ABLATION_JSON = REPORTS_DIR / "bandwidth_ablation.json"


def gamma_scale(X: np.ndarray) -> float:
    """scikit-learn's gamma="scale" for the rows X: 1 / (n_features * X.var())."""
    return 1.0 / (X.shape[1] * X.var())


def rbf_scores(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit SVC(kernel="rbf", C=1.0) and return predictions and sigmoid(decision) scores."""
    clf = SVC(kernel="rbf", gamma=gamma, C=1.0).fit(X_train, y_train)
    return clf.predict(X_eval), 1.0 / (1.0 + np.exp(-clf.decision_function(X_eval)))


def sweep_rbf_scales(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scales: tuple[float, ...],
    gamma_ref: float,
) -> list[dict[str, float]]:
    """Validation ROC-AUC of the RBF SVC at gamma = gamma_ref * s**2 for each scale s."""
    rows = []
    for s in scales:
        gamma = gamma_ref * s**2
        _, val_scores = rbf_scores(X_train, y_train, X_val, gamma)
        rows.append(
            {"scale": s, "gamma": gamma, "val_roc_auc": float(roc_auc_score(y_val, val_scores))}
        )
    return rows


def offdiag_corr(A: np.ndarray, B: np.ndarray) -> float:
    """Pearson correlation between the upper-triangle off-diagonal entries of two kernels."""
    iu = np.triu_indices(len(A), k=1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def run(ablation: dict) -> dict:
    spec = ablation["prespecified"]
    scales = tuple(spec["scales"])
    ensure_dataset()
    b = prepare_data(
        quantum_n=DEFAULT_QUANTUM_SUBSAMPLE,
        quantum_k=DEFAULT_QUBITS,
        seed=spec["seed"],
        cache=False,
    )
    X_val_q = b.quantum_selector.transform(b.X_val)
    rec = ablation["data"]
    same_rows = (
        b.quantum_feature_names == rec["quantum_features"]
        and len(b.y_train_q) == rec["n_train_q"]
        and len(b.y_val) == rec["n_val"]
        and len(b.y_test_q) == rec["n_test_q"]
        and np.isclose(b.X_train_q.min(), rec["quantum_train_input_min"])
        and np.isclose(b.X_train_q.max(), rec["quantum_train_input_max"])
    )
    if not same_rows:
        raise SystemExit("The data split does not match the ablation's; stopping without writing.")

    g_ref = gamma_scale(b.X_train_q)
    sweep = sweep_rbf_scales(b.X_train_q, b.y_train_q, X_val_q, b.y_val, scales, g_ref)
    selected = select_scale({r["scale"]: r["val_roc_auc"] for r in sweep})
    gamma_sel = g_ref * selected**2
    y_pred, y_proba = rbf_scores(b.X_train_q, b.y_train_q, b.X_test_q, gamma_sel)
    test = {
        "scale": selected,
        "gamma": gamma_sel,
        **compute_metrics_with_ci(
            b.y_test_q, y_pred, y_proba, n_boot=spec["n_boot"], seed=spec["seed"]
        ),
    }
    for r in sweep:
        print(f"  s={r['scale']:<5} gamma={r['gamma']:.6f} val_roc_auc={r['val_roc_auc']:.4f}")
    print(
        f"selected s={selected}; test ROC-AUC {test['roc_auc']:.4f} "
        f"[{test['roc_auc_ci_low']:.4f}, {test['roc_auc_ci_high']:.4f}]"
    )

    K_rbf = rbf_kernel(b.X_train_q, gamma=gamma_sel)
    corr = {}
    for name, m in ablation["feature_maps"].items():
        fm = build_feature_map(name, n_features=b.n_quantum_features, reps=rec["reps"])
        s = m["selected_scale"]
        corr[name] = {
            "fidelity_kernel_scale": s,
            "corr": offdiag_corr(exact_fidelity_kernel(fm, s * b.X_train_q), K_rbf),
        }
        print(
            f"  {name} kernel at s={s} vs tuned RBF kernel: off-diagonal corr {corr[name]['corr']:.4f}"
        )

    return {
        "post_hoc": True,
        "description": (
            "POST-HOC control, designed after the ablation results above were committed: an RBF "
            "SVC (C=1) on the same split with gamma = gamma_scale * s**2 over the ablation's s "
            "grid, s selected with the same rule on the same validation rows, test scored once. "
            "See the docstring of scripts/posthoc_tuned_rbf_control.py."
        ),
        "gamma_scale": g_ref,
        "sweep": sweep,
        "selected_scale": selected,
        "test": test,
        "n_boot": spec["n_boot"],
        "seed": spec["seed"],
        "kernel_offdiag_corr_with_tuned_rbf": corr,
        "scikit_learn": sklearn.__version__,
    }


def main() -> None:
    ablation = json.loads(ABLATION_JSON.read_text(encoding="utf-8"))
    ablation[KEY] = run(ablation)
    ABLATION_JSON.write_text(json.dumps(ablation, indent=2) + "\n", encoding="utf-8")
    print(f"Added '{KEY}' to {ABLATION_JSON}.")


if __name__ == "__main__":
    main()
