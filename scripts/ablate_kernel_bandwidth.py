"""Kernel bandwidth ablation for the three QSVM feature maps.

Added after an audit found that the committed QSVM kernels sit at the random-state fidelity:
the quantum inputs are StandardScaler z-scores (magnitudes up to about 3) used directly as
rotation angles, and the off-diagonal kernel mean is close to 1/2^6, the mean fidelity between
two random 6-qubit states. This script tests whether that encoding scale, rather than the
sample size or the feature count, explains the chance-level QSVM results. It does not change
the pipeline or reports/results.json.

Everything below was fixed before this script's first run. This is not a blind test: the audit
that found the concentration had already scored the test rows at s in {1, 0.5, 0.25, 0.1}
(custom 0.798 at s = 0.1) before this grid and selection rule were committed, so this is a
confirmatory rerun with a selection rule fixed before the committed run.

- Intervention: multiply the quantum inputs (train and evaluation rows alike) by a scale
  factor s before encoding. SCALES = (0.05, 0.1, 0.2, 0.5, 1.0); s = 1.0 is the pipeline.
- Model: SVC(kernel="precomputed", C=1.0) on the exact fidelity kernel from statevector
  overlaps (qml_healthcare.models.quantum_kernels.exact_fidelity_kernel). This is the model the
  pipeline's QSVC fits, with the same feature maps (reps=2) and the same score (a sigmoid of the
  decision function). C is not tuned; s is the only thing selected.
- Training rows: the pipeline's 200-row class-balanced quantum training subsample, top-6
  features (prepare_data with the pipeline defaults, seed 42).
- Selection rule: for each feature map, the s with the highest validation ROC-AUC. Validation
  is the pipeline's full validation split (otherwise unused by the pipeline) restricted to the
  same 6 features. Ties go to the larger s. Test rows are never used for selection.
- Test evaluation: only at s = 1.0 and at the selected s, on the pipeline's 200-row
  class-balanced test subsample, with compute_metrics_with_ci (1000 seeded bootstrap draws).
- Check: at s = 1.0 the test ROC-AUC and accuracy must equal the committed QSVM rows in
  reports/results.json. The script stops without writing anything if they do not.
- Matched-data controls on the same 200-row, 6-feature split, with the same bootstrap: the
  pipeline's classical baselines (train_baseline: logistic regression, SVM-RBF, random forest)
  and the APACHE hospital-death-probability column alone as a score (no model).
- Reference: the APACHE column alone on the full test split, next to the committed logistic
  regression row.
- Seed: RANDOM_SEED = 42 throughout.
- Outputs: reports/bandwidth_ablation.json and reports/figures/bandwidth_ablation.png. Wall
  times are recorded for the exact-kernel computations.

Usage:
    python scripts/ablate_kernel_bandwidth.py
"""

from __future__ import annotations

import json
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit_machine_learning
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from qml_healthcare.bandwidth import select_scale, svc_on_kernel, sweep_scales
from qml_healthcare.config import (
    DEFAULT_QUANTUM_SUBSAMPLE,
    DEFAULT_QUBITS,
    DEFAULT_REPS,
    FIGURES_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)
from qml_healthcare.data.download import ensure_dataset
from qml_healthcare.data.preprocess import prepare_data
from qml_healthcare.evaluation import bootstrap_metric_ci, compute_metrics_with_ci, load_results
from qml_healthcare.models.classical import train_baseline
from qml_healthcare.models.quantum_kernels import (
    FEATURE_MAP_NAMES,
    build_feature_map,
    exact_fidelity_kernel,
)

# Fixed before this script's first run, not blind to test results (see the module docstring).
SCALES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0)
SEED: int = RANDOM_SEED
N_BOOT: int = 1000
APACHE: str = "apache_4a_hospital_death_prob"
OUT_JSON = REPORTS_DIR / "bandwidth_ablation.json"
OUT_FIG = FIGURES_DIR / "bandwidth_ablation.png"

# First three slots of a colorblind-validated categorical palette, one per feature map.
MAP_COLORS = {"zz": "#2a78d6", "pauli": "#eb6834", "custom": "#1baf7a"}
MAP_LABELS = {"zz": "ZZ", "pauli": "Pauli Z+XX", "custom": "custom"}
NEUTRAL = "#52514e"


def score_only_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """ROC-AUC and PR-AUC of a raw score, with the pipeline's bootstrap CIs.

    A raw score has no decision threshold, so only the probability-based metrics apply; the
    y_pred argument of bootstrap_metric_ci is not used for them.
    """
    out = {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
    }
    out.update(
        bootstrap_metric_ci(
            y_true, score, score, metrics=("roc_auc", "pr_auc"), n_boot=N_BOOT, seed=SEED
        )
    )
    return out


def fit_and_score(fm, X_train, y_train, X_eval, s: float):
    """Exact kernels at scale s, SVC fit, predictions on X_eval. Returns (pred, proba, seconds)."""
    t0 = time.perf_counter()
    K_train = exact_fidelity_kernel(fm, s * X_train)
    K_eval = exact_fidelity_kernel(fm, s * X_eval, s * X_train)
    y_pred, y_proba = svc_on_kernel(K_train, y_train, K_eval)
    return y_pred, y_proba, time.perf_counter() - t0


def run() -> dict:
    t_start = time.perf_counter()
    ensure_dataset()
    b = prepare_data(
        quantum_n=DEFAULT_QUANTUM_SUBSAMPLE, quantum_k=DEFAULT_QUBITS, seed=SEED, cache=False
    )
    X_val_q = b.quantum_selector.transform(b.X_val)
    committed = load_results()
    k = b.n_quantum_features

    maps: dict[str, dict] = {}
    for name in FEATURE_MAP_NAMES:
        print(f"--- {name} ---")
        fm = build_feature_map(name, n_features=k, reps=DEFAULT_REPS)
        t0 = time.perf_counter()
        sweep = sweep_scales(fm, b.X_train_q, b.y_train_q, X_val_q, b.y_val, SCALES)
        sweep_seconds = time.perf_counter() - t0
        for r in sweep:
            print(
                f"  s={r['scale']:<5} offdiag_mean={r['offdiag_mean']:.4f} "
                f"val_roc_auc={r['val_roc_auc']:.4f}"
            )
        selected = select_scale({r["scale"]: r["val_roc_auc"] for r in sweep})

        test: dict[str, dict] = {}
        seconds: dict[str, float] = {}
        for label, s in (("s_1", 1.0), ("selected", selected)):
            y_pred, y_proba, secs = fit_and_score(fm, b.X_train_q, b.y_train_q, b.X_test_q, s)
            test[label] = {
                "scale": s,
                **compute_metrics_with_ci(b.y_test_q, y_pred, y_proba, n_boot=N_BOOT, seed=SEED),
            }
            seconds[label] = secs

        ref = committed["qsvm"][f"qsvm_{name}"]
        for key in ("roc_auc", "accuracy"):
            if abs(test["s_1"][key] - ref[key]) > 1e-9:
                raise SystemExit(
                    f"{name}: s=1.0 {key} {test['s_1'][key]} does not reproduce the committed "
                    f"{ref[key]}; stopping without writing outputs."
                )
        print(
            f"  selected s={selected}; test ROC-AUC s=1.0 {test['s_1']['roc_auc']:.4f} "
            f"(committed {ref['roc_auc']:.4f}), selected {test['selected']['roc_auc']:.4f}"
        )
        maps[name] = {
            "sweep": sweep,
            "selected_scale": selected,
            "test": test,
            "committed_pipeline": {
                "roc_auc": ref["roc_auc"],
                "roc_auc_ci_low": ref["roc_auc_ci_low"],
                "roc_auc_ci_high": ref["roc_auc_ci_high"],
                "train_seconds": ref["train_seconds"],
            },
            "seconds": {
                "sweep_all_scales": sweep_seconds,
                "fit_and_score_s_1": seconds["s_1"],
                "fit_and_score_selected": seconds["selected"],
            },
        }

    fitted = train_baseline(b.X_train_q, b.y_train_q, b.X_test_q)
    controls = {
        name: compute_metrics_with_ci(
            b.y_test_q, f.y_pred, f.y_proba, train_seconds=f.train_seconds, n_boot=N_BOOT, seed=SEED
        )
        for name, f in fitted.items()
    }
    if APACHE in b.quantum_feature_names:
        j = b.quantum_feature_names.index(APACHE)
        controls["apache_score_alone"] = score_only_metrics(b.y_test_q, b.X_test_q[:, j])
    jf = b.feature_names.index(APACHE)

    return {
        "description": (
            "Kernel bandwidth ablation for the QSVM feature maps, added after an audit found the "
            "committed kernels at the random-state fidelity. See the docstring of "
            "scripts/ablate_kernel_bandwidth.py for the pre-specified design."
        ),
        "prespecified": {
            "scales": list(SCALES),
            "model": "SVC(kernel='precomputed', C=1.0) on the exact fidelity kernel",
            "selection_rule": (
                "highest validation ROC-AUC per feature map; ties go to the larger scale; "
                "test rows never used for selection"
            ),
            "validation_rows": "full validation split, same 6 quantum features",
            "test_evaluated_at": ["s = 1.0", "selected s"],
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "data": {
            "n_train_q": int(len(b.y_train_q)),
            "n_train_q_positive": int(b.y_train_q.sum()),
            "n_val": int(len(b.y_val)),
            "n_val_positive": int(b.y_val.sum()),
            "n_test_q": int(len(b.y_test_q)),
            "n_test_q_positive": int(b.y_test_q.sum()),
            "quantum_features": b.quantum_feature_names,
            "quantum_train_input_min": float(b.X_train_q.min()),
            "quantum_train_input_max": float(b.X_train_q.max()),
            "n_qubits": k,
            "reps": DEFAULT_REPS,
            "random_state_fidelity": 1.0 / 2**k,
        },
        "feature_maps": maps,
        "matched_controls_quantum_split": controls,
        "reference_full_test_split": {
            "n_test": int(len(b.y_test)),
            "apache_score_alone": score_only_metrics(b.y_test, b.X_test[:, jf]),
            "logreg_committed": {
                key: committed["classical"]["logreg"][key]
                for key in ("roc_auc", "roc_auc_ci_low", "roc_auc_ci_high")
            },
        },
        "total_seconds": time.perf_counter() - t_start,
        "environment": {
            "python": platform.python_version(),
            "qiskit": qiskit.__version__,
            "qiskit_machine_learning": qiskit_machine_learning.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }


def plot(res: dict) -> None:
    ref = res["data"]["random_state_fidelity"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    ax_k, ax_v, ax_t = axes

    for name, m in res["feature_maps"].items():
        s = [r["scale"] for r in m["sweep"]]
        c = MAP_COLORS[name]
        ax_k.plot(s, [r["offdiag_mean"] for r in m["sweep"]], "-o", color=c, lw=2, ms=8)
        ax_v.plot(s, [r["val_roc_auc"] for r in m["sweep"]], "-o", color=c, lw=2, ms=8)
        sel = m["selected_scale"]
        sel_auc = next(r["val_roc_auc"] for r in m["sweep"] if r["scale"] == sel)
        ax_v.plot([sel], [sel_auc], "o", ms=16, mfc="none", mec=c, mew=2)
        ax_k.plot([], [], "-o", color=c, lw=2, ms=8, label=MAP_LABELS[name])

    ax_k.axhline(ref, color=NEUTRAL, ls="--", lw=1.5)
    ax_k.text(
        0.05, ref * 1.15, f"random states, 1/{int(round(1 / ref))}", color=NEUTRAL, fontsize=12
    )
    ax_k.set_yscale("log")
    ax_k.set_ylabel("Mean off-diagonal kernel entry")
    ax_k.set_title("Kernel concentration (train)")
    ax_k.legend(loc="upper right", fontsize=12)

    ax_v.axhline(0.5, color=NEUTRAL, ls="--", lw=1.5)
    ax_v.set_ylabel("Validation ROC-AUC")
    ax_v.set_title(f"Validation ({res['data']['n_val']} rows); ring = selected s")
    for ax in (ax_k, ax_v):
        ax.set_xscale("log")
        ax.set_xticks(res["prespecified"]["scales"])
        ax.set_xticklabels([f"{s:g}" for s in res["prespecified"]["scales"]])
        ax.set_xlabel("Input scale s (s = 1 is the pipeline)")

    rows = []
    for name, m in res["feature_maps"].items():
        for key, tag in (("s_1", "s = 1"), ("selected", f"s = {m['selected_scale']:g}")):
            rows.append((f"QSVM {MAP_LABELS[name]}, {tag}", m["test"][key], MAP_COLORS[name], key))
    ctrl_labels = {
        "logreg": "Logistic regression",
        "svm_rbf": "SVM-RBF",
        "random_forest": "Random forest",
        "apache_score_alone": "APACHE column alone",
    }
    for name, label in ctrl_labels.items():
        if name in res["matched_controls_quantum_split"]:
            rows.append((label, res["matched_controls_quantum_split"][name], NEUTRAL, "ctrl"))
    y = np.arange(len(rows))[::-1]
    for yi, (_label, mt, c, key) in zip(y, rows, strict=True):
        lo, hi = mt["roc_auc_ci_low"], mt["roc_auc_ci_high"]
        ax_t.plot([lo, hi], [yi, yi], color=c, lw=2)
        face = "none" if key == "s_1" else c
        ax_t.plot([mt["roc_auc"]], [yi], "o", ms=9, mfc=face, mec=c, mew=2)
        ax_t.text(hi + 0.01, yi, f"{mt['roc_auc']:.3f}", va="center", fontsize=11, color="0.15")
    ax_t.set_yticks(y)
    ax_t.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax_t.axvline(0.5, color=NEUTRAL, ls="--", lw=1.5)
    ax_t.set_xlim(0.3, 1.0)
    ax_t.set_xlabel("Test ROC-AUC with 95% bootstrap CI")
    ax_t.set_title(f"Same {res['data']['n_test_q']}-row quantum test split")

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)


def main() -> None:
    res = run()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    plot(res)
    print(f"Wrote {OUT_JSON} and {OUT_FIG} in {res['total_seconds']:.1f} s.")


if __name__ == "__main__":
    main()
