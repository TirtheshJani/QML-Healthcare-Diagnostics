"""Render ``reports/results.json`` as markdown tables.

Single source of truth for the results table so the README and the docs site never
drift. ``scripts/update_readme_table.py`` injects ``build_table`` output into the README;
``scripts/build_docs_tables.py`` writes the same tables (plus split-out classical/quantum
variants) into the docs site.

Every value rendered here comes straight from the metrics dump. Nothing is hard-coded, so
the tables stay honest by construction.
"""

from __future__ import annotations

PRETTY: dict[str, str] = {
    "svm_rbf": "SVM (RBF)",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "qsvm_zz": "QSVM (ZZFeatureMap)",
    "qsvm_pauli": "QSVM (Pauli Z+XX)",
    "qsvm_custom": "QSVM (custom feature map)",
    "vqc": "VQC",
    "qnn": "QNN (SamplerQNN)",
}
KIND: dict[str, str] = {
    "svm_rbf": "classical",
    "logreg": "classical",
    "random_forest": "classical",
    "qsvm_zz": "quantum",
    "qsvm_pauli": "quantum",
    "qsvm_custom": "quantum",
    "vqc": "quantum",
    "qnn": "quantum",
}


def flatten(results: dict) -> dict[str, dict[str, float]]:
    """Collapse the ``{classical, qsvm, bonus}`` hierarchy into one model -> metrics map."""
    flat: dict[str, dict[str, float]] = {}
    flat.update(results.get("classical", {}))
    flat.update(results.get("qsvm", {}))
    flat.update(results.get("bonus", {}))
    return flat


def fmt_roc_auc(m: dict[str, float]) -> str:
    """ROC-AUC with its 95% bootstrap CI when present, e.g. ``0.745 [0.65, 0.83]``."""
    v = m.get("roc_auc", float("nan"))
    lo = m.get("roc_auc_ci_low")
    hi = m.get("roc_auc_ci_high")
    if lo is not None and hi is not None:
        return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
    return f"{v:.3f}"


def row(name: str, m: dict[str, float]) -> str:
    """Render one model as a markdown table row."""
    return (
        f"| {PRETTY.get(name, name)} | {KIND.get(name, '?')} | "
        f"{m.get('accuracy', float('nan')):.3f} | "
        f"{m.get('balanced_accuracy', float('nan')):.3f} | "
        f"{fmt_roc_auc(m)} | "
        f"{m.get('pr_auc', float('nan')):.3f} | "
        f"{m.get('f1', float('nan')):.3f} | "
        f"{m.get('train_seconds', float('nan')):.2f} |"
    )


def build_table(results: dict) -> str:
    """Build a markdown table from a results dict, sorted by ROC-AUC descending."""
    flat = flatten(results)
    if not flat:
        return "_No results yet - run `python scripts/reproduce_all.py`._"
    ordered = sorted(flat.items(), key=lambda kv: -kv[1].get("roc_auc", 0))
    header = (
        "| Model | Type | Accuracy | Balanced acc. | ROC-AUC [95% CI] | PR-AUC | F1 | Train (s) |\n"
        "|-------|------|---------:|--------------:|:----------------|-------:|---:|----------:|"
    )
    body = "\n".join(row(n, m) for n, m in ordered)
    return f"{header}\n{body}"
