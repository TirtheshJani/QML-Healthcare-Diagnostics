"""Replace the BEGIN_RESULTS_TABLE / END_RESULTS_TABLE block in README.md
with a markdown table built from reports/results.json.

Idempotent — run it after every pipeline run to keep the README honest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "reports" / "results.json"


PRETTY = {
    "svm_rbf": "SVM (RBF)",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "qsvm_zz": "QSVM (ZZFeatureMap)",
    "qsvm_pauli": "QSVM (Pauli Z+XX)",
    "qsvm_custom": "QSVM (custom feature map)",
    "vqc": "VQC",
    "qnn": "QNN (SamplerQNN)",
}
KIND = {
    "svm_rbf": "classical",
    "logreg": "classical",
    "random_forest": "classical",
    "qsvm_zz": "quantum",
    "qsvm_pauli": "quantum",
    "qsvm_custom": "quantum",
    "vqc": "quantum",
    "qnn": "quantum",
}


def _flatten(results: dict) -> dict[str, dict[str, float]]:
    flat = {}
    flat.update(results.get("classical", {}))
    flat.update(results.get("qsvm", {}))
    flat.update(results.get("bonus", {}))
    return flat


def _fmt_roc_auc(m: dict[str, float]) -> str:
    """ROC-AUC with its 95% bootstrap CI when present, e.g. ``0.745 [0.65, 0.83]``."""
    v = m.get("roc_auc", float("nan"))
    lo = m.get("roc_auc_ci_low")
    hi = m.get("roc_auc_ci_high")
    if lo is not None and hi is not None:
        return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
    return f"{v:.3f}"


def _row(name: str, m: dict[str, float]) -> str:
    return (
        f"| {PRETTY.get(name, name)} | {KIND.get(name, '?')} | "
        f"{m.get('accuracy', float('nan')):.3f} | "
        f"{m.get('balanced_accuracy', float('nan')):.3f} | "
        f"{_fmt_roc_auc(m)} | "
        f"{m.get('pr_auc', float('nan')):.3f} | "
        f"{m.get('f1', float('nan')):.3f} | "
        f"{m.get('train_seconds', float('nan')):.2f} |"
    )


def build_table(results: dict) -> str:
    flat = _flatten(results)
    if not flat:
        return "_No results yet — run `python scripts/reproduce_all.py`._"
    ordered = sorted(flat.items(), key=lambda kv: -kv[1].get("roc_auc", 0))
    header = (
        "| Model | Type | Accuracy | Balanced acc. | ROC-AUC [95% CI] | PR-AUC | F1 | Train (s) |\n"
        "|-------|------|---------:|--------------:|:----------------|-------:|---:|----------:|"
    )
    body = "\n".join(_row(n, m) for n, m in ordered)
    return f"{header}\n{body}"


def main() -> None:
    if not RESULTS.exists():
        raise FileNotFoundError(f"{RESULTS} not found — run the pipeline first.")
    with RESULTS.open(encoding="utf-8") as f:
        results = json.load(f)
    table = build_table(results)

    text = README.read_text(encoding="utf-8")
    pattern = re.compile(r"<!-- BEGIN_RESULTS_TABLE -->.*?<!-- END_RESULTS_TABLE -->", re.DOTALL)
    new_block = f"<!-- BEGIN_RESULTS_TABLE -->\n{table}\n<!-- END_RESULTS_TABLE -->"
    if not pattern.search(text):
        raise RuntimeError("README is missing BEGIN_RESULTS_TABLE / END_RESULTS_TABLE markers.")
    new_text = pattern.sub(new_block, text)
    README.write_text(new_text, encoding="utf-8")
    print(f"Updated {README} with {len(_flatten(results))} model rows.")


if __name__ == "__main__":
    main()
