"""Train classical baselines and write metrics + figures."""

from __future__ import annotations

import warnings

from qml_healthcare.pipeline import run_baseline

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    metrics = run_baseline()
    print("\nClassical baseline metrics:")
    for name, m in metrics.items():
        print(
            f"  {name:>14}  acc={m['accuracy']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  "
            f"roc_auc={m.get('roc_auc', float('nan')):.3f}  f1={m['f1']:.3f}"
        )
