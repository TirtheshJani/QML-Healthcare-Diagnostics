"""Train VQC and QNN bonus models."""

from __future__ import annotations

import argparse
import warnings

from qml_healthcare.config import DEFAULT_QUANTUM_SUBSAMPLE, DEFAULT_QUBITS, DEFAULT_REPS
from qml_healthcare.pipeline import run_bonus

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VQC and QNN bonus models")
    parser.add_argument("--n", type=int, default=DEFAULT_QUANTUM_SUBSAMPLE)
    parser.add_argument("--k", type=int, default=DEFAULT_QUBITS)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--maxiter", type=int, default=60)
    args = parser.parse_args()

    metrics = run_bonus(quantum_n=args.n, quantum_k=args.k, reps=args.reps, maxiter=args.maxiter)
    print("\nBonus model metrics:")
    for name, m in metrics.items():
        print(
            f"  {name:>4}  acc={m['accuracy']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  "
            f"roc_auc={m.get('roc_auc', float('nan')):.3f}  f1={m['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
