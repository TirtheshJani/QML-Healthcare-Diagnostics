"""Train QSVM models with each feature map."""

from __future__ import annotations

import argparse
import warnings

from qml_healthcare.config import DEFAULT_QUANTUM_SUBSAMPLE, DEFAULT_QUBITS, DEFAULT_REPS
from qml_healthcare.pipeline import run_qsvm

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QSVMs across feature maps")
    parser.add_argument("--n", type=int, default=DEFAULT_QUANTUM_SUBSAMPLE)
    parser.add_argument("--k", type=int, default=DEFAULT_QUBITS)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument(
        "--feature-maps",
        nargs="+",
        default=["zz", "pauli", "custom"],
        choices=["zz", "pauli", "custom"],
    )
    args = parser.parse_args()

    metrics = run_qsvm(
        quantum_n=args.n,
        quantum_k=args.k,
        reps=args.reps,
        feature_maps=tuple(args.feature_maps),
    )
    print("\nQSVM metrics:")
    for name, m in metrics.items():
        print(
            f"  {name:>16}  acc={m['accuracy']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  "
            f"roc_auc={m.get('roc_auc', float('nan')):.3f}  f1={m['f1']:.3f}  "
            f"train_s={m['train_seconds']:.1f}"
        )


if __name__ == "__main__":
    main()
