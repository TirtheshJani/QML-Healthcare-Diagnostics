"""Reproduce every figure and number in the README."""

from __future__ import annotations

import argparse
import warnings

from qml_healthcare.config import DEFAULT_QUANTUM_SUBSAMPLE, DEFAULT_QUBITS, DEFAULT_REPS
from qml_healthcare.pipeline import run_all

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the full pipeline")
    parser.add_argument("--n", type=int, default=DEFAULT_QUANTUM_SUBSAMPLE)
    parser.add_argument("--k", type=int, default=DEFAULT_QUBITS)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--maxiter", type=int, default=60)
    args = parser.parse_args()

    run_all(
        quantum_n=args.n,
        quantum_k=args.k,
        reps=args.reps,
        bonus_maxiter=args.maxiter,
    )
    # Refresh the README results table from the freshest results.json
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "scripts/update_readme_table.py"])
    print("\nAll stages complete. See reports/figures/ and reports/results.json.")


if __name__ == "__main__":
    main()
