# Reproducibility

Every number, figure, and table in this project is regenerated from a fixed seed by one command.

## Environment

- Python 3.10 to 3.12. On Windows, use the launcher: `py -3.11`.
- Verified stack: Qiskit 1.4.5, qiskit-machine-learning 0.8.4, scikit-learn 1.8, pandas 3.0,
  numpy 2.4.

Install the package with its development tools:

```bash
py -3.11 -m pip install -e ".[dev]"
```

## Reproduce everything

```bash
py -3.11 scripts/reproduce_all.py
```

This runs all five pipeline stages (data, classical baseline, QSVM, VQC and QNN, reports) and then
refreshes the README results table. Override the defaults when you want a quicker or larger run:

```bash
py -3.11 scripts/reproduce_all.py --n 200 --k 6 --reps 2 --maxiter 60
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--n` | 200 | quantum training subsample (the kernel is O(N squared)) |
| `--k` | 6 | features selected for the quantum encoding (qubits) |
| `--reps` | 2 | feature-map and ansatz repetitions |
| `--maxiter` | 60 | COBYLA iterations for the VQC and QNN |

Individual stages are also available as scripts: `scripts/download_data.py`,
`scripts/train_baseline.py`, `scripts/train_qsvm.py`, and `scripts/train_vqc_qnn.py`.

## Determinism and the data caveat

All randomness is seeded with `RANDOM_SEED = 42`: the splits, the model fits (including the VQC and
QNN initial weights and shot sampling), and the bootstrap resampling. Re-running the pipeline
reproduces the same metrics; only the wall-clock training times change.

Without Kaggle credentials the pipeline uses the synthetic fallback, so the numbers shown on this site
come from synthetic data. They are internally consistent and reproducible, but they are not the real
WiDS results. To run on the real data, place a Kaggle API token at `~/.kaggle/kaggle.json` and re-run.

## Rebuild this site

```bash
py -3.11 -m pip install -e ".[docs]"
py -3.11 scripts/sync_docs_assets.py    # tables + chart data + demo model + figures
py -3.11 -m mkdocs serve                # preview at http://127.0.0.1:8000/
py -3.11 -m mkdocs build --strict       # production build into site/
```

`scripts/sync_docs_assets.py` regenerates the results table, the chart data, and the demo model, then
copies the figures into the site. The deploy workflow runs the same steps in CI.

## Tests

```bash
py -3.11 -m pytest
```

The suite is deterministic and offline (it uses the synthetic generator), and it includes a check that
the exported demo model's JavaScript math matches scikit-learn, and that the docs tables carry no
fabricated numbers.
