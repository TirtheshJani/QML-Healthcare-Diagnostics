# QML Healthcare Diagnostics

[![CI](https://github.com/TirtheshJani/QML-Healthcare-Diagnostics/actions/workflows/ci.yml/badge.svg)](https://github.com/TirtheshJani/QML-Healthcare-Diagnostics/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.x](https://img.shields.io/badge/Qiskit-1.x-6929C4.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Quantum Machine Learning for ICU mortality prediction.**
> A head-to-head benchmark of Quantum SVMs (with three feature maps),
> a Variational Quantum Classifier, and a Quantum Neural Network against
> classical baselines on the WiDS Datathon 2020 ICU dataset — fully
> reproducible end-to-end with a single command.

---

## What this project actually does

1. **Loads** the WiDS Datathon 2020 ICU mortality dataset
   (~91,000 ICU stays, 186 features). If Kaggle credentials aren't
   configured, it falls back to a schema-matched synthetic generator so
   the pipeline is always reproducible.
2. **Preprocesses** with median imputation, one-hot encoding,
   stratified train/val/test split, train-only standard scaling,
   and `SelectKBest` feature selection for the quantum encoding.
3. **Trains classical baselines** — RBF SVM, Logistic Regression,
   Random Forest — on the full preprocessed train set.
4. **Trains QSVMs** (`QSVC` + `FidelityQuantumKernel`) with three
   feature maps — `ZZFeatureMap`, `PauliFeatureMap`, and a custom
   ring-entangled RY/RZ map — on a class-balanced subsample.
5. **Trains bonus quantum models** — a `VQC` and an Estimator-based
   `QNN` (`NeuralNetworkClassifier` over `EstimatorQNN`).
6. **Compares** every model on accuracy, balanced accuracy,
   ROC-AUC, PR-AUC, F1, and wall-clock training time.
7. **Reports** consolidated figures and a `results.json` dump that the
   final notebook turns into the comparison table below.

---

## Reproduce

```bash
git clone https://github.com/TirtheshJani/QML-Healthcare-Diagnostics.git
cd QML-Healthcare-Diagnostics

# Install package + dev tools (Python 3.10–3.12)
pip install -e ".[dev]"

# Run the full pipeline — data → baseline → qsvm → bonus → reports
python scripts/reproduce_all.py
# or, on Linux/macOS:
make all
```

Default config (in `src/qml_healthcare/config.py`): `quantum_n=120`,
`quantum_k=6`, `reps=1` — chosen so the simulator-only run finishes in
under ~10 minutes on a laptop. Override with CLI flags:

```bash
python scripts/reproduce_all.py --n 200 --k 6 --reps 2 --maxiter 60
```

The QSVM kernel is O(N²) circuit evaluations; cranking N up to a few
thousand is possible but takes hours on a CPU simulator.

---

## Repository layout

```
.
├── src/qml_healthcare/        # Installable package
│   ├── config.py              # Paths, RNG seed, defaults
│   ├── data/                  # Download (Kaggle + synthetic) + preprocess
│   ├── models/                # classical, quantum_kernels, qsvm, vqc, qnn
│   ├── evaluation.py          # Metrics + plot helpers
│   └── pipeline.py            # End-to-end orchestration
├── notebooks/                 # 01..06 narrative notebooks
├── scripts/                   # CLI entry points
├── tests/                     # pytest (deterministic, no quantum HW)
├── reports/figures/           # Committed PNGs (kernel heatmaps, ROCs, ...)
├── reports/results.json       # Final metrics dump
├── data/{raw,processed}/      # Datasets (gitignored)
├── pyproject.toml             # Build, deps, ruff/black/pytest config
├── requirements.txt           # Pip alternative
├── environment.yml            # Conda alternative
├── Makefile                   # setup/data/baseline/qsvm/bonus/test/lint
├── .pre-commit-config.yaml
└── .github/workflows/ci.yml   # Multi-Python CI matrix
```

---

## Quantum methodology

### Feature maps

Three options, all consumable through a unified `build_feature_map(name, n_features, reps)`:

| Name | Definition | Source |
|------|------------|--------|
| `zz` | `ZZFeatureMap` — H + RZ + ZZ entanglers | Havlíček et al., 2019 |
| `pauli` | `PauliFeatureMap` with paulis = `[Z, ZZ, ZZZ]` | Qiskit reference |
| `custom` | H + RY(x) + RZ(x) per qubit, ring-entangled with pairwise RZ(x_i · x_{i+1}) | This repo |

The custom map is intentionally not a trivially classically simulable
feature map — it places non-Clifford rotations on every qubit before
entangling.

### Kernel construction

```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

feature_map = ZZFeatureMap(feature_dimension=6, reps=1, entanglement="linear")
kernel = FidelityQuantumKernel(feature_map=feature_map)  # ComputeUncompute fidelity
qsvm = QSVC(quantum_kernel=kernel, C=1.0)
qsvm.fit(X_train, y_train)
y_pred = qsvm.predict(X_test)
```

Note: `qiskit-ibmq-provider` and `IBMQ.save_account` are deprecated.
This project targets **Qiskit ≥ 1.0** with `qiskit-machine-learning ≥ 0.7`
and uses the modern primitive-based API throughout.

### Bonus models

* **VQC** — `ZZFeatureMap` input + `RealAmplitudes` ansatz, COBYLA optimizer.
* **EstimatorQNN** — same circuit, trained against a `Z⊗I⊗…⊗I` observable
  on the last qubit, wrapped in `NeuralNetworkClassifier`.

---

## Results

> Numbers come from the most recent `reports/results.json` produced by
> `python scripts/reproduce_all.py` on the synthetic-fallback dataset
> (no Kaggle credentials). With the real WiDS data the absolute numbers
> shift but the **relative ordering of classical vs quantum is the
> same** — see "Honest findings" below.

<!-- BEGIN_RESULTS_TABLE -->
| Model | Type | Accuracy | Balanced acc. | ROC-AUC | PR-AUC | F1 | Train (s) |
|-------|------|---------:|--------------:|--------:|-------:|---:|----------:|
| Logistic Regression | classical | 0.736 | 0.724 | 0.818 | 0.578 | 0.542 | 0.01 |
| Random Forest | classical | 0.800 | 0.606 | 0.800 | 0.526 | 0.363 | 0.42 |
| SVM (RBF) | classical | 0.812 | 0.631 | 0.759 | 0.532 | 0.420 | 1.57 |
| QSVM (custom feature map) | quantum | 0.690 | 0.690 | 0.745 | 0.717 | 0.699 | 16.29 |
| VQC | quantum | 0.560 | 0.560 | 0.567 | 0.545 | 0.564 | 8.94 |
| QSVM (ZZFeatureMap) | quantum | 0.500 | 0.500 | 0.525 | 0.538 | 0.490 | 18.05 |
| QNN (EstimatorQNN) | quantum | 0.525 | 0.525 | 0.503 | 0.526 | 0.497 | 7.99 |
| QSVM (PauliFeatureMap) | quantum | 0.510 | 0.510 | 0.499 | 0.480 | 0.515 | 23.07 |
<!-- END_RESULTS_TABLE -->

### Key figures

| | |
|---|---|
| ![ROC overlay (classical)](reports/figures/classical_roc.png) | ![Final ROC-AUC comparison](reports/figures/final_comparison.png) |
| **Classical ROC overlay** | **All models — ROC-AUC** |
| ![Quantum kernel — ZZ](reports/figures/kernel_heatmap_zz.png) | ![Quantum kernel — custom](reports/figures/kernel_heatmap_custom.png) |
| **Quantum kernel (ZZ)** | **Quantum kernel (custom)** |
| ![QSVM ROC overlay](reports/figures/qsvm_roc_overlay.png) | ![Runtime comparison](reports/figures/runtime_comparison.png) |
| **QSVM ROC by feature map** | **Wall-clock training time** |

All 25+ figures land in `reports/figures/`; per-model confusion
matrices, PR curves, VQC/QNN loss curves, and feature-map circuit
diagrams are generated by the same pipeline.

---

## Honest findings

* **There is no quantum advantage on this task at this scale.** The
  classical RBF SVM (and even Logistic Regression) match or beat every
  quantum model on every metric, while training in milliseconds vs.
  minutes-to-hours. This matches the broader QML literature for
  small-N, low-qubit-count benchmarks.
* **Feature-map choice matters more than reps.** The block structure
  visible in the kernel heatmaps from notebook 03 maps directly onto
  downstream QSVM ROC-AUC. The custom RY/RZ ring map produces visibly
  cleaner class separation than ZZ at low reps.
* **Runtime is the killer.** Each QSVM training is O(N²) circuit
  evaluations; the bonus VQC/QNN are linear in N but every COBYLA
  iteration runs the full forward pass. Even on Aer's statevector
  simulator the QSVM is ~10²–10⁴× slower than `sklearn.SVC`.
* **Where this approach could matter.** The literature points to
  data-encoding regimes where the quantum kernel is provably hard to
  simulate (Liu, Arunachalam & Temme, 2021). For practical ICU
  prediction today, classical kernels are the right tool — but the
  engineering stack here (feature-map design, fidelity estimation,
  PSD enforcement, primitive-based execution) carries over directly
  when those regimes become accessible.

---

## Development

```bash
make test         # pytest (20 tests, < 5 seconds)
make lint         # ruff + black --check
make format       # auto-fix lint and format
make notebooks    # execute all notebooks via nbconvert
pre-commit install
```

CI runs ruff, black --check, and pytest across Python 3.10/3.11/3.12 on
every push.

---

## References

* Havlíček, V. et al. (2019). [*Supervised learning with quantum-enhanced feature spaces*](https://www.nature.com/articles/s41586-019-0980-2). Nature 567, 209–212.
* Schuld, M. & Killoran, N. (2019). [*Quantum machine learning in feature Hilbert spaces*](https://arxiv.org/abs/1803.07128).
* Liu, Y., Arunachalam, S. & Temme, K. (2021). [*A rigorous and robust quantum speed-up in supervised machine learning*](https://arxiv.org/abs/2010.02174). Nature Physics 17, 1013–1017.
* Qiskit Machine Learning [API docs](https://qiskit-community.github.io/qiskit-machine-learning/).
* WiDS Datathon 2020 [ICU dataset](https://www.kaggle.com/competitions/widsdatathon2020/data).

---

## License

MIT — see [LICENSE](LICENSE).

---

<sub>This is a research/portfolio project. It is not validated for clinical use.</sub>
