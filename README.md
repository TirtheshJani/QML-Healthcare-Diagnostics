# QML Healthcare Diagnostics

[![CI](https://github.com/TirtheshJani/QML-Healthcare-Diagnostics/actions/workflows/ci.yml/badge.svg)](https://github.com/TirtheshJani/QML-Healthcare-Diagnostics/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.x](https://img.shields.io/badge/Qiskit-1.x-6929C4.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docs: live site](https://img.shields.io/badge/docs-live%20site-2965f1.svg)](https://tirtheshjani.github.io/QML-Healthcare-Diagnostics/)

> **Quantum Machine Learning for ICU mortality prediction.**  
> A reproducible, end-to-end benchmark of Quantum SVMs (three feature maps),
> a Variational Quantum Classifier, and a Quantum Neural Network against
> classical baselines, all on the WiDS Datathon 2020 ICU dataset.
> Runs offline from a single command using a schema-matched synthetic fallback
> when Kaggle credentials are not available.

**Live site and interactive demo:** <https://tirtheshjani.github.io/QML-Healthcare-Diagnostics/>
(documentation, interactive charts, and a client-side classical prediction demo, built with
`docs/` + MkDocs and deployed via GitHub Pages).

---

## Table of Contents

- [What this project does](#what-this-project-does)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Notebooks](#notebooks)
- [Quantum methodology](#quantum-methodology)
- [Results](#results)
- [Honest findings](#honest-findings)
- [Development](#development)
- [References](#references)
- [License](#license)

---

## What this project does

| Stage | Description |
|-------|-------------|
| **1. Data** | Downloads the WiDS 2020 ICU dataset (~91 k stays, 186 features) via Kaggle, or synthesises a schema-matched 5 000-row dataset when credentials are absent. |
| **2. Preprocess** | Median imputation → one-hot encode → stratified 70/10/20 train/val/test split → train-only `StandardScaler` → `SelectKBest` (ANOVA-F) to pick the top *k* features for quantum encoding. |
| **3. Classical baselines** | RBF SVM, Logistic Regression, Random Forest on the full preprocessed feature set. |
| **4. QSVM** | `QSVC` + `FidelityQuantumKernel` with three feature maps on a class-balanced subsample (O(N²) kernel constraint). |
| **5. Bonus quantum models** | Variational Quantum Classifier (`VQC`) and a `SamplerQNN`-based `NeuralNetworkClassifier`. |
| **6. Evaluation** | Accuracy, balanced accuracy, ROC-AUC, PR-AUC, F1, and wall-clock training time per model. Outputs figures to `reports/figures/` and a JSON metrics dump. |

---

## Quick start

```bash
git clone https://github.com/TirtheshJani/QML-Healthcare-Diagnostics.git
cd QML-Healthcare-Diagnostics

# Install the package and dev tools (Python 3.10–3.12)
pip install -e ".[dev]"

# Run every stage — data → baseline → QSVM → bonus → reports
python scripts/reproduce_all.py
# or, on Linux/macOS:
make all
```

The pipeline automatically falls back to synthetic data when Kaggle credentials
are absent, so **no account is needed** to run a complete experiment.

### Optional: real WiDS data

```bash
# Put your Kaggle API key in ~/.kaggle/kaggle.json, then:
pip install -e ".[kaggle]"
python scripts/reproduce_all.py     # will download training_v2.csv automatically
```

### Override defaults

```bash
python scripts/reproduce_all.py --n 400 --k 8 --reps 2 --maxiter 100
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--n` | 200 | Quantum subsample size N (QSVM kernel is O(N²)) |
| `--k` | 6 | Number of features selected for quantum encoding |
| `--reps` | 2 | Feature-map / ansatz repetitions |
| `--maxiter` | 60 | COBYLA iterations for VQC and QNN |

---

## Repository layout

```
.
├── src/qml_healthcare/        # Installable Python package
│   ├── config.py              # Paths, RNG seed, feature lists, defaults
│   ├── data/
│   │   ├── download.py        # Kaggle download + synthetic generator
│   │   ├── loader.py          # Raw CSV reader
│   │   └── preprocess.py      # Full preprocessing pipeline → DataBundle
│   ├── models/
│   │   ├── _base.py           # FittedModel dataclass (shared by all trainers)
│   │   ├── classical.py       # SVM-RBF, Logistic Regression, Random Forest
│   │   ├── quantum_kernels.py # Feature maps + FidelityQuantumKernel wrapper
│   │   ├── qsvm.py            # QSVC trainer
│   │   ├── vqc.py             # VQC trainer (ZZFeatureMap + RealAmplitudes)
│   │   └── qnn.py             # SamplerQNN + NeuralNetworkClassifier trainer
│   ├── evaluation.py          # Metrics computation + all plot helpers
│   └── pipeline.py            # Orchestration: run_data / run_baseline / run_qsvm
│                              #               / run_bonus / run_reports / run_all
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_classical_baseline.ipynb
│   ├── 03_quantum_kernels.ipynb
│   ├── 04_qsvm_training.ipynb
│   ├── 05_vqc_qnn_bonus.ipynb
│   └── 06_results_analysis.ipynb
├── scripts/
│   ├── reproduce_all.py       # Full pipeline CLI (calls run_all + update_readme_table)
│   ├── download_data.py       # Standalone data acquisition
│   ├── train_baseline.py      # Classical baselines only
│   ├── train_qsvm.py          # QSVM only (--feature-maps zz pauli custom)
│   ├── train_vqc_qnn.py       # VQC + QNN only
│   └── update_readme_table.py # Refresh the results table in this README
├── tests/                     # pytest — 28 deterministic tests, fast
├── reports/
│   ├── figures/               # All generated PNGs (committed)
│   └── results.json           # Latest metrics dump
├── data/{raw,processed}/      # Dataset files (gitignored)
├── pyproject.toml             # Build, deps, ruff/black/pytest/coverage config
├── requirements.txt           # pip install -r alternative
├── environment.yml            # Conda environment spec
├── Makefile                   # Convenience targets
├── .pre-commit-config.yaml    # ruff + black hooks
└── .github/workflows/ci.yml   # Matrix CI: Python 3.10 / 3.11 / 3.12
```

---

## Configuration

All shared constants live in `src/qml_healthcare/config.py`:

```python
RANDOM_SEED              = 42
DEFAULT_QUBITS           = 6    # k — features selected for quantum encoding
DEFAULT_REPS             = 2    # feature-map / ansatz reps
DEFAULT_QUANTUM_SUBSAMPLE = 200  # N — training points for QSVM/VQC/QNN
```

The 17 curated features (16 numeric + 1 binary surgical flag) cover
vitals, lab values, pre-ICU length of stay, GCS components, and
the APACHE IV hospital-death probability estimate.

---

## Notebooks

Each notebook is self-contained but shares the installed `qml_healthcare`
package, so any cell can be re-run independently after `pip install -e .`.

| Notebook | Content |
|----------|---------|
| `01_data_exploration` | Shape, class balance (~8 % mortality), missingness patterns, feature distributions stratified by outcome, correlation heatmap |
| `02_classical_baseline` | Trains and evaluates SVM-RBF, Logistic Regression, Random Forest; produces ROC/PR curves and confusion matrices |
| `03_quantum_kernels` | Draws the three feature-map circuits; computes and visualises 60×60 kernel matrices; analyses eigenvalue spectra |
| `04_qsvm_training` | Trains one QSVC per feature map; overlaid ROC curves; per-map confusion matrices |
| `05_vqc_qnn_bonus` | VQC (ZZFeatureMap + RealAmplitudes) and QNN (PauliFeatureMap + RealAmplitudes via SamplerQNN) training with loss curves |
| `06_results_analysis` | Final comparison table from `results.json`; key findings discussion; runtime breakdown |

---

## Quantum methodology

### Feature maps

All three maps share the same interface: `build_feature_map(name, n_features, reps)`.

| Name | Circuit | Reference |
|------|---------|-----------|
| `zz` | `ZZFeatureMap` — H layer → RZ(2φ(x)) → ZZ entanglers | Havlíček et al., 2019 |
| `pauli` | `PauliFeatureMap` with `paulis=["Z", "XX"]` — X-basis entanglement | Qiskit reference |
| `custom` | H → RZ(2x) per qubit → CZ entanglement (ring) — explicit non-Clifford map | This repo |

> **Why `["Z", "XX"]` and not `["Z", "ZZ"]`?** `ZZFeatureMap` is *exactly*
> `PauliFeatureMap(paulis=["Z", "ZZ"])`, so the two would produce identical
> kernels and identical results. The `pauli` map deliberately uses `XX`
> entanglement so the three feature maps form a genuine comparison rather than
> two duplicates plus one. A regression test
> (`tests/test_quantum_kernels.py::test_zz_and_pauli_feature_maps_differ`) locks
> this in.

The custom map applies a Hadamard to all qubits, encodes each feature as
`RZ(2xᵢ)`, then entangles adjacent pairs with `CZ` gates. Unlike ZZFeatureMap,
the entanglement step is a fixed Clifford (`CZ`), which makes the non-classical
contribution come entirely from the data-dependent rotations.

### Kernel construction

```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

feature_map = ZZFeatureMap(feature_dimension=6, reps=2)
kernel = FidelityQuantumKernel(feature_map=feature_map)  # ComputeUncompute fidelity
qsvc = QSVC(quantum_kernel=kernel)
qsvc.fit(X_train_q, y_train_q)
y_pred = qsvc.predict(X_test_q)
```

`FidelityQuantumKernel` uses the ComputeUncompute circuit to estimate
K(x, x') = |⟨φ(x)|φ(x')⟩|². The kernel matrix is guaranteed PSD by
`enforce_psd=True` (default).

> **API note:** This project targets **Qiskit ≥ 1.0 / qiskit-machine-learning ≥ 0.7**
> and uses the modern primitive-based API throughout. The deprecated
> `qiskit-ibmq-provider` / `IBMQ.save_account` pattern is not used anywhere.

### Bonus quantum models

| Model | Architecture | Optimizer |
|-------|-------------|-----------|
| **VQC** | `ZZFeatureMap` (input) + `RealAmplitudes` ansatz + cross-entropy loss | COBYLA via `scipy.optimize.minimize` |
| **QNN** | `PauliFeatureMap` + `RealAmplitudes` via `SamplerQNN` + `NeuralNetworkClassifier` | COBYLA |

Both use the `StatevectorSampler` primitive from `qiskit.primitives` for
exact statevector simulation. The QNN uses a parity interpret function
(`x % 2`) to produce a 2-class probability output, trained with one-hot
cross-entropy loss (`NeuralNetworkClassifier(loss="cross_entropy", one_hot=True)`),
which is required for a 2-output `SamplerQNN`.

### Uncertainty estimates

Point metrics from a single split can be misleading on a subsampled test set
(N = 200 for the quantum models). Two complementary checks are reported:

- **Bootstrap confidence intervals (all models).** The test predictions are
  resampled with replacement (1000 seeded draws) to produce 95% CIs for ROC-AUC,
  PR-AUC, F1, balanced accuracy, and accuracy. These appear next to ROC-AUC in
  the results table and as error bars in
  `reports/figures/roc_auc_ci_comparison.png`. No retraining is needed, so the
  same procedure applies uniformly to classical and quantum models.
- **5-fold stratified cross-validation (classical only).** The classical
  baselines are cheap to refit, so they also report CV mean ± std for ROC-AUC,
  F1, and balanced accuracy. The quantum models are *not* cross-validated: the
  fidelity kernel is O(N²) per fold, which is prohibitive on a CPU simulator.
  This asymmetry is intentional and called out here so the comparison stays
  honest.

---

## Results

> Numbers come from the most recent `reports/results.json` produced by
> `python scripts/reproduce_all.py` on the **synthetic-fallback dataset**
> (no Kaggle credentials). Real WiDS data was not run in this reproduction.
> Brackets are 95% bootstrap CIs; the classical rows additionally have 5-fold
> CV (see [Uncertainty estimates](#uncertainty-estimates)). Train times are
> wall-clock and machine-dependent: the VQC and QNN rows were regenerated on a
> slower Linux machine after their seeding was fixed, while the other rows come
> from the original run. See
> [Honest findings](#honest-findings) for what these numbers do and do not show.

<!-- BEGIN_RESULTS_TABLE -->
| Model | Type | Accuracy | Balanced acc. | ROC-AUC [95% CI] | PR-AUC | F1 | Train (s) |
|-------|------|---------:|--------------:|:----------------|-------:|---:|----------:|
| Logistic Regression | classical | 0.810 | 0.651 | 0.817 [0.787, 0.845] | 0.578 | 0.460 | 0.01 |
| Random Forest | classical | 0.808 | 0.646 | 0.792 [0.758, 0.824] | 0.540 | 0.451 | 0.54 |
| SVM (RBF) | classical | 0.812 | 0.631 | 0.759 [0.721, 0.796] | 0.532 | 0.420 | 1.10 |
| QSVM (Pauli Z+XX) | quantum | 0.520 | 0.520 | 0.522 [0.440, 0.600] | 0.518 | 0.543 | 174.46 |
| QSVM (ZZFeatureMap) | quantum | 0.535 | 0.535 | 0.513 [0.434, 0.590] | 0.518 | 0.551 | 104.08 |
| QSVM (custom feature map) | quantum | 0.490 | 0.490 | 0.513 [0.437, 0.591] | 0.513 | 0.474 | 36.96 |
| QNN (SamplerQNN) | quantum | 0.505 | 0.505 | 0.484 [0.399, 0.559] | 0.477 | 0.526 | 129.85 |
| VQC | quantum | 0.450 | 0.450 | 0.437 [0.353, 0.518] | 0.459 | 0.476 | 132.97 |
<!-- END_RESULTS_TABLE -->

### Key figures

| | |
|---|---|
| ![ROC overlay (classical)](reports/figures/classical_roc.png) | ![ROC-AUC with 95% CI](reports/figures/roc_auc_ci_comparison.png) |
| **Classical baselines — ROC** | **All models — ROC-AUC with 95% bootstrap CI** |
| ![Quantum kernel — ZZ](reports/figures/kernel_heatmap_zz.png) | ![Quantum kernel — custom](reports/figures/kernel_heatmap_custom.png) |
| **Quantum kernel (ZZ feature map)** | **Quantum kernel (custom feature map)** |
| ![QSVM ROC overlay](reports/figures/qsvm_roc_overlay.png) | ![Runtime comparison](reports/figures/runtime_comparison.png) |
| **QSVM — ROC by feature map** | **Wall-clock training time** |

All figures are generated to `reports/figures/`; the pipeline also produces
per-model confusion matrices, PR curves, VQC/QNN loss curves, and Pauli
kernel heatmaps.

---

## Honest findings

- **The quantum models are not distinguishable from random here.** On this
  synthetic benchmark at N = 200 and 6 qubits, every quantum model's 95%
  bootstrap CI for ROC-AUC includes 0.5 (QSVM-Pauli 0.52 [0.44, 0.60], QSVM-ZZ
  0.51 [0.43, 0.59], QSVM-custom 0.51 [0.44, 0.59], QNN 0.48 [0.40, 0.56],
  VQC 0.44 [0.35, 0.52]). Before the VQC and QNN were seeded, an earlier
  committed run gave VQC 0.54 and QNN 0.51, so their point estimates move by up
  to 0.1 between seeds while staying inside chance-level CIs. The honest
  statement is not "quantum is worse" but
  "quantum is indistinguishable from chance on this data" — exactly what the CIs
  are there to make testable.

- **The classical baselines have real, stable signal.** Logistic Regression
  reaches ROC-AUC 0.82 [0.79, 0.85] with a CI well above 0.5, and its 5-fold CV
  mean (0.81 ± 0.01) matches the single-split estimate — so the classical result
  is not a lucky split. SVM-RBF and Random Forest behave the same way.

- **Feature-map choice is within noise at this scale.** The three QSVM feature
  maps (ZZ, Pauli Z+XX, custom) land within ~0.01 ROC-AUC of one another and all
  overlap 0.5. With only N = 200 the fidelity kernel is close to the identity
  (most off-diagonal entries near 0), so the SVM has little structure to exploit;
  the near-orthogonal kernel heatmaps in notebook 03 show this directly.

- **Runtime is the bottleneck.** Each QSVM training requires O(N²) circuit
  evaluations; VQC/QNN are linear in N but every COBYLA iteration runs the
  full forward pass on all training points. On the statevector simulator the
  quantum models take ~37-174 s versus ~0.006-1.1 s for the classical baselines
  (roughly 10²-10⁴× slower).

- **Where quantum kernels could matter.** Liu, Arunachalam & Temme (2021)
  identify data-encoding regimes where the quantum kernel is provably
  classically hard to approximate. For practical ICU mortality prediction
  today, classical kernels are the right tool — but the engineering stack
  here (feature-map design, fidelity estimation, PSD enforcement,
  primitive-based execution) carries over directly when those regimes become
  accessible on fault-tolerant hardware.

---

## Development

```bash
make test         # pytest — 28 deterministic tests, fast
make lint         # ruff + black --check
make format       # auto-fix formatting and lint
make notebooks    # execute all notebooks via nbconvert
pre-commit install
```

CI runs `ruff`, `black --check`, and `pytest` across Python 3.10 / 3.11 / 3.12
on every push and pull request.

> **Verified stack.** Most recently reproduced on Windows with Python 3.11 and
> Qiskit 1.4.5, qiskit-machine-learning 0.8.4, scikit-learn 1.8, pandas 3.0,
> numpy 2.4. On Windows use `python scripts/reproduce_all.py` (the `make`
> targets assume a Unix shell).

### Individual CLI scripts

```bash
# Data only
python scripts/download_data.py

# Classical baselines
python scripts/train_baseline.py

# QSVM — choose feature maps
python scripts/train_qsvm.py --n 200 --k 6 --reps 2 --feature-maps zz pauli custom

# VQC + QNN
python scripts/train_vqc_qnn.py --n 200 --k 6 --reps 2 --maxiter 60
```

---

## References

- Havlíček, V. et al. (2019). [*Supervised learning with quantum-enhanced feature spaces*](https://www.nature.com/articles/s41586-019-0980-2). Nature 567, 209–212.
- Schuld, M. & Killoran, N. (2019). [*Quantum machine learning in feature Hilbert spaces*](https://arxiv.org/abs/1803.07128). PRL 122, 040504.
- Liu, Y., Arunachalam, S. & Temme, K. (2021). [*A rigorous and robust quantum speed-up in supervised machine learning*](https://arxiv.org/abs/2010.02174). Nature Physics 17, 1013–1017.
- Qiskit Machine Learning [documentation](https://qiskit-community.github.io/qiskit-machine-learning/).
- WiDS Datathon 2020 [ICU dataset](https://www.kaggle.com/competitions/widsdatathon2020/data).

---

## License

MIT — see [LICENSE](LICENSE).

---

<sub>This is a research and portfolio project. It has not been validated for clinical use.</sub>
