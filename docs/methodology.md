# Methodology

Every stage below is implemented in the `qml_healthcare` package and driven by a fixed random seed
(`RANDOM_SEED = 42`), so the whole pipeline is deterministic.

## Dataset

The task is binary in-hospital mortality prediction on the **WiDS Datathon 2020** ICU dataset
(roughly 91,000 stays, 186 columns, about 8% mortality). When Kaggle credentials are present the real
data is downloaded; otherwise a schema-matched **synthetic fallback** is generated so anyone can run
the pipeline end to end offline.

The synthetic generator (`qml_healthcare.data.download.generate_synthetic_icu`) draws correlated,
realistically distributed vitals, labs, and Glasgow Coma Scale components, then assigns mortality
through a logistic link on a severity score, with about 3% missingness per numeric column. It does not
reproduce the real class balance: its positive rate is 22% (1,111 of 5,000 rows), not about 8%. The
numbers shown across this site were produced on that synthetic fallback; see
[Limitations](limitations.md).

A curated, interpretable subset of features is used (`config.NUMERIC_FEATURES`,
`config.CATEGORICAL_FEATURES`), including age, BMI, key APACHE vitals and labs, the three GCS
components, and the APACHE-IVa baseline risk score, plus gender, ethnicity, ICU type, and elective
surgery. The target is `hospital_death`.

## Preprocessing

The chain lives in `qml_healthcare.data.preprocess` and runs
`select_features -> clean -> make_splits -> scale -> top_k_features -> subsample_for_quantum`:

- **`clean`** drops rows with a missing target, imputes numeric columns with the median and
  categoricals with the mode, then one-hot encodes string categories (`drop_first=True`). It runs
  before the split, so the imputation medians and modes are computed on all rows, including the
  validation and test rows.
- **`make_splits`** produces a stratified 70 / 10 / 20 train / validation / test split.
- **`scale`** fits a `StandardScaler` on the training split only and applies it to all three, so the
  scaler statistics carry no information from the validation or test rows.
- **`top_k_features`** uses `SelectKBest` with the ANOVA F-statistic to choose the `k = 6` strongest
  features for the quantum encoding (six qubits).
- **`subsample_for_quantum`** draws a class-balanced subsample of `N = 200` training points, because
  the fidelity kernel is O(N squared) and does not scale on a CPU simulator.

## Classical baselines

Three scikit-learn models (`qml_healthcare.models.classical`) form the reference point:

- `SVC(kernel="rbf", probability=True)`
- `LogisticRegression(max_iter=1000)`
- `RandomForestClassifier(n_estimators=100)`

All use `random_state = 42`. They are trained on the full preprocessed feature set, and
`cross_validate_baselines` additionally reports stratified 5-fold cross-validation for ROC-AUC, F1,
and balanced accuracy.

## Quantum models

All quantum models run on a noiseless simulator (no hardware) at 6 qubits. The QSVM kernels are
exact: they use the default `ComputeUncompute` fidelity with Qiskit's reference `Sampler`, which
returns exact probabilities. The VQC and QNN use Qiskit's `StatevectorSampler`, which samples 1,024
shots per circuit from the exact statevector; the shot sampling and the initial weights are both
seeded from `RANDOM_SEED`. Feature maps and ansatzes use `reps = 2`, except the QNN feature map,
which uses `reps = 1`.

### Quantum SVM with three feature maps

`qml_healthcare.models.quantum_kernels.build_feature_map` builds three deliberately distinct encodings:

- **ZZFeatureMap** (`paulis = ["Z", "ZZ"]`)
- **Pauli Z+XX** (`paulis = ["Z", "XX"]`), genuinely different from the ZZ map
- a **custom** map: Hadamard, then `RZ(2x)` per qubit, then a linear chain of `CZ` gates on
  neighbouring qubits

Each map is wrapped in a `FidelityQuantumKernel` (with `enforce_psd=True`), which computes
`K(x, x') = |<phi(x) | phi(x')>| squared`. A standard `QSVC` is then trained on that kernel.

### Variational Quantum Classifier (VQC)

A `ZZFeatureMap` followed by a `RealAmplitudes` ansatz, optimized with COBYLA on a cross-entropy loss
through the `StatevectorSampler`. The per-iteration loss is logged for the training-curve figure.

### Quantum Neural Network (QNN)

A `PauliFeatureMap` composed with a `RealAmplitudes` ansatz, wrapped in a `SamplerQNN` and trained
through a `NeuralNetworkClassifier` with one-hot cross-entropy loss and COBYLA. Its interpret function
`x % 2` reads out qubit 0, not the parity of the bitstring: Qiskit orders bits little-endian, so
`x % 2` is qubit 0's bit, a Z measurement on that qubit.

## Metrics and uncertainty

`qml_healthcare.evaluation` computes accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and
PR-AUC. For each test-set metric, `bootstrap_metric_ci` draws 1000 seeded bootstrap resamples to form
a 95% percentile confidence interval (resamples with a single class are discarded, since AUC and F1
are undefined there).

Classical models also get 5-fold cross-validation; the quantum models do not, because refitting an
O(N squared) kernel per fold is prohibitive on a simulator. That asymmetry is intentional and is noted
in [Limitations](limitations.md).
