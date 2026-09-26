# Limitations

This project is built to be honest about its scope. The headline negative result is real, but it is a
result about a specific setting, not a general claim.

## What the benchmark does not claim

- **It does not show quantum machine learning fails.** It shows that, with the pipeline's unscaled
  angle encoding, the quantum models do not beat chance on this task. For the QSVMs a pre-specified
  [bandwidth ablation](findings.md#kernel-bandwidth-ablation) traces that to the encoding: the
  StandardScaler z-scores go in as rotation angles, the kernel sits at the random-state fidelity, and
  with the input scale picked on validation data the QSVMs reach test ROC-AUC 0.70 to 0.80 on the same
  200 rows. The chance-level result is about this encoding, not about quantum methods in general.
- **It is not a clinical study.** The models are trained and evaluated on competition or synthetic
  data, with no external validation, calibration, or fairness analysis. Nothing here is fit for
  clinical use.

## Data

The numbers on this site were produced on the **synthetic fallback**, not the real WiDS data, because
the public site is built without Kaggle credentials. The synthetic generator is schema-matched and
realistic, but it is a model of the data, not the data. Its positive rate (22%) is also well above the
real data's (about 8%). Running on the real dataset can shift the absolute numbers, and whether the
classical-versus-quantum gap persists there has not been tested.

The synthetic label also favours logistic regression by construction. `generate_synthetic_icu` draws
each outcome from sigmoid(severity - 2.7 + noise), where severity is a weighted sum of 13 inputs
(linear in all but temperature, which enters as |temp - 37|) and the noise is Gaussian with standard
deviation 0.5. The APACHE hospital-death-probability column is sigmoid(severity - 2.5) plus Gaussian
noise (standard deviation 0.04). So logistic regression is close to correctly specified, and the
APACHE column alone, with no model, scores ROC-AUC 0.814 [0.784, 0.844] on the full test split, next
to logistic regression's 0.817 [0.787, 0.845].

## Scale and simulation

- The pipeline's `FidelityQuantumKernel` runs one ComputeUncompute circuit per kernel entry, so a
  QSVM fit needs **O(N squared)** circuit runs. That is what dominates its runtime and why it uses a
  200-row subsample. It is not a limit of simulating 6 qubits: an exact statevector kernel simulates
  each point once, and the exact training and test kernels plus the SVC fit take 0.9 to 5.6 seconds
  per feature map. Full-N and cross-validated quantum runs were not done here.
- The quantum inputs are **unscaled z-scores used as rotation angles**, which leaves the fidelity
  kernels at the random-state value (off-diagonal mean 0.0160 to 0.0194 against 1/2^6 = 0.0156). The
  bandwidth ablation picks the scale on the validation split, but for ZZ and Pauli the chosen value is
  the smallest on the grid, so the best scale may be smaller still. The VQC and QNN use the same
  encoding and were not rerun with a rescaled one.
- The **VQC and QNN are under-trained.** Each has 18 trainable weights (`RealAmplitudes(6, reps=2)`)
  and gets 60 COBYLA loss evaluations. qiskit-machine-learning's cross-entropy uses log base 2, so a
  constant 0.5 prediction costs 1.0 bit; the committed runs end at 0.955 bits (VQC) and 0.950 bits
  (QNN).
- All quantum models run on a **noiseless statevector simulator** with no hardware effects. The QSVM
  kernels are exact; the VQC and QNN add 1,024-shot sampling noise (seeded). Real devices would add
  noise, not remove the scaling problem.
- The two model families are **not trained or scored on the same rows**. The classical models use the
  full 3,500-row training split with all 29 input columns and are scored on the full 1,000-row test
  split (22.2% positive). The quantum models use a class-balanced 200-row training subsample with the
  top 6 features and are scored on a class-balanced 200-row subsample of the same test split. Accuracy
  and PR-AUC therefore have different chance levels in the two groups (about 0.78 and 0.22 for the
  classical rows, 0.5 for the quantum rows), so compare ROC-AUC and balanced accuracy across families.
  For a like-for-like check, the bandwidth ablation refits the classical baselines on the quantum
  split with default hyperparameters: logistic regression scores 0.794 [0.731, 0.851] there. A
  post-hoc RBF SVM with its bandwidth tuned on validation, the way each QSVM's input scale was,
  scores 0.810 [0.747, 0.865].
- The quantum models are evaluated with **bootstrap confidence intervals only**, not cross-validation;
  with the pipeline's ComputeUncompute kernel each refit takes minutes. The classical models do get
  5-fold cross-validation, so the two families are not measured identically. The bootstrap intervals
  cover test-set sampling only, not seed or optimizer variation, which is large for the VQC and QNN:
  reseeding moved VQC ROC-AUC from 0.540 to 0.437.

## The live demo

The [live demo](demo.md) is a **simplified model**, not the benchmark model:

- It is a logistic regression on six interpretable raw inputs, retrained specifically so the
  coefficients map to labeled sliders.
- One of those inputs is the APACHE-IVa baseline risk score, an existing clinical predictor, so the
  demo scores well for reasons that have nothing to do with quantum computing.
- Its held-out ROC-AUC is reported in the widget itself. Treat it as an illustration of client-side
  inference, not as a validated risk calculator.

## Where the result could change

Liu, Arunachalam, and Temme (2021) prove there exist learning problems where quantum kernels offer a
provable advantage and are hard to simulate classically. ICU mortality on tabular features is not
known to be such a problem. The value of this repository is the audited, reproducible pipeline: the
feature maps, fidelity kernel, PSD enforcement, and Qiskit primitives transfer directly to a setting
where a quantum advantage is plausible.
