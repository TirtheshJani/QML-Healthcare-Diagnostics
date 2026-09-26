# Limitations

This project is built to be honest about its scope. The headline negative result is real, but it is a
result about a specific setting, not a general claim.

## What the benchmark does not claim

- **It does not show quantum machine learning fails.** It shows that, at N=200 on a statevector
  simulator with these feature maps, the quantum models do not beat chance on this task. That is a
  statement about scale and setup, not about quantum methods in general.
- **It is not a clinical study.** The models are trained and evaluated on competition or synthetic
  data, with no external validation, calibration, or fairness analysis. Nothing here is fit for
  clinical use.

## Data

The numbers on this site were produced on the **synthetic fallback**, not the real WiDS data, because
the public site is built without Kaggle credentials. The synthetic generator is schema-matched and
realistic, but it is a model of the data, not the data. Its positive rate (22%) is also well above the
real data's (about 8%). Running on the real dataset can shift the absolute numbers, and whether the
classical-versus-quantum gap persists there has not been tested.

## Scale and simulation

- The fidelity quantum kernel is **O(N squared)**, which forces a small training subsample (N=200) and
  a small feature count (6 qubits). Larger N might change the picture, but it is not feasible on a CPU
  simulator.
- All quantum models run on a **noiseless statevector simulator** with no hardware effects. The QSVM
  kernels are exact; the VQC and QNN add 1,024-shot sampling noise (seeded). Real devices would add
  noise, not remove the scaling problem.
- The two model families are **not trained or scored on the same rows**. The classical models use the
  full 3,500-row training split with all 29 input columns and are scored on the full 1,000-row test
  split (22.2% positive). The quantum models use a class-balanced 200-row training subsample with the
  top 6 features and are scored on a class-balanced 200-row subsample of the same test split. Accuracy
  and PR-AUC therefore have different chance levels in the two groups (about 0.78 and 0.22 for the
  classical rows, 0.5 for the quantum rows), so compare ROC-AUC and balanced accuracy across families.
- The quantum models are evaluated with **bootstrap confidence intervals only**, not cross-validation,
  because refitting an O(N squared) kernel per fold is prohibitive. The classical models do get 5-fold
  cross-validation, so the two families are not measured identically. The bootstrap intervals are wide
  enough that this asymmetry does not change the conclusion.

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
