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
realistic, but it is a model of the data, not the data. Running on the real dataset can shift the
absolute numbers, though the classical-versus-quantum gap is expected to persist.

## Scale and simulation

- The fidelity quantum kernel is **O(N squared)**, which forces a small training subsample (N=200) and
  a small feature count (6 qubits). Larger N might change the picture, but it is not feasible on a CPU
  simulator.
- All quantum models run on an **exact statevector simulator** with no shot noise and no hardware
  effects. Real devices would add noise, not remove the scaling problem.
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
