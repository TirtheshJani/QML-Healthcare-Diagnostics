# Findings

The benchmark trains three classical baselines and five quantum models on the same ICU mortality
task, using the schema-matched synthetic fallback rather than the real WiDS data, evaluates them on
held-out test rows, and attaches a 95% bootstrap confidence interval to every metric. The classical
models are scored on the full 1,000-row test split and the quantum models on a class-balanced 200-row
subsample of it, so compare ROC-AUC and balanced accuracy across the two groups, not accuracy or
PR-AUC (see [Limitations](limitations.md)). The result is a clean separation: the classical models
carry real signal, and the quantum models, at this scale, do not.

## Results

The table below is generated directly from `reports/results.json` by
`scripts/build_docs_tables.py`. No value here is typed by hand.

<!-- BEGIN_RESULTS_TABLE -->
| Model | Type | Accuracy | Balanced acc. | ROC-AUC [95% CI] | PR-AUC | F1 | Train (s) |
|-------|------|---------:|--------------:|:----------------|-------:|---:|----------:|
| Logistic Regression | classical | 0.810 | 0.651 | 0.817 [0.787, 0.845] | 0.578 | 0.460 | 0.01 |
| Random Forest | classical | 0.808 | 0.646 | 0.792 [0.758, 0.824] | 0.540 | 0.451 | 0.54 |
| SVM (RBF) | classical | 0.812 | 0.631 | 0.759 [0.721, 0.796] | 0.532 | 0.420 | 1.10 |
| VQC | quantum | 0.520 | 0.520 | 0.540 [0.457, 0.616] | 0.567 | 0.556 | 77.50 |
| QSVM (Pauli Z+XX) | quantum | 0.520 | 0.520 | 0.522 [0.440, 0.600] | 0.518 | 0.543 | 174.46 |
| QSVM (ZZFeatureMap) | quantum | 0.535 | 0.535 | 0.513 [0.434, 0.590] | 0.518 | 0.551 | 104.08 |
| QSVM (custom feature map) | quantum | 0.490 | 0.490 | 0.513 [0.437, 0.591] | 0.513 | 0.474 | 36.96 |
| QNN (SamplerQNN) | quantum | 0.500 | 0.500 | 0.506 [0.425, 0.583] | 0.506 | 0.510 | 74.03 |
<!-- END_RESULTS_TABLE -->

ROC-AUC is shown with its 95% bootstrap confidence interval. A model whose interval includes 0.5 is
statistically indistinguishable from a coin flip on this test set.

![ROC-AUC with 95% bootstrap confidence intervals for every model](assets/figures/roc_auc_ci_comparison.png){ loading=lazy }

## Five honest takeaways

1. **The quantum models are at chance.** Every quantum model's ROC-AUC confidence interval includes
   0.5. The strongest is the VQC at 0.540 [0.457, 0.616]; the QSVM variants land at 0.513 to 0.522
   and the QNN at 0.506 [0.425, 0.583]. None clears the bar for a learned signal.

2. **The classical baselines have stable, real signal.** Logistic Regression reaches ROC-AUC 0.817
   [0.787, 0.845], and 5-fold cross-validation agrees closely (0.810 plus or minus 0.013), so the
   single-split estimate is not a fluke. Random Forest (0.792) and SVM-RBF (0.759) behave similarly.

3. **The feature-map choice is within noise.** The three QSVM feature maps (ZZ, Pauli Z+XX, and the
   custom map) differ by about 0.01 ROC-AUC, well inside their confidence intervals. The fidelity
   kernel matrices are close to the identity at N=200, so there is little structure for the SVM to use.

4. **Runtime is the real bottleneck.** Classical models train in 0.006 to 1.1 seconds. The quantum
   models take 37 to 174 seconds on a statevector simulator, roughly 30 to 30,000 times slower,
   driven by the O(N squared) cost of the fidelity kernel.

5. **The engineering carries forward.** The result is a statement about this scale and this simulator,
   not about quantum machine learning in general. Liu, Arunachalam, and Temme (2021) identify regimes
   where quantum kernels are provably hard to simulate classically; the feature maps, fidelity kernel,
   PSD enforcement, and Qiskit primitives used here transfer directly to that setting.

## Confusion matrices and training curves

<div class="figure-grid">
  <figure>
    <img src="assets/figures/classical_roc.png" alt="Classical ROC curves" loading="lazy">
    <figcaption>Classical ROC curves</figcaption>
  </figure>
  <figure>
    <img src="assets/figures/qsvm_roc_overlay.png" alt="QSVM ROC curves by feature map" loading="lazy">
    <figcaption>QSVM ROC curves by feature map</figcaption>
  </figure>
  <figure>
    <img src="assets/figures/confusion_qsvm_zz.png" alt="QSVM (ZZ) confusion matrix" loading="lazy">
    <figcaption>QSVM (ZZ) confusion matrix</figcaption>
  </figure>
  <figure>
    <img src="assets/figures/confusion_vqc.png" alt="VQC confusion matrix" loading="lazy">
    <figcaption>VQC confusion matrix</figcaption>
  </figure>
  <figure>
    <img src="assets/figures/vqc_loss.png" alt="VQC training loss" loading="lazy">
    <figcaption>VQC training loss (COBYLA)</figcaption>
  </figure>
  <figure>
    <img src="assets/figures/qnn_loss.png" alt="QNN training loss" loading="lazy">
    <figcaption>QNN training loss (COBYLA)</figcaption>
  </figure>
  <figure>
    <img src="assets/figures/kernel_heatmap_zz.png" alt="ZZ fidelity kernel matrix" loading="lazy">
    <figcaption>ZZ fidelity kernel matrix</figcaption>
  </figure>
  <figure>
    <img src="assets/figures/runtime_comparison.png" alt="Training time by model" loading="lazy">
    <figcaption>Training time by model (log scale)</figcaption>
  </figure>
</div>

See [Methodology](methodology.md) for how each number is produced, or
[Limitations](limitations.md) for what this benchmark does and does not claim.
