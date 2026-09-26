# Findings

The benchmark trains three classical baselines and five quantum models on the same ICU mortality
task, using the schema-matched synthetic fallback rather than the real WiDS data, evaluates them on
held-out test rows, and attaches a 95% bootstrap confidence interval to every metric. The classical
models are scored on the full 1,000-row test split and the quantum models on a class-balanced 200-row
subsample of it, so compare ROC-AUC and balanced accuracy across the two groups, not accuracy or
PR-AUC (see [Limitations](limitations.md)). With the pipeline's encoding the classical models carry
real signal and the quantum models do not. A pre-specified
[kernel bandwidth ablation](#kernel-bandwidth-ablation) shows that, for the QSVMs, this comes from
the unscaled angle encoding rather than from the sample size.

## Results

The table below is generated directly from `reports/results.json` by
`scripts/build_docs_tables.py`. No value here is typed by hand.

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

ROC-AUC is shown with its 95% bootstrap confidence interval. A model whose interval includes 0.5 is
statistically indistinguishable from a coin flip on this test set.

![ROC-AUC with 95% bootstrap confidence intervals for every model](assets/figures/roc_auc_ci_comparison.png){ loading=lazy }

## Five honest takeaways

1. **With the pipeline's encoding, the quantum models are at chance.** Every quantum model's ROC-AUC
   confidence interval includes 0.5. The QSVM variants land at 0.513 to 0.522, the QNN at
   0.484 [0.399, 0.559], and the VQC at 0.437 [0.353, 0.518]. For the QSVMs the cause is the unscaled
   angle encoding, which leaves the kernel at the random-state value; with the input scale picked on
   validation data they reach 0.70 to 0.80 (see below). The VQC and QNN are also under-trained: 18
   weights, 60 COBYLA loss evaluations, and a final training loss of 0.955 and 0.950 bits against 1.0
   bit for a constant prediction. Before they were seeded, an earlier committed run gave them 0.540 and
   0.506; the bootstrap intervals cover test-set sampling only, not that seed variation.

2. **The classical baselines have stable, real signal.** Logistic Regression reaches ROC-AUC 0.817
   [0.787, 0.845], and 5-fold cross-validation agrees closely (0.810 plus or minus 0.013), so the
   single-split estimate is not a fluke. Random Forest (0.792) and SVM-RBF (0.759) behave similarly.
   The synthetic label is close to a logistic model by construction, which favours logistic
   regression (see [Limitations](limitations.md#data)).

3. **The feature-map choice cannot show up at the pipeline's encoding.** The three QSVM feature maps
   (ZZ, Pauli Z+XX, and the custom map) differ by about 0.01 ROC-AUC, well inside their confidence
   intervals. The fidelity kernel matrices are close to the identity, so there is little structure for
   the SVM to use. After bandwidth selection the point estimates spread out (custom 0.798, Pauli
   0.728, ZZ 0.701), but neighbouring intervals overlap.

4. **Runtime reflects the kernel implementation.** Classical models train in 0.006 to 1.1 seconds.
   The quantum models take 37 to 174 seconds on a statevector simulator, roughly 30 to 30,000 times
   slower, driven for the QSVMs by the O(N squared) ComputeUncompute circuits of the fidelity kernel,
   one per kernel entry. That factor is specific to this implementation: an exact statevector kernel
   simulates each point once, and the exact training and test kernels plus the SVC fit take 0.9 to 5.6
   seconds per feature map.

5. **The engineering carries forward.** The chance-level result is a statement about the pipeline's
   unscaled encoding on this synthetic benchmark, not about quantum machine learning in general. Liu,
   Arunachalam, and Temme (2021) identify regimes where quantum kernels are provably hard to simulate
   classically; the feature maps, fidelity kernel, PSD enforcement, and Qiskit primitives used here
   transfer directly to that setting.

## Kernel bandwidth ablation

The pipeline feeds `StandardScaler` z-scores (from -2.85 to 3.18 on the quantum training rows)
straight into the rotation angles. At that scale the mean off-diagonal entry of the 200 x 200
training kernel is 0.0160 to 0.0194, against 1/2^6 = 0.0156 for two random 6-qubit states: the
exponential concentration described by Thanasilp et al. (Nature Communications 15, 2024,
doi:10.1038/s41467-024-49287-w). The input scale is the kernel bandwidth studied by Shaydulin and Wild
(arXiv:2111.05451).

`scripts/ablate_kernel_bandwidth.py` was added after an audit found this concentration, and its design
was committed before it was run. It multiplies the inputs by s in {0.05, 0.1, 0.2, 0.5, 1}, picks s
for each feature map by ROC-AUC on the 500-row validation split (unused by the pipeline), and scores
the test rows only at s = 1 and at the chosen s, with the same bootstrap as the table above. Its exact
statevector kernels reproduce the committed QSVM rows at s = 1 exactly. All rows below use the same
200 training rows, 6 features and 200 test rows.

| Model | s | Kernel off-diag. mean | Test ROC-AUC [95% CI] |
|---|---:|---:|:---|
| QSVM ZZ, pipeline encoding | 1 | 0.0163 | 0.513 [0.434, 0.590] |
| QSVM ZZ, s chosen on validation | 0.05 | 0.2399 | 0.701 [0.632, 0.769] |
| QSVM Pauli Z+XX, pipeline encoding | 1 | 0.0160 | 0.522 [0.440, 0.600] |
| QSVM Pauli Z+XX, s chosen on validation | 0.05 | 0.3600 | 0.728 [0.660, 0.801] |
| QSVM custom, pipeline encoding | 1 | 0.0194 | 0.513 [0.437, 0.591] |
| QSVM custom, s chosen on validation | 0.1 | 0.7755 | 0.798 [0.737, 0.854] |
| Logistic regression | | | 0.794 [0.731, 0.851] |
| Random forest | | | 0.743 [0.673, 0.808] |
| SVM (RBF) | | | 0.708 [0.635, 0.781] |
| APACHE probability column alone (no model) | | | 0.812 [0.750, 0.869] |

![Kernel concentration, validation ROC-AUC by input scale, and test ROC-AUC with matched classical controls](assets/figures/bandwidth_ablation.png){ loading=lazy }

With the scale chosen on validation data every QSVM is clear of chance, and the best one (custom,
s = 0.1) is level with logistic regression on the same rows. No model's point estimate beats the
APACHE column alone, which is one of the six encoded features. For ZZ and Pauli the chosen s = 0.05 is
the smallest value on the grid, so a smaller s might score higher; the grid was not extended after
seeing the results. The VQC and QNN use the same encoding and were not rerun. The numbers are in
`reports/bandwidth_ablation.json`.

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
    <figcaption>Training time by model (seconds)</figcaption>
  </figure>
</div>

See [Methodology](methodology.md) for how each number is produced, or
[Limitations](limitations.md) for what this benchmark does and does not claim.
