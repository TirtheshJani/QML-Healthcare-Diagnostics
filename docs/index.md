---
hide:
  - navigation
  - toc
---

<section class="hero" markdown>

# Quantum ML for ICU Mortality Prediction

A reproducible benchmark of **Quantum SVMs, a VQC, and a QNN** against classical baselines on the
WiDS Datathon 2020 ICU dataset. Built on Qiskit, fully seeded, and honest about what it found.

[View the findings](findings.md){ .md-button .md-button--primary }
[Try the live demo](demo.md){ .md-button }
[Source on GitHub](https://github.com/TirtheshJani/QML-Healthcare-Diagnostics){ .md-button }

</section>

<div class="callout-headline" markdown>
**Headline:** at this scale, the classical models win cleanly and the quantum models sit at chance.
Logistic Regression reaches ROC-AUC **0.817** [0.787, 0.845]; every quantum model's 95% confidence
interval includes 0.5, while running two to four orders of magnitude slower. That negative result is
the point: it is what a careful, leakage-free audit actually shows.
</div>

## At a glance

<div class="kpi-grid">
  <div class="kpi-card kpi-classical">
    <div class="kpi-value" data-kpi="best-classical-auc">--</div>
    <div class="kpi-label">Best classical ROC-AUC</div>
    <div class="kpi-sub" data-kpi="best-classical-label">&nbsp;</div>
  </div>
  <div class="kpi-card kpi-quantum">
    <div class="kpi-value" data-kpi="best-quantum-auc">--</div>
    <div class="kpi-label">Best quantum ROC-AUC</div>
    <div class="kpi-sub" data-kpi="best-quantum-label">&nbsp;</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value" data-kpi="model-count">--</div>
    <div class="kpi-label">Models benchmarked</div>
    <div class="kpi-sub">classical + quantum</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value" data-kpi="slowdown">--</div>
    <div class="kpi-label">Quantum slowdown</div>
    <div class="kpi-sub">slowest quantum vs fastest classical</div>
  </div>
</div>

## The one chart that tells the story

Each bar spans the 95% bootstrap confidence interval for ROC-AUC. The dashed line marks 0.5, the
score of a coin flip. Classical bars sit clearly to the right of chance; every quantum bar crosses it.

<div class="chart-card">
  <canvas id="chart-roc-auc" height="220" data-results-url="assets/data/results_data.json"></canvas>
</div>

<div class="chart-card">
  <canvas id="chart-runtime" height="200" data-results-url="assets/data/results_data.json"></canvas>
</div>

## Explore

<div class="nav-grid">
  <a class="nav-card" href="findings.html">
    <h3>Findings</h3>
    <p>The full results table, confidence intervals, and the five honest takeaways.</p>
  </a>
  <a class="nav-card" href="methodology.html">
    <h3>Methodology</h3>
    <p>Dataset, preprocessing, feature maps, quantum circuits, and how uncertainty is estimated.</p>
  </a>
  <a class="nav-card" href="architecture.html">
    <h3>Architecture</h3>
    <p>How the package is laid out and how data flows from raw CSV to results and figures.</p>
  </a>
  <a class="nav-card" href="demo.html">
    <h3>Live demo</h3>
    <p>Enter ICU values and get an instant risk estimate from a simplified classical model.</p>
  </a>
  <a class="nav-card" href="reproducibility.html">
    <h3>Reproducibility</h3>
    <p>One command rebuilds every number, figure, and table from a fixed seed.</p>
  </a>
  <a class="nav-card" href="limitations.html">
    <h3>Limitations</h3>
    <p>What this benchmark does not claim, and where the quantum result could change.</p>
  </a>
</div>

!!! note "Not a clinical tool"
    This is a research and portfolio project on synthetic or de-identified competition data. Nothing
    here is validated for clinical use.
