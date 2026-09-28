# Live demo

This widget runs a **simplified classical model** entirely in your browser. There is no server: the
logistic regression's coefficients and the scaler statistics are exported to JSON, and the sigmoid is
recomputed in JavaScript as you move the sliders.

!!! warning "Read this first"
    This is a deliberately simplified demonstration model fit on six interpretable inputs. It is fit
    on the synthetic fallback data, not real patients. It is **not** the full benchmark model and is
    **not** for clinical use. Its own held-out ROC-AUC is shown below so you can judge it honestly.
    The quantum models are not used here; with the pipeline's encoding they sit at chance on this task
    (see [Findings](findings.md)).

<div class="qml-demo" data-model-url="assets/data/demo_model.json">
  <p class="demo-loading">Loading the demo model...</p>
</div>

## How it works

When you change an input, the browser computes, for each feature,

```text
z_i   = (input_i - scaler_mean_i) / scaler_scale_i
logit = intercept + sum_i ( coef_i * z_i )
p     = 1 / (1 + exp(-logit))
```

which is exactly what scikit-learn's `LogisticRegression` does on `StandardScaler`-transformed inputs.
`tests/test_demo_export.py` asserts this JavaScript formula reproduces the Python model's
`predict_proba` to within 1e-9, so what you see in the browser matches the trained model.

The model is produced by `scripts/export_demo_model.py`, which fits the compact logistic regression
on a held-out split, records its ROC-AUC, and writes everything to `docs/assets/data/demo_model.json`.
Because one of the six inputs is the APACHE-IVa baseline risk score (already a strong clinical
predictor), this small model scores close to the full benchmark model, which is expected rather than
a quantum effect.
