/* Client-side ICU risk demo. Loads demo_model.json and recomputes the logistic-regression
   sigmoid in the browser. This mirrors scripts/export_demo_model.py exactly; the round-trip is
   asserted in tests/test_demo_export.py. No server is involved. */

(function () {
  "use strict";

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  // p = sigmoid(intercept + sum_i coef_i * (x_i - mean_i) / scale_i)
  function predict(model, values) {
    var logit = model.intercept;
    for (var i = 0; i < model.coef.length; i++) {
      var z = (values[i] - model.scaler_mean[i]) / model.scaler_scale[i];
      logit += model.coef[i] * z;
    }
    return sigmoid(logit);
  }

  function band(p) {
    if (p < 0.1) return { text: "Lower estimated risk", cls: "band-low" };
    if (p < 0.3) return { text: "Moderate estimated risk", cls: "band-moderate" };
    return { text: "Elevated estimated risk", cls: "band-elevated" };
  }

  function fieldHtml(feature, index) {
    var step = feature.step != null ? feature.step : 1;
    return (
      '<div class="demo-field">' +
      '<label for="demo-f' +
      index +
      '"><span>' +
      feature.label +
      '</span><span class="demo-readout" id="demo-r' +
      index +
      '"></span></label>' +
      '<span class="demo-unit">' +
      feature.unit +
      "</span>" +
      '<input type="range" id="demo-f' +
      index +
      '" min="' +
      feature.min +
      '" max="' +
      feature.max +
      '" step="' +
      step +
      '" value="' +
      feature.default +
      '">' +
      "</div>"
    );
  }

  function render(root, model) {
    var aucText =
      typeof model.demo_test_roc_auc === "number"
        ? model.demo_test_roc_auc.toFixed(3)
        : "n/a";
    var fields = model.features.map(fieldHtml).join("");

    root.innerHTML =
      '<div class="demo-banner">' +
      (model.note || "Simplified demonstration model.") +
      "</div>" +
      '<div class="demo-grid">' +
      "<div>" +
      fields +
      "</div>" +
      '<div class="demo-result">' +
      '<div class="demo-prob" id="demo-prob">--</div>' +
      '<div class="demo-band" id="demo-band">&nbsp;</div>' +
      '<div class="demo-bar"><span id="demo-fill"></span></div>' +
      "<div>" +
      (model.positive_class_label || "Positive class") +
      "</div>" +
      '<div class="demo-auc">Demo model held-out ROC-AUC: ' +
      aucText +
      ". Not for clinical use.</div>" +
      "</div>" +
      "</div>";

    var inputs = model.features.map(function (_f, i) {
      return root.querySelector("#demo-f" + i);
    });
    var readouts = model.features.map(function (_f, i) {
      return root.querySelector("#demo-r" + i);
    });
    var probEl = root.querySelector("#demo-prob");
    var bandEl = root.querySelector("#demo-band");
    var fillEl = root.querySelector("#demo-fill");

    function update() {
      var values = inputs.map(function (inp, i) {
        var v = parseFloat(inp.value);
        readouts[i].textContent = v;
        return v;
      });
      var p = predict(model, values);
      var pct = (p * 100).toFixed(1) + "%";
      probEl.textContent = pct;
      var b = band(p);
      bandEl.textContent = b.text;
      bandEl.className = "demo-band " + b.cls;
      fillEl.style.width = Math.min(100, p * 100) + "%";
    }

    inputs.forEach(function (inp) {
      inp.addEventListener("input", update);
    });
    update();
  }

  async function initOne(root) {
    var url = root.getAttribute("data-model-url") || "assets/data/demo_model.json";
    try {
      var resp = await fetch(url);
      var model = await resp.json();
      render(root, model);
    } catch (e) {
      console.error("Could not load demo model:", e);
      root.innerHTML = "<p>Could not load the demo model.</p>";
    }
  }

  function init() {
    document.querySelectorAll(".qml-demo").forEach(initOne);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
