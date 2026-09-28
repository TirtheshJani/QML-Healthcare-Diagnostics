/* Build the dashboard charts and fill the KPI cards from results_data.json.
   Every value comes from reports/results.json via scripts/build_docs_tables.py. */

(function () {
  "use strict";

  var CLASSICAL = "#5c6bc0";
  var QUANTUM = "#fb8c00";

  function colorFor(type) {
    return type === "quantum" ? QUANTUM : CLASSICAL;
  }

  function setKpi(name, value) {
    var nodes = document.querySelectorAll('[data-kpi="' + name + '"]');
    nodes.forEach(function (el) {
      el.textContent = value;
    });
  }

  function fillKpis(models) {
    var classical = models.filter(function (m) {
      return m.type === "classical" && typeof m.roc_auc === "number";
    });
    var quantum = models.filter(function (m) {
      return m.type === "quantum" && typeof m.roc_auc === "number";
    });
    if (classical.length) {
      var bestC = classical.reduce(function (a, b) {
        return b.roc_auc > a.roc_auc ? b : a;
      });
      setKpi("best-classical-auc", bestC.roc_auc.toFixed(3));
      setKpi("best-classical-label", bestC.label);
    }
    if (quantum.length) {
      var bestQ = quantum.reduce(function (a, b) {
        return b.roc_auc > a.roc_auc ? b : a;
      });
      setKpi("best-quantum-auc", bestQ.roc_auc.toFixed(3));
      setKpi("best-quantum-label", bestQ.label + ", pipeline encoding");
    }
    setKpi("model-count", String(models.length));
  }

  // Dashed vertical line at ROC-AUC = 0.5 (chance).
  var chanceLine = {
    id: "chanceLine",
    afterDraw: function (chart) {
      if (!chart.scales.x) return;
      var x = chart.scales.x.getPixelForValue(0.5);
      var area = chart.chartArea;
      var ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = "rgba(128,128,128,0.9)";
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, area.top);
      ctx.lineTo(x, area.bottom);
      ctx.stroke();
      ctx.fillStyle = "rgba(128,128,128,0.95)";
      ctx.setLineDash([]);
      ctx.font = "12px sans-serif";
      ctx.fillText("chance (0.5)", x + 4, area.top + 12);
      ctx.restore();
    },
  };

  function drawRocAuc(models) {
    var canvas = document.getElementById("chart-roc-auc");
    if (!canvas || typeof Chart === "undefined") return;
    var withCi = models.filter(function (m) {
      return typeof m.roc_auc_ci_low === "number" && typeof m.roc_auc_ci_high === "number";
    });
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: withCi.map(function (m) {
          return m.label;
        }),
        datasets: [
          {
            label: "95% bootstrap CI for ROC-AUC",
            data: withCi.map(function (m) {
              return [m.roc_auc_ci_low, m.roc_auc_ci_high];
            }),
            backgroundColor: withCi.map(function (m) {
              return colorFor(m.type);
            }),
            borderWidth: 0,
            borderSkipped: false,
            barPercentage: 0.7,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: "ROC-AUC with 95% bootstrap confidence interval" },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var v = ctx.raw;
                return "CI: " + v[0].toFixed(3) + " to " + v[1].toFixed(3);
              },
            },
          },
        },
        scales: {
          x: { min: 0.3, max: 0.9, title: { display: true, text: "ROC-AUC" } },
        },
      },
      plugins: [chanceLine],
    });
  }

  function drawRuntime(models) {
    var canvas = document.getElementById("chart-runtime");
    if (!canvas || typeof Chart === "undefined") return;
    var withTime = models.filter(function (m) {
      return typeof m.train_seconds === "number" && m.train_seconds > 0;
    });
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: withTime.map(function (m) {
          return m.label;
        }),
        datasets: [
          {
            label: "Training time (s)",
            data: withTime.map(function (m) {
              return m.train_seconds;
            }),
            backgroundColor: withTime.map(function (m) {
              return colorFor(m.type);
            }),
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: "Training time by model (log scale, seconds)" },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.raw.toFixed(3) + " s";
              },
            },
          },
        },
        scales: {
          y: { type: "logarithmic", title: { display: true, text: "seconds" } },
          x: { ticks: { maxRotation: 40, minRotation: 30 } },
        },
      },
    });
  }

  async function init() {
    var el = document.querySelector("[data-results-url]");
    if (!el) return;
    var url = el.getAttribute("data-results-url");
    var data;
    try {
      var resp = await fetch(url);
      data = await resp.json();
    } catch (e) {
      console.error("Could not load results data:", e);
      return;
    }
    var models = data.models || [];
    fillKpis(models);
    drawRocAuc(models);
    drawRuntime(models);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
