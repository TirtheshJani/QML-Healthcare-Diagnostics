"""Generate the docs-site results artifacts from reports/results.json.

Writes two kinds of output, both derived entirely from the metrics dump so the site
never carries a hand-typed number:

  1. The markdown results table, injected between the BEGIN_RESULTS_TABLE /
     END_RESULTS_TABLE markers in docs/findings.md (same pattern as the README updater).
  2. docs/assets/data/results_data.json - a flat, chart-ready view of every model, read
     by docs/assets/js/charts.js to draw the comparison charts and fill the home-page cards.

Run it after every pipeline run (or via scripts/sync_docs_assets.py).
"""

from __future__ import annotations

import json
import re

from qml_healthcare.config import PROJECT_ROOT
from qml_healthcare.evaluation import load_results
from qml_healthcare.reporting.tables import KIND, PRETTY, build_table, flatten

DOCS_DIR = PROJECT_ROOT / "docs"
FINDINGS_MD = DOCS_DIR / "findings.md"
RESULTS_DATA = DOCS_DIR / "assets" / "data" / "results_data.json"

_CHART_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "roc_auc_ci_low",
    "roc_auc_ci_high",
    "pr_auc",
    "f1",
    "train_seconds",
)


def build_results_data(results: dict) -> dict:
    """Flatten results.json into a chart-ready payload, ordered by ROC-AUC descending."""
    flat = flatten(results)
    ordered = sorted(flat.items(), key=lambda kv: -kv[1].get("roc_auc", 0.0))
    models = []
    for key, metrics in ordered:
        entry = {
            "key": key,
            "label": PRETTY.get(key, key),
            "type": KIND.get(key, "unknown"),
        }
        for m in _CHART_METRICS:
            if m in metrics:
                entry[m] = metrics[m]
        models.append(entry)
    return {"generated_from": "reports/results.json", "models": models}


def _inject(md_text: str, marker: str, table: str) -> str:
    """Replace the content between BEGIN_<marker> / END_<marker> comments."""
    pattern = re.compile(
        rf"<!-- BEGIN_{marker} -->.*?<!-- END_{marker} -->",
        re.DOTALL,
    )
    if not pattern.search(md_text):
        raise RuntimeError(f"docs page is missing BEGIN_{marker} / END_{marker} markers.")
    block = f"<!-- BEGIN_{marker} -->\n{table}\n<!-- END_{marker} -->"
    return pattern.sub(block, md_text)


def write_docs_tables() -> None:
    """Refresh docs/findings.md table and docs/assets/data/results_data.json."""
    results = load_results()

    text = FINDINGS_MD.read_text(encoding="utf-8")
    text = _inject(text, "RESULTS_TABLE", build_table(results))
    FINDINGS_MD.write_text(text, encoding="utf-8")

    RESULTS_DATA.parent.mkdir(parents=True, exist_ok=True)
    payload = build_results_data(results)
    RESULTS_DATA.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Updated {FINDINGS_MD} and {RESULTS_DATA} ({len(payload['models'])} models).")


def main() -> None:
    write_docs_tables()


if __name__ == "__main__":
    main()
