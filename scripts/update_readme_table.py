"""Replace the BEGIN_RESULTS_TABLE / END_RESULTS_TABLE block in README.md
with a markdown table built from reports/results.json.

Idempotent - run it after every pipeline run to keep the README honest. The table
itself is rendered by ``qml_healthcare.reporting.tables`` so the README and the docs
site share one source of truth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qml_healthcare.reporting.tables import build_table, flatten

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "reports" / "results.json"


def main() -> None:
    if not RESULTS.exists():
        raise FileNotFoundError(f"{RESULTS} not found - run the pipeline first.")
    with RESULTS.open(encoding="utf-8") as f:
        results = json.load(f)
    table = build_table(results)

    text = README.read_text(encoding="utf-8")
    pattern = re.compile(r"<!-- BEGIN_RESULTS_TABLE -->.*?<!-- END_RESULTS_TABLE -->", re.DOTALL)
    new_block = f"<!-- BEGIN_RESULTS_TABLE -->\n{table}\n<!-- END_RESULTS_TABLE -->"
    if not pattern.search(text):
        raise RuntimeError("README is missing BEGIN_RESULTS_TABLE / END_RESULTS_TABLE markers.")
    new_text = pattern.sub(new_block, text)
    README.write_text(new_text, encoding="utf-8")
    print(f"Updated {README} with {len(flatten(results))} model rows.")


if __name__ == "__main__":
    main()
