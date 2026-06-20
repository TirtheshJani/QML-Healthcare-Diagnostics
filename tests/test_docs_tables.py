"""Tests for the docs-table generator.

These lock in the honesty contract: every number the docs show must come straight from
results.json, never a hand-typed literal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from qml_healthcare.evaluation import load_results
from qml_healthcare.reporting.tables import PRETTY, build_table, flatten, fmt_roc_auc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_docs_tables import build_results_data  # noqa: E402

# A tiny, self-contained results dict so the assertions do not depend on the live run.
SAMPLE = {
    "classical": {
        "logreg": {
            "accuracy": 0.81,
            "balanced_accuracy": 0.65,
            "roc_auc": 0.817,
            "roc_auc_ci_low": 0.787,
            "roc_auc_ci_high": 0.845,
            "pr_auc": 0.578,
            "f1": 0.46,
            "train_seconds": 0.006,
        }
    },
    "qsvm": {
        "qsvm_zz": {
            "accuracy": 0.535,
            "balanced_accuracy": 0.535,
            "roc_auc": 0.513,
            "roc_auc_ci_low": 0.434,
            "roc_auc_ci_high": 0.59,
            "pr_auc": 0.518,
            "f1": 0.551,
            "train_seconds": 104.08,
        }
    },
    "bonus": {
        "vqc": {
            "accuracy": 0.52,
            "balanced_accuracy": 0.52,
            "roc_auc": 0.54,
            "roc_auc_ci_low": 0.457,
            "roc_auc_ci_high": 0.616,
            "pr_auc": 0.567,
            "f1": 0.556,
            "train_seconds": 77.5,
        }
    },
}


def test_build_table_lists_every_model() -> None:
    table = build_table(SAMPLE)
    for key in flatten(SAMPLE):
        assert PRETTY[key] in table


def test_table_cell_is_sourced_not_fabricated() -> None:
    """The ROC-AUC cell text must equal what fmt_roc_auc derives from the dict."""
    table = build_table(SAMPLE)
    expected = fmt_roc_auc(SAMPLE["classical"]["logreg"])  # "0.817 [0.787, 0.845]"
    assert expected in table


def test_table_orders_by_roc_auc_descending() -> None:
    table = build_table(SAMPLE)
    pos = [table.index(PRETTY[k]) for k in ("logreg", "vqc", "qsvm_zz")]
    assert pos == sorted(pos)


def test_results_data_matches_flatten() -> None:
    data = build_results_data(SAMPLE)
    flat = flatten(SAMPLE)
    assert {m["key"] for m in data["models"]} == set(flat)
    for entry in data["models"]:
        src = flat[entry["key"]]
        assert entry["roc_auc"] == src["roc_auc"]
        assert entry["type"] in {"classical", "quantum"}


def test_live_results_json_builds_without_fabrication() -> None:
    """Smoke test against the committed results.json: it builds and lists its models."""
    results_path = Path(__file__).resolve().parents[1] / "reports" / "results.json"
    if not results_path.exists():
        pytest.skip("reports/results.json not present")
    results = load_results()
    table = build_table(results)
    for key in flatten(results):
        assert PRETTY.get(key, key) in table
