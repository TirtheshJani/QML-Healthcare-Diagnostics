.PHONY: help setup data baseline qsvm bonus reports notebooks all test lint format clean

PYTHON ?= python

help:
	@echo "Available targets:"
	@echo "  setup       Editable-install the package + dev extras + pre-commit hook"
	@echo "  data        Download (or synthesize) the WiDS ICU dataset"
	@echo "  baseline    Train classical baselines (SVM, LogReg, RF)"
	@echo "  qsvm        Train QSVMs across feature maps (zz, pauli, custom)"
	@echo "  bonus       Train VQC + QNN bonus models"
	@echo "  reports     Build final comparison plots from results.json"
	@echo "  notebooks   Execute notebooks/01..06 with nbconvert"
	@echo "  all         setup → data → baseline → qsvm → bonus → reports"
	@echo "  test        Run pytest"
	@echo "  lint        Ruff + Black --check"
	@echo "  format      Ruff --fix + Black"
	@echo "  clean       Remove caches and processed data"

setup:
	$(PYTHON) -m pip install -e ".[dev]"
	-pre-commit install || true

data:
	$(PYTHON) scripts/download_data.py

baseline:
	$(PYTHON) scripts/train_baseline.py

qsvm:
	$(PYTHON) scripts/train_qsvm.py

bonus:
	$(PYTHON) scripts/train_vqc_qnn.py

reports:
	$(PYTHON) -c "from qml_healthcare.pipeline import run_reports; run_reports()"
	$(PYTHON) scripts/update_readme_table.py

notebooks:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

all: data baseline qsvm bonus reports

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m black .

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/processed/*.parquet
