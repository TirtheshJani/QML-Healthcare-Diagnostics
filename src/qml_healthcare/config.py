"""Project paths and shared constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_PATH: Path = REPORTS_DIR / "results.json"

RANDOM_SEED: int = 42
DEFAULT_QUBITS: int = 6
DEFAULT_REPS: int = 2
DEFAULT_QUANTUM_SUBSAMPLE: int = 200

# Curated WiDS feature subset (used by both real and synthetic paths)
NUMERIC_FEATURES: list[str] = [
    "age",
    "bmi",
    "pre_icu_los_days",
    "heart_rate_apache",
    "map_apache",
    "temp_apache",
    "resprate_apache",
    "wbc_apache",
    "creatinine_apache",
    "bun_apache",
    "glucose_apache",
    "hemaglobin_apache",
    "sodium_apache",
    "gcs_eyes_apache",
    "gcs_motor_apache",
    "gcs_verbal_apache",
    "apache_4a_hospital_death_prob",
]
CATEGORICAL_FEATURES: list[str] = ["gender", "ethnicity", "icu_type", "elective_surgery"]
TARGET: str = "hospital_death"


def ensure_dirs() -> None:
    """Create writable directories that don't ship in the repo."""
    for d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
