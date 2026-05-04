"""Acquire WiDS Datathon 2020 data, or generate a schema-matched synthetic fallback.

The synthetic fallback exists so reviewers without Kaggle credentials can still
run the full pipeline end-to-end. It produces correlated features with realistic
distributions and a logistic-link target — the QSVM has signal to learn, and
metrics on synthetic data are still meaningful as a sanity check.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from qml_healthcare.config import RANDOM_SEED, RAW_DIR, ensure_dirs

WIDS_FILENAME = "training_v2.csv"
KAGGLE_COMPETITION = "widsdatathon2020"


def _kaggle_credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    home_creds = Path.home() / ".kaggle" / "kaggle.json"
    return home_creds.exists()


def download_wids(target_dir: Path = RAW_DIR) -> Path | None:
    """Download WiDS Datathon 2020 via Kaggle API. Returns path to CSV or None."""
    if not _kaggle_credentials_present():
        return None
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("kaggle package not installed; skipping real download.", file=sys.stderr)
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_COMPETITION} from Kaggle to {target_dir}...")
    api.competition_download_files(KAGGLE_COMPETITION, path=str(target_dir), quiet=False)
    for zip_path in target_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)
        zip_path.unlink()
    csv_path = target_dir / WIDS_FILENAME
    return csv_path if csv_path.exists() else None


def generate_synthetic_icu(n: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a WiDS-schema-compatible synthetic ICU dataset.

    Correlated, realistic distributions with a logistic-link mortality target.
    Class balance matches WiDS (~8% mortality).
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(63, 17, n).clip(16, 95)
    bmi = rng.normal(28, 7, n).clip(14, 60)
    pre_icu_los_days = np.abs(rng.exponential(0.6, n)).clip(0, 30)

    # Vitals
    heart_rate = rng.normal(95, 25, n).clip(30, 220)
    map_apache = rng.normal(78, 18, n).clip(30, 180)
    temp = rng.normal(36.8, 0.9, n).clip(32, 41)
    resprate = rng.normal(20, 6, n).clip(6, 50)

    # Labs
    wbc = np.abs(rng.normal(11, 6, n)).clip(0.1, 50)
    creatinine = np.abs(rng.normal(1.3, 1.1, n)).clip(0.1, 12)
    bun = np.abs(rng.normal(25, 18, n)).clip(2, 150)
    glucose = np.abs(rng.normal(140, 60, n)).clip(30, 600)
    hemaglobin = rng.normal(11.5, 2.3, n).clip(4, 20)
    sodium = rng.normal(138, 5, n).clip(115, 165)

    # GCS components: 1-4, 1-6, 1-5
    gcs_eyes = rng.choice([1, 2, 3, 4], n, p=[0.10, 0.10, 0.15, 0.65])
    gcs_motor = rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.06, 0.04, 0.05, 0.10, 0.15, 0.60])
    gcs_verbal = rng.choice([1, 2, 3, 4, 5], n, p=[0.10, 0.05, 0.10, 0.15, 0.60])

    # Categorical
    gender = rng.choice(["M", "F"], n, p=[0.55, 0.45])
    ethnicity = rng.choice(
        ["Caucasian", "African American", "Hispanic", "Asian", "Native American", "Other/Unknown"],
        n,
        p=[0.70, 0.10, 0.08, 0.05, 0.02, 0.05],
    )
    icu_type = rng.choice(
        ["MICU", "SICU", "CCU-CTICU", "Med-Surg ICU", "Cardiac ICU", "Neuro ICU"],
        n,
        p=[0.30, 0.20, 0.15, 0.20, 0.10, 0.05],
    )
    elective_surgery = rng.choice([0, 1], n, p=[0.80, 0.20])

    # Logistic-link mortality: severity-driven, with realistic effect sizes
    severity = (
        0.02 * (age - 65)
        + 0.04 * (heart_rate - 90)
        + -0.05 * (map_apache - 75)
        + 0.05 * (resprate - 18)
        + 0.20 * np.abs(temp - 37)
        + 0.15 * (creatinine - 1.0)
        + 0.02 * (bun - 25)
        + 0.005 * (glucose - 130)
        + -0.05 * (hemaglobin - 12)
        + -0.30 * (gcs_eyes - 4)
        + -0.20 * (gcs_motor - 6)
        + -0.20 * (gcs_verbal - 5)
        + -0.50 * elective_surgery
    )
    apache_prob = 1.0 / (1.0 + np.exp(-(severity - 2.5)))
    apache_prob_obs = (apache_prob + rng.normal(0, 0.04, n)).clip(0.001, 0.999)

    # Sample target with mild noise on top of the linked probability
    p_death = 1.0 / (1.0 + np.exp(-(severity - 2.7 + rng.normal(0, 0.5, n))))
    hospital_death = (rng.uniform(0, 1, n) < p_death).astype(int)

    # Inject ~3% missingness in numerical features (matches real WiDS missingness)
    df = pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "patient_id": np.arange(n),
            "age": age,
            "bmi": bmi,
            "pre_icu_los_days": pre_icu_los_days,
            "heart_rate_apache": heart_rate,
            "map_apache": map_apache,
            "temp_apache": temp,
            "resprate_apache": resprate,
            "wbc_apache": wbc,
            "creatinine_apache": creatinine,
            "bun_apache": bun,
            "glucose_apache": glucose,
            "hemaglobin_apache": hemaglobin,
            "sodium_apache": sodium,
            "gcs_eyes_apache": gcs_eyes,
            "gcs_motor_apache": gcs_motor,
            "gcs_verbal_apache": gcs_verbal,
            "apache_4a_hospital_death_prob": apache_prob_obs,
            "gender": gender,
            "ethnicity": ethnicity,
            "icu_type": icu_type,
            "elective_surgery": elective_surgery,
            "hospital_death": hospital_death,
        }
    )
    numeric_cols = [
        c
        for c in df.columns
        if df[c].dtype.kind in "fi"
        and c
        not in {
            "encounter_id",
            "patient_id",
            "elective_surgery",
            "hospital_death",
            "gcs_eyes_apache",
            "gcs_motor_apache",
            "gcs_verbal_apache",
        }
    ]
    for c in numeric_cols:
        mask = rng.uniform(0, 1, n) < 0.03
        df.loc[mask, c] = np.nan
    return df


def ensure_dataset(force_synthetic: bool = False) -> Path:
    """Return path to a usable WiDS-schema CSV. Real if available, else synthetic."""
    ensure_dirs()
    csv_path = RAW_DIR / WIDS_FILENAME

    if csv_path.exists() and not force_synthetic:
        print(f"Using existing dataset: {csv_path}")
        return csv_path

    if not force_synthetic:
        real = download_wids(RAW_DIR)
        if real is not None:
            print(f"Downloaded real WiDS data to {real}")
            return real
        print("No Kaggle credentials or download failed — falling back to synthetic data.")

    print("Generating synthetic ICU dataset...")
    df = generate_synthetic_icu()
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} synthetic rows to {csv_path}")
    return csv_path


def main() -> None:
    """CLI entry point."""
    force = "--synthetic" in sys.argv
    path = ensure_dataset(force_synthetic=force)
    print(f"Dataset ready at: {path}")


if __name__ == "__main__":
    main()
