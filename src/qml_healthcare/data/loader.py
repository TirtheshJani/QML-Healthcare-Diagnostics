"""Raw CSV loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qml_healthcare.config import RAW_DIR


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load raw WiDS CSV. Defaults to the canonical training_v2.csv path."""
    path = path or (RAW_DIR / "training_v2.csv")
    return pd.read_csv(path)
