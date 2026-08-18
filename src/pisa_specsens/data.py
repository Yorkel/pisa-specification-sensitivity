"""Loading the PISA 2022 UK extract.

The PISA data is not distributed with this repository. See scripts/download_data.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    CANDIDATE_INDICES,
    CATEGORICAL_COLUMNS,
    PV_COLUMNS,
    REPLICATE_WEIGHT_COLUMNS,
    WEIGHT_COLUMN,
)


class DataNotFoundError(FileNotFoundError):
    """Raised when the PISA extract is absent from the expected location."""


def required_columns() -> list[str]:
    """Every column the study reads. Loading is restricted to these."""
    return [
        *CANDIDATE_INDICES,
        *CATEGORICAL_COLUMNS,
        *PV_COLUMNS,
        WEIGHT_COLUMN,
        *REPLICATE_WEIGHT_COLUMNS,
    ]


def load_pisa(path: str | Path) -> pd.DataFrame:
    """Load the UK extract, keeping only the columns the study uses.

    The published file carries 1278 columns. Restricting the read at load time
    keeps memory bounded and makes the study's data dependency explicit.
    """
    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(
            f"PISA extract not found at {path}. "
            "Run scripts/download_data.py for acquisition instructions."
        )

    header = pd.read_csv(path, nrows=0)
    wanted = [c for c in required_columns() if c in header.columns]
    missing = sorted(set(required_columns()) - set(wanted))
    if WEIGHT_COLUMN in missing or any(pv in missing for pv in PV_COLUMNS):
        raise ValueError(
            f"Extract at {path} lacks required weight or plausible value columns: {missing}"
        )

    frame = pd.read_csv(path, usecols=wanted, low_memory=False)
    return frame
