"""Synthetic frames standing in for the PISA extract.

Every test in this suite runs offline and without the OECD data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pisa_specsens.config import CANDIDATE_INDICES, PV_COLUMNS, WEIGHT_COLUMN


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """A small frame with the column structure the study expects.

    ESCS is constructed to carry real signal so that rank assertions are
    meaningful; the remaining indices are noise. Two indices are given high
    missingness so that the drop rule is exercised.
    """
    rng = np.random.default_rng(7)
    n = 600

    data: dict[str, np.ndarray] = {}
    for name in CANDIDATE_INDICES:
        data[name] = rng.normal(0, 1, n)

    escs = data["ESCS"]
    latent = 470 + 45 * escs + rng.normal(0, 35, n)
    for column in PV_COLUMNS:
        data[column] = latent + rng.normal(0, 12, n)

    data[WEIGHT_COLUMN] = rng.uniform(0.5, 3.0, n)
    frame = pd.DataFrame(data)

    frame["ST004D01T"] = rng.choice([1, 2], n)
    frame["REGION"] = rng.choice([82611, 82612, 82613, 82620], n)

    # Exercise the missingness drop rule.
    frame.loc[frame.sample(frac=0.45, random_state=1).index, "LEARRES"] = np.nan
    frame.loc[frame.sample(frac=0.50, random_state=2).index, "SDLEFF"] = np.nan
    # Ordinary missingness that must survive the rule and be imputed.
    frame.loc[frame.sample(frac=0.12, random_state=3).index, "BELONG"] = np.nan

    return frame
