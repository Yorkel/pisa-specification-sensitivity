"""Deterministic preprocessing shared by every specification.

Every cell of the grid calls these functions. Two specifications differ only by
their declared axes, never by an incidental difference in data handling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CANDIDATE_INDICES,
    CATEGORICAL_COLUMNS,
    MISSINGNESS_DROP_THRESHOLD,
    PROFICIENCY_CUTS,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
    WEIGHT_COLUMN,
)


@dataclass
class Splits:
    """Train, validation and test partitions with aligned weights."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    w_train: np.ndarray
    w_val: np.ndarray
    w_test: np.ndarray
    feature_names: list[str]


def select_features(frame: pd.DataFrame) -> list[str]:
    """Drop indices whose missingness exceeds the declared threshold.

    The threshold is applied to the full frame before splitting because it is a
    property of the questionnaire design (rotated forms), not of any split.
    """
    present = [c for c in CANDIDATE_INDICES if c in frame.columns]
    missingness = frame[present].isna().mean()
    kept = [c for c in present if missingness[c] <= MISSINGNESS_DROP_THRESHOLD]
    return kept


def encode_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode gender and UK nation, dropping the first level."""
    present = [c for c in CATEGORICAL_COLUMNS if c in frame.columns]
    if not present:
        return pd.DataFrame(index=frame.index)
    return pd.get_dummies(
        frame[present].astype("object"), columns=present, drop_first=True, dtype=float
    )


def build_design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Assemble the feature matrix used by every specification."""
    numeric = select_features(frame)
    categorical = encode_categoricals(frame)
    design = pd.concat([frame[numeric], categorical], axis=1)
    return design


def make_target(scores: pd.Series, target_form: str, threshold: str | None) -> np.ndarray:
    """Derive the outcome from a plausible value column.

    The binary target marks students *below* the named OECD proficiency cut, so
    a positive case is a low performer. This is the OECD construct, not a median
    split of the score distribution.
    """
    if target_form == "continuous":
        return scores.to_numpy(dtype=float)
    if threshold not in PROFICIENCY_CUTS:
        raise ValueError(f"Unknown proficiency threshold: {threshold}")
    return (scores < PROFICIENCY_CUTS[threshold]).to_numpy().astype(int)


def impute_median(splits: Splits) -> Splits:
    """Median imputation fitted on the training partition only.

    Fitting the imputer on train and transforming validation and test prevents
    information from the held-out partitions reaching the model.
    """
    medians = splits.x_train.median(numeric_only=True)
    return Splits(
        x_train=splits.x_train.fillna(medians),
        x_val=splits.x_val.fillna(medians),
        x_test=splits.x_test.fillna(medians),
        w_train=splits.w_train,
        w_val=splits.w_val,
        w_test=splits.w_test,
        feature_names=splits.feature_names,
    )


def split_data(design: pd.DataFrame, weights: pd.Series, seed: int = RANDOM_SEED) -> Splits:
    """Partition into train, validation and test with a fixed seed.

    The split is computed once from the design matrix and reused for every
    plausible value, so that pooled estimates differ only by the outcome draw.
    """
    idx = np.arange(len(design))
    idx_rest, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=seed)
    idx_train, idx_val = train_test_split(
        idx_rest, test_size=VAL_SIZE / (1.0 - TEST_SIZE), random_state=seed
    )

    w = weights.to_numpy(dtype=float)
    splits = Splits(
        x_train=design.iloc[idx_train].reset_index(drop=True),
        x_val=design.iloc[idx_val].reset_index(drop=True),
        x_test=design.iloc[idx_test].reset_index(drop=True),
        w_train=w[idx_train],
        w_val=w[idx_val],
        w_test=w[idx_test],
        feature_names=list(design.columns),
    )
    splits = impute_median(splits)
    splits.split_indices = (idx_train, idx_val, idx_test)  # type: ignore[attr-defined]
    return splits


def prepare(frame: pd.DataFrame, seed: int = RANDOM_SEED) -> tuple[Splits, pd.DataFrame]:
    """Full preprocessing path. Returns splits and the retained plausible values."""
    design = build_design_matrix(frame)
    complete = frame[WEIGHT_COLUMN].notna()
    design = design.loc[complete].reset_index(drop=True)
    weights = frame.loc[complete, WEIGHT_COLUMN].reset_index(drop=True)
    pv_frame = frame.loc[complete, [c for c in frame.columns if c.startswith("PV")]]
    pv_frame = pv_frame.reset_index(drop=True)
    return split_data(design, weights, seed=seed), pv_frame
