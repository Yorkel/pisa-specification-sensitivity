"""Execution of the specification grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

from .config import (
    BOOTSTRAP_RESAMPLES,
    FOCAL_FEATURE,
    PV_COLUMNS,
    RANDOM_SEED,
    TOP_K,
    Specification,
)
from .models import build_model, feature_importances, fit_model
from .pooling import bootstrap_variance, pool_rubin
from .preprocessing import Splits, make_target


def _auc(y_true, y_score, weights) -> float:
    if len(np.unique(y_true)) < 2:
        raise ValueError("Only one class present")
    return float(roc_auc_score(y_true, y_score, sample_weight=weights))


def _r2(y_true, y_score, weights) -> float:
    return float(r2_score(y_true, y_score, sample_weight=weights))


def metric_for(target_form: str):
    """Primary performance metric and its name for a target formulation.

    Binary and continuous cells are not on a common scale. Ranges are therefore
    reported within each formulation, never pooled across the two.
    """
    if target_form == "binary":
        return _auc, "weighted_auc"
    return _r2, "weighted_r2"


def run_single_pv(
    spec: Specification, splits: Splits, scores: pd.Series, seed: int
) -> tuple[float, float, np.ndarray]:
    """Fit one model on one plausible value and evaluate it on the test split.

    Returns the metric, its bootstrap sampling variance, and the importance
    vector.
    """
    idx_train, _, idx_test = splits.split_indices  # type: ignore[attr-defined]
    y_train = make_target(scores.iloc[idx_train], spec.target_form, spec.threshold)
    y_test = make_target(scores.iloc[idx_test], spec.target_form, spec.threshold)

    model = build_model(spec.model_family, spec.target_form, seed=seed)
    model = fit_model(model, splits.x_train, y_train, splits.w_train, spec.weighting)

    if spec.target_form == "binary":
        y_score = model.predict_proba(splits.x_test)[:, 1]
    else:
        y_score = model.predict(splits.x_test)

    metric_fn, _ = metric_for(spec.target_form)
    eval_weights = (
        splits.w_test if spec.weighting == "weighted" else np.ones_like(splits.w_test)
    )
    value = metric_fn(y_test, y_score, eval_weights)
    variance = bootstrap_variance(
        metric_fn, y_test, y_score, eval_weights, BOOTSTRAP_RESAMPLES, seed
    )
    importances = feature_importances(
        model,
        splits.x_test,
        y_test,
        eval_weights,
        spec.target_form,
        splits.feature_names,
        seed=seed,
    )
    return value, variance, importances


def run_cell(
    spec: Specification, splits: Splits, pv_frame: pd.DataFrame, seed: int = RANDOM_SEED
) -> dict:
    """Run one specification cell and record its headline conclusions.

    Under 'pv1_only' a single model is fitted to PV1MATH. Under 'pooled_rubin'
    ten models are fitted, one per plausible value, and their estimates are
    combined with Rubin's rules. Importances are averaged across the ten fits
    before ranking.
    """
    columns = [PV_COLUMNS[0]] if spec.pv_handling == "pv1_only" else PV_COLUMNS

    values, variances, importance_rows = [], [], []
    for offset, column in enumerate(columns):
        value, variance, importances = run_single_pv(
            spec, splits, pv_frame[column], seed=seed + offset
        )
        values.append(value)
        variances.append(variance)
        importance_rows.append(importances)

    pooled = pool_rubin(values, variances)
    mean_importance = np.mean(np.vstack(importance_rows), axis=0)

    order = np.argsort(mean_importance)[::-1]
    ranked = [splits.feature_names[i] for i in order]
    focal_rank = ranked.index(FOCAL_FEATURE) + 1 if FOCAL_FEATURE in ranked else None

    _, metric_name = metric_for(spec.target_form)
    record = spec.as_dict()
    record.update(
        {
            "metric_name": metric_name,
            "n_fits": len(columns),
            "top_k": ranked[:TOP_K],
            "focal_feature": FOCAL_FEATURE,
            "focal_rank": focal_rank,
            "focal_importance": float(mean_importance[order[focal_rank - 1]])
            if focal_rank
            else None,
            **pooled.as_dict(),
        }
    )
    return record


def run_grid(specs, splits, pv_frame, seed: int = RANDOM_SEED, progress=None) -> pd.DataFrame:
    """Run every specification and return one row per cell."""
    records = []
    for position, spec in enumerate(specs):
        records.append(run_cell(spec, splits, pv_frame, seed=seed))
        if progress is not None:
            progress(position + 1, len(specs), spec)
    return pd.DataFrame.from_records(records)
