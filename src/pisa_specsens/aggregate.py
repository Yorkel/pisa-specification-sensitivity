"""Aggregation across the specification grid.

The unit of analysis here is the grid, not the model. Each function answers a
question about how far a conclusion moves, and which specification axis moves it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AXES = ["pv_handling", "weighting", "model_family", "target_form", "threshold"]


def _matched_sets(frame: pd.DataFrame, axis: str) -> list[pd.DataFrame]:
    """Group cells that are identical except on the named axis.

    Comparing only within matched sets isolates the axis. An unmatched
    comparison would confound it with every other choice that also differs.
    """
    others = [a for a in AXES if a != axis]
    groups = []
    for _, group in frame.groupby(others, dropna=False, observed=True):
        if group[axis].nunique(dropna=False) > 1:
            groups.append(group)
    return groups


def axis_attribution(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Average movement in an outcome attributable to each axis.

    For each axis, cells are grouped so that every other axis is held fixed.
    Within a group the range of the outcome is the movement caused by that axis
    alone. The reported figure is the mean of those within-group ranges.
    """
    rows = []
    for axis in AXES:
        groups = _matched_sets(frame, axis)
        ranges = []
        for group in groups:
            values = pd.to_numeric(group[outcome], errors="coerce").dropna()
            if len(values) > 1:
                ranges.append(float(values.max() - values.min()))
        if not ranges:
            continue
        rows.append(
            {
                "axis": axis,
                "outcome": outcome,
                "mean_within_set_range": float(np.mean(ranges)),
                "max_within_set_range": float(np.max(ranges)),
                "matched_sets": len(ranges),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("mean_within_set_range", ascending=False).reset_index(drop=True)


def focal_rank_summary(frame: pd.DataFrame) -> dict:
    """How stable is the rank of the focal predictor across the grid."""
    ranks = pd.to_numeric(frame["focal_rank"], errors="coerce").dropna()
    if ranks.empty:
        return {"n": 0}
    counts = ranks.value_counts().sort_index()
    return {
        "n": int(ranks.size),
        "min_rank": int(ranks.min()),
        "max_rank": int(ranks.max()),
        "median_rank": float(ranks.median()),
        "share_rank_1": float((ranks == 1).mean()),
        "share_top_3": float((ranks <= 3).mean()),
        "distribution": {int(k): int(v) for k, v in counts.items()},
    }


def top_k_stability(frame: pd.DataFrame) -> dict:
    """How much the reported set of leading predictors changes across the grid."""
    sets = [frozenset(row) for row in frame["top_k"]]
    distinct = {s for s in sets}
    counter: dict[str, int] = {}
    for s in sets:
        for feature in s:
            counter[feature] = counter.get(feature, 0) + 1

    pairwise = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if union:
                pairwise.append(len(sets[i] & sets[j]) / len(union))

    always = sorted(f for f, c in counter.items() if c == len(sets))
    return {
        "n_specifications": len(sets),
        "distinct_top_k_sets": len(distinct),
        "mean_pairwise_jaccard": float(np.mean(pairwise)) if pairwise else float("nan"),
        "features_in_every_top_k": always,
        "appearance_counts": dict(
            sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }


def metric_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    """Range of the performance metric, reported separately per target form."""
    rows = []
    for (target_form, metric_name), group in frame.groupby(
        ["target_form", "metric_name"], observed=True
    ):
        estimates = pd.to_numeric(group["estimate"], errors="coerce").dropna()
        rows.append(
            {
                "target_form": target_form,
                "metric": metric_name,
                "n_cells": int(len(group)),
                "min": float(estimates.min()),
                "max": float(estimates.max()),
                "range": float(estimates.max() - estimates.min()),
                "median": float(estimates.median()),
                "widest_ci": float(
                    (group["ci_high"] - group["ci_low"]).max()
                ),
            }
        )
    return pd.DataFrame(rows)


def pv_inflation(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare interval width under PV1 only against correct pooling.

    Using a single plausible value omits the between-imputation component of
    variance. This function reports how much the interval widens once that
    component is restored.
    """
    keys = ["weighting", "model_family", "target_form", "threshold"]
    frame = frame.copy()
    frame["ci_width"] = frame["ci_high"] - frame["ci_low"]
    wide = frame.pivot_table(
        index=keys, columns="pv_handling", values="ci_width", dropna=False, observed=True
    ).dropna()
    if wide.empty or not {"pv1_only", "pooled_rubin"}.issubset(wide.columns):
        return pd.DataFrame()
    wide = wide.reset_index()
    wide["width_ratio"] = wide["pooled_rubin"] / wide["pv1_only"]
    return wide.sort_values("width_ratio", ascending=False).reset_index(drop=True)


def summarise(frame: pd.DataFrame) -> dict:
    """Every headline figure the README reports."""
    attribution = {
        outcome: axis_attribution(frame, outcome).to_dict("records")
        for outcome in ("estimate", "focal_rank")
    }
    return {
        "n_cells": int(len(frame)),
        "focal_rank": focal_rank_summary(frame),
        "top_k_stability": top_k_stability(frame),
        "metric_ranges": metric_ranges(frame).to_dict("records"),
        "axis_attribution": attribution,
        "pv_inflation": pv_inflation(frame).to_dict("records"),
    }
