import pandas as pd

from pisa_specsens.aggregate import (
    axis_attribution,
    focal_rank_summary,
    metric_ranges,
    top_k_stability,
)


def _frame():
    """Two cells differing only on pv_handling, and two only on model_family."""
    rows = [
        dict(pv_handling="pv1_only", weighting="weighted", model_family="rf",
             target_form="binary", threshold="level_2", estimate=0.70,
             ci_low=0.68, ci_high=0.72, focal_rank=1, metric_name="weighted_auc",
             top_k=["ESCS", "A", "B", "C", "D"]),
        dict(pv_handling="pooled_rubin", weighting="weighted", model_family="rf",
             target_form="binary", threshold="level_2", estimate=0.72,
             ci_low=0.66, ci_high=0.78, focal_rank=1, metric_name="weighted_auc",
             top_k=["ESCS", "A", "B", "C", "D"]),
        dict(pv_handling="pv1_only", weighting="weighted", model_family="tree",
             target_form="binary", threshold="level_2", estimate=0.60,
             ci_low=0.57, ci_high=0.63, focal_rank=4, metric_name="weighted_auc",
             top_k=["A", "B", "C", "D", "E"]),
        dict(pv_handling="pooled_rubin", weighting="weighted", model_family="tree",
             target_form="binary", threshold="level_2", estimate=0.61,
             ci_low=0.55, ci_high=0.67, focal_rank=4, metric_name="weighted_auc",
             top_k=["A", "B", "C", "D", "E"]),
    ]
    return pd.DataFrame(rows)


def test_attribution_ranks_the_larger_mover_first():
    """model_family moves the metric by ~0.10, pv_handling by ~0.015."""
    result = axis_attribution(_frame(), "estimate")
    assert result.iloc[0]["axis"] == "model_family"
    top = result.set_index("axis")["mean_within_set_range"]
    assert top["model_family"] > top["pv_handling"]


def test_attribution_compares_only_within_matched_sets():
    result = axis_attribution(_frame(), "estimate")
    row = result.set_index("axis").loc["pv_handling"]
    # Two matched sets exist: one per model family.
    assert row["matched_sets"] == 2


def test_focal_rank_summary_reports_spread():
    summary = focal_rank_summary(_frame())
    assert summary["min_rank"] == 1
    assert summary["max_rank"] == 4
    assert summary["share_rank_1"] == 0.5


def test_top_k_stability_counts_distinct_sets():
    stability = top_k_stability(_frame())
    assert stability["distinct_top_k_sets"] == 2
    assert stability["n_specifications"] == 4


def test_metric_ranges_are_reported_per_target_form():
    ranges = metric_ranges(_frame())
    assert len(ranges) == 1
    row = ranges.iloc[0]
    assert row["target_form"] == "binary"
    assert row["n_cells"] == 4
    assert abs(row["range"] - 0.12) < 1e-9
