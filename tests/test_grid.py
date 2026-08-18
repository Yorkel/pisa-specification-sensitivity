import numpy as np
import pytest

from pisa_specsens.config import Specification
from pisa_specsens.grid import metric_for, run_cell
from pisa_specsens.preprocessing import prepare


@pytest.fixture
def prepared(synthetic_frame):
    return prepare(synthetic_frame)


def test_metric_choice_follows_target_form():
    assert metric_for("binary")[1] == "weighted_auc"
    assert metric_for("continuous")[1] == "weighted_r2"


def test_pv1_cell_fits_one_model(prepared):
    splits, pv = prepared
    spec = Specification("pv1_only", "weighted", "pruned_tree", "continuous", None)
    record = run_cell(spec, splits, pv, seed=1)
    assert record["n_fits"] == 1
    assert record["m"] == 1


def test_pooled_cell_fits_ten_models(prepared):
    splits, pv = prepared
    spec = Specification("pooled_rubin", "weighted", "pruned_tree", "continuous", None)
    record = run_cell(spec, splits, pv, seed=1)
    assert record["n_fits"] == 10
    assert record["m"] == 10


def test_escs_leads_when_it_carries_the_signal(prepared):
    """The synthetic frame builds the outcome from ESCS, so it must rank first."""
    splits, pv = prepared
    spec = Specification("pv1_only", "weighted", "random_forest", "continuous", None)
    record = run_cell(spec, splits, pv, seed=1)
    assert record["focal_rank"] == 1
    assert "ESCS" in record["top_k"]


def test_record_carries_every_axis_and_interval(prepared):
    splits, pv = prepared
    spec = Specification("pv1_only", "unweighted", "gradient_boosting", "binary", "level_2")
    record = run_cell(spec, splits, pv, seed=1)
    for key in ("pv_handling", "weighting", "model_family", "target_form", "threshold"):
        assert key in record
    assert record["ci_low"] <= record["estimate"] <= record["ci_high"]
    assert len(record["top_k"]) == 5


def test_binary_metric_lies_in_the_auc_range(prepared):
    splits, pv = prepared
    spec = Specification("pv1_only", "weighted", "random_forest", "binary", "level_2")
    record = run_cell(spec, splits, pv, seed=1)
    assert 0.0 <= record["estimate"] <= 1.0
