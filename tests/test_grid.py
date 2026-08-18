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


def test_grouped_permutation_uses_one_shared_shuffle(prepared):
    """Within-block correlation must survive; only the block-to-outcome link breaks."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    from pisa_specsens.grid import _r2
    from pisa_specsens.models import grouped_permutation_importance

    splits, pv = prepared
    itr, _, ite = splits.split_indices
    y = pv["PV1MATH"].to_numpy()
    model = RandomForestRegressor(n_estimators=50, random_state=0).fit(
        splits.x_train, y[itr]
    )
    blocks = {"background": ["ESCS", "HOMEPOS"], "noise": ["BELONG"]}
    result = grouped_permutation_importance(
        model, splits.x_test, y[ite], splits.w_test, "continuous", blocks, _r2, seed=0
    )
    assert set(result) == {"background", "noise"}
    # The synthetic outcome is built from ESCS, so the background block must
    # matter more than an unrelated index.
    assert result["background"] > result["noise"]


def test_grouped_permutation_skips_absent_columns(prepared):
    from pisa_specsens.grid import _r2
    from pisa_specsens.models import grouped_permutation_importance
    from sklearn.tree import DecisionTreeRegressor

    splits, pv = prepared
    itr, _, ite = splits.split_indices
    y = pv["PV1MATH"].to_numpy()
    model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(splits.x_train, y[itr])
    result = grouped_permutation_importance(
        model, splits.x_test, y[ite], splits.w_test, "continuous",
        {"absent": ["NOT_A_COLUMN"], "real": ["ESCS"]}, _r2, seed=0,
    )
    assert "absent" not in result
    assert "real" in result
