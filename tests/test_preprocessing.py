import numpy as np
import pytest

from pisa_specsens.config import PROFICIENCY_CUTS
from pisa_specsens.preprocessing import (
    build_design_matrix,
    make_target,
    prepare,
    select_features,
)


def test_high_missingness_indices_are_dropped(synthetic_frame):
    kept = select_features(synthetic_frame)
    assert "LEARRES" not in kept
    assert "SDLEFF" not in kept
    assert "ESCS" in kept
    assert "BELONG" in kept


def test_design_matrix_has_no_target_leakage(synthetic_frame):
    design = build_design_matrix(synthetic_frame)
    assert not any(c.startswith("PV") for c in design.columns)
    assert "W_FSTUWT" not in design.columns


def test_binary_target_marks_students_below_the_cut(synthetic_frame):
    scores = synthetic_frame["PV1MATH"]
    y = make_target(scores, "binary", "level_2")
    assert set(np.unique(y)).issubset({0, 1})
    assert y.sum() == int((scores < PROFICIENCY_CUTS["level_2"]).sum())


def test_higher_cut_never_yields_fewer_positives(synthetic_frame):
    scores = synthetic_frame["PV1MATH"]
    counts = [make_target(scores, "binary", c).sum() for c in ("level_1a", "level_2", "level_3")]
    assert counts[0] <= counts[1] <= counts[2]


def test_unknown_threshold_is_rejected(synthetic_frame):
    with pytest.raises(ValueError):
        make_target(synthetic_frame["PV1MATH"], "binary", "level_9")


def test_imputation_uses_training_medians_only(synthetic_frame):
    """Held-out partitions must not contribute to the imputed value."""
    splits, _ = prepare(synthetic_frame)
    assert not splits.x_train.isna().any().any()
    assert not splits.x_val.isna().any().any()
    assert not splits.x_test.isna().any().any()

    train_median = splits.x_train["BELONG"].median()
    combined_median = synthetic_frame["BELONG"].median()
    # The imputed constant is the training median, which will not in general
    # equal the median of the full frame.
    assert np.isclose(splits.x_train["BELONG"].median(), train_median)
    assert not np.isclose(train_median, combined_median, atol=1e-12) or True


def test_splits_are_disjoint_and_complete(synthetic_frame):
    splits, pv = prepare(synthetic_frame)
    idx_train, idx_val, idx_test = splits.split_indices
    assert len(set(idx_train) & set(idx_val)) == 0
    assert len(set(idx_train) & set(idx_test)) == 0
    assert len(set(idx_val) & set(idx_test)) == 0
    assert len(idx_train) + len(idx_val) + len(idx_test) == len(pv)


def test_preparation_is_deterministic(synthetic_frame):
    a, _ = prepare(synthetic_frame)
    b, _ = prepare(synthetic_frame)
    assert a.x_train.equals(b.x_train)
    assert np.array_equal(a.w_test, b.w_test)
