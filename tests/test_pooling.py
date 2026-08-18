import numpy as np
import pytest

from pisa_specsens.pooling import bootstrap_variance, pool_rubin


def test_identical_estimates_give_no_between_variance():
    pooled = pool_rubin([0.7] * 10, [0.001] * 10)
    assert pooled.between_variance == 0.0
    assert np.isclose(pooled.estimate, 0.7)
    assert np.isclose(pooled.standard_error, np.sqrt(0.001))


def test_between_variance_widens_the_interval():
    """This is the substantive point of pooling across plausible values."""
    tight = pool_rubin([0.70] * 10, [0.001] * 10)
    spread = pool_rubin(list(np.linspace(0.60, 0.80, 10)), [0.001] * 10)
    assert spread.between_variance > 0
    assert spread.standard_error > tight.standard_error
    assert (spread.ci_high - spread.ci_low) > (tight.ci_high - tight.ci_low)


def test_single_estimate_omits_between_variance_entirely():
    """A single plausible value cannot express between-imputation variance."""
    single = pool_rubin([0.7], [0.001])
    assert single.m == 1
    assert single.between_variance == 0.0


def test_rubin_inflation_factor_is_applied():
    estimates = [0.5, 0.9]
    pooled = pool_rubin(estimates, [0.0, 0.0])
    b = np.var(estimates, ddof=1)
    assert np.isclose(pooled.between_variance, b)
    assert np.isclose(pooled.standard_error, np.sqrt((1 + 1 / 2) * b))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        pool_rubin([], [])


def test_bootstrap_variance_is_non_negative_and_seeded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    w = np.ones(200)
    fn = lambda a, b, c: float(np.average(b, weights=c))
    first = bootstrap_variance(fn, y, s, w, 50, seed=11)
    second = bootstrap_variance(fn, y, s, w, 50, seed=11)
    assert first >= 0
    assert first == second


def test_brr_variance_uses_the_fay_denominator():
    """R = 80 and k = 0.5 give a denominator of 20, not 80."""
    import numpy as np

    from pisa_specsens.pooling import brr_variance

    y = np.zeros(50)
    score = np.zeros(50)
    reps = np.ones((50, 4))
    # Metric returns the mean replicate weight, so replicate r yields value r.
    fn = lambda a, b, w: float(np.mean(w))
    reps = np.column_stack([np.full(50, v) for v in (1.0, 2.0, 3.0, 4.0)])
    v = brr_variance(fn, y, score, reps, point_estimate=0.0)
    expected = sum(x**2 for x in (1, 2, 3, 4)) / (4 * 0.25)
    assert np.isclose(v, expected)


def test_brr_variance_is_zero_without_replicates():
    import numpy as np

    from pisa_specsens.pooling import brr_variance

    assert brr_variance(lambda a, b, w: 1.0, np.zeros(3), np.zeros(3), np.empty((3, 0)), 1.0) == 0.0
