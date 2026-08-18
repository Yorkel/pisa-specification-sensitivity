"""Rubin's rules for combining estimates across plausible values.

PISA reports ten plausible values per domain. Each is a draw from the posterior
distribution of a student's proficiency given their responses. Analysing one
plausible value treats a draw as if it were a measurement and understates the
variance of any resulting estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PooledEstimate:
    """A point estimate and interval combined across M imputations."""

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    within_variance: float
    between_variance: float
    m: int

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "within_variance": self.within_variance,
            "between_variance": self.between_variance,
            "m": self.m,
        }


def pool_rubin(estimates, variances, confidence: float = 0.95) -> PooledEstimate:
    """Combine M estimates and their sampling variances.

    Total variance is the within-imputation variance plus the between-imputation
    variance inflated by (1 + 1/M). Degrees of freedom follow Rubin (1987).
    """
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(variances, dtype=float)
    m = q.size
    if m == 0:
        raise ValueError("No estimates supplied to pool")

    q_bar = float(q.mean())
    u_bar = float(u.mean())

    if m == 1:
        total = u_bar
        b = 0.0
        df = np.inf
    else:
        b = float(((q - q_bar) ** 2).sum() / (m - 1))
        total = u_bar + (1.0 + 1.0 / m) * b
        if b <= 0 or total <= 0:
            df = np.inf
        else:
            gamma = (1.0 + 1.0 / m) * b / total
            df = (m - 1) / (gamma**2) if gamma > 0 else np.inf

    se = float(np.sqrt(max(total, 0.0)))
    crit = float(stats.t.ppf(0.5 + confidence / 2.0, df)) if np.isfinite(df) else 1.959963985
    return PooledEstimate(
        estimate=q_bar,
        standard_error=se,
        ci_low=q_bar - crit * se,
        ci_high=q_bar + crit * se,
        within_variance=u_bar,
        between_variance=b,
        m=int(m),
    )


def bootstrap_variance(metric_fn, y_true, y_score, weights, resamples: int, seed: int) -> float:
    """Sampling variance of a metric, by resampling the evaluation partition.

    The fitted model is held fixed and only the evaluation rows are resampled.
    This isolates evaluation uncertainty and keeps the grid affordable; it does
    not capture uncertainty arising from refitting the model.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        try:
            values.append(metric_fn(y_true[idx], y_score[idx], weights[idx]))
        except ValueError:
            continue
    if len(values) < 2:
        return 0.0
    return float(np.var(values, ddof=1))
