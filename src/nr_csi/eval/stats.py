"""Comparison statistics over Monte-Carlo drops and users.

The MU comparison methodology (3GPP-style) reports three numbers per
(scheme, overhead) point: the mean sum rate with a confidence interval over
drops, and a cell-edge proxy -- a low percentile of the pooled per-user
rates.  These helpers compute them from the raw samples ``evaluate`` /
``evaluate_mu`` now expose (``per_drop_sgcs``, ``per_drop_user_rates``).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def mean_ci(samples, confidence: float = 0.95) -> tuple[float, float]:
    """Student-t confidence interval on the mean.

    Returns ``(mean, half_width)``; ``half_width`` is 0 for fewer than two
    samples (a point estimate carries no spread information).
    """
    x = np.asarray(samples, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("mean_ci needs at least one sample")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    m = float(np.mean(x))
    if x.size < 2:
        return m, 0.0
    sem = float(np.std(x, ddof=1)) / np.sqrt(x.size)
    t = float(sps.t.ppf(0.5 + confidence / 2, df=x.size - 1))
    return m, t * sem


def bootstrap_ci(
    samples,
    stat=np.mean,
    confidence: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile-bootstrap ``(lo, hi)`` interval for ``stat`` over samples.

    Distribution-free companion to ``mean_ci`` for statistics like
    percentiles whose sampling distribution is not close to normal.
    """
    x = np.asarray(samples, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("bootstrap_ci needs at least one sample")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = np.apply_along_axis(stat, 1, x[idx])
    alpha = 1 - confidence
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def edge_rate(per_user_rates, q: float = 5.0) -> float:
    """Cell-edge proxy: the q-th percentile of the pooled per-user rates.

    ``per_user_rates`` is any nesting of per-user rate samples (e.g. one SNR
    slice of ``MuEvalResult.per_drop_user_rates``, shape (n_drops, K)); the
    pool flattens drops and users together, mirroring how system-level
    evaluations pool the user-throughput CDF across drops.
    """
    x = np.asarray(per_user_rates, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("edge_rate needs at least one sample")
    return float(np.percentile(x, q))
