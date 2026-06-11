"""Evaluation harness: codebooks (or ML schemes) vs. ideal beamforming.

Any object implementing ``CodebookScheme`` (``select``/``precoder``/
``overhead_bits``) can be evaluated -- this is the drop-in point for a
learned CSI feedback algorithm: wrap the encoder/decoder pair in the same
interface and pass it to ``evaluate`` together with the 3GPP codebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..baselines.ideal import eigen_precoder
from ..channel.base import ChannelSource
from ..codebooks.base import CodebookScheme
from ..metrics.se import su_rate
from ..metrics.similarity import sgcs


@dataclass
class EvalResult:
    scheme: str
    snr_db: list[float]
    se: list[float]  # mean SE per SNR point (bits/s/Hz)
    se_upper_bound: list[float]  # per-interval eigen beamforming
    sgcs: float  # mean SGCS vs per-interval eigen targets
    overhead_bits: float  # mean feedback bits per report
    per_drop_sgcs: list[float] = field(default_factory=list)


def evaluate(
    scheme: CodebookScheme,
    channel: ChannelSource,
    snr_db: list[float] | np.ndarray = (0.0, 10.0, 20.0),
    rank: int = 1,
    n_drops: int = 50,
    n_slots: int = 1,
    rng: np.random.Generator | None = None,
) -> EvalResult:
    """Monte-Carlo evaluation of one scheme over channel drops.

    The scheme sees the channel for ``n_slots`` slot intervals (1 for
    non-Doppler codebooks, N4 for the R18 codebook); SE and SGCS are scored
    per interval against the true channel of that interval.
    """
    rng = rng or np.random.default_rng(0)
    snr_db = list(np.atleast_1d(snr_db))
    rhos = [10 ** (s / 10) for s in snr_db]
    se = np.zeros(len(rhos))
    se_ub = np.zeros(len(rhos))
    sgcs_vals = []
    bits = []
    for _ in range(n_drops):
        H = channel.generate(n_slots=n_slots, rng=rng)  # (S, N3, Nr, P)
        pmi = scheme.select(H, rank=rank)
        W = scheme.precoder(pmi)  # (S_out, N3, P, v)
        S_out = W.shape[0]
        # non-Doppler schemes report once; replicate across intervals
        W_all = W if S_out == H.shape[0] else np.repeat(W[-1:], H.shape[0], axis=0)
        W_ref = eigen_precoder(H, rank=rank)
        sgcs_vals.append(sgcs(W_ref, W_all))
        bits.append(scheme.total_overhead_bits(pmi))
        for i, rho in enumerate(rhos):
            se[i] += su_rate(H, W_all, rho)
            se_ub[i] += su_rate(H, W_ref, rho)
    return EvalResult(
        scheme=scheme.name,
        snr_db=snr_db,
        se=list(se / n_drops),
        se_upper_bound=list(se_ub / n_drops),
        sgcs=float(np.mean(sgcs_vals)),
        overhead_bits=float(np.mean(bits)),
        per_drop_sgcs=[float(x) for x in sgcs_vals],
    )
