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


def _add_measurement_noise(
    H: np.ndarray, measurement_snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """H_est = H + sigma * CN(0, 1) at the given per-entry measurement SNR."""
    sigma = np.sqrt(np.mean(np.abs(H) ** 2) / 10 ** (measurement_snr_db / 10))
    noise = rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape)
    return H + sigma * noise / np.sqrt(2)


def evaluate(
    scheme: CodebookScheme,
    channel: ChannelSource,
    snr_db: list[float] | np.ndarray = (0.0, 10.0, 20.0),
    rank: int = 1,
    n_drops: int = 50,
    n_slots: int = 1,
    rng: np.random.Generator | None = None,
    feedback_delay_slots: int = 0,
    measurement_snr_db: float | None = None,
) -> EvalResult:
    """Monte-Carlo evaluation of one scheme over channel drops.

    The scheme sees the channel for ``n_slots`` slot intervals (1 for
    non-Doppler codebooks, N4 for the R18 codebook); SE and SGCS are scored
    per interval against the true channel of that interval.

    Realism knobs:
    * ``feedback_delay_slots``: the PMI is selected on the first ``n_slots``
      intervals but scored on the window shifted that many intervals into
      the future (CSI aging -- this is what the R18 codebook compensates).
    * ``measurement_snr_db``: complex Gaussian estimation noise added to the
      channel the scheme sees; scoring still uses the true channel.
    """
    rng = rng or np.random.default_rng(0)
    snr_db = list(np.atleast_1d(snr_db))
    rhos = [10 ** (s / 10) for s in snr_db]
    se = np.zeros(len(rhos))
    se_ub = np.zeros(len(rhos))
    sgcs_vals = []
    bits = []
    for _ in range(n_drops):
        H_full = channel.generate(n_slots=n_slots + feedback_delay_slots, rng=rng)
        H_meas = H_full[:n_slots]  # (S, N3, Nr, P)
        if measurement_snr_db is not None:
            H_meas = _add_measurement_noise(H_meas, measurement_snr_db, rng)
        H = H_full[feedback_delay_slots:]  # scoring window
        pmi = scheme.select(H_meas, rank=rank)
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


@dataclass
class MuEvalResult:
    scheme: str
    snr_db: list[float]
    sum_rate: list[float]  # mean ZF/RZF sum rate from reported PMIs
    sum_rate_full_csi: list[float]  # same precoding from true eigen directions
    n_users: int


def evaluate_mu(
    scheme: CodebookScheme,
    channel: ChannelSource,
    n_users: int = 2,
    snr_db: list[float] | np.ndarray = (0.0, 10.0, 20.0),
    n_drops: int = 20,
    rng: np.random.Generator | None = None,
    regularization: float | None = None,
) -> MuEvalResult:
    """MU-MIMO evaluation (paper eq. 2): each user reports a rank-1 PMI on
    its own channel drop; the gNB zero-forces across the reported precoder
    directions (RZF with ``regularization`` if given) and the per-user rates
    include the residual inter-user interference.

    The full-CSI reference applies the same cross-user precoding to the true
    per-user eigenvectors, so the gap isolates the feedback quantization
    loss -- the comparison Type II codebooks (and ML schemes) exist for.
    """
    from ..baselines.ideal import eigen_precoder as eig

    rng = rng or np.random.default_rng(0)
    snr_db = list(np.atleast_1d(snr_db))
    rhos = [10 ** (s / 10) for s in snr_db]
    sums = np.zeros(len(rhos))
    sums_full = np.zeros(len(rhos))
    for _ in range(n_drops):
        Hs = [channel.generate(n_slots=1, rng=rng) for _ in range(n_users)]
        reported = []
        ideal = []
        for H in Hs:
            W = scheme.precoder(scheme.select(H, rank=1))  # (1, N3, P, 1)
            reported.append(W[0, :, :, 0])  # (N3, P)
            ideal.append(eig(H, rank=1)[0, :, :, 0])
        H_users = np.stack([H[0] for H in Hs])  # (K, N3, Nr, P)
        for i, rho in enumerate(rhos):
            sums[i] += _zf_sum_rate(H_users, np.stack(reported), rho, regularization)
            sums_full[i] += _zf_sum_rate(H_users, np.stack(ideal), rho, regularization)
    return MuEvalResult(
        scheme=scheme.name,
        snr_db=snr_db,
        sum_rate=list(sums / n_drops),
        sum_rate_full_csi=list(sums_full / n_drops),
        n_users=n_users,
    )


def _zf_sum_rate(
    H_users: np.ndarray, directions: np.ndarray, rho: float, xi: float | None
) -> float:
    """ZF/RZF across reported directions, scored with paper eq. (2).

    H_users: (K, N3, Nr, P) true channels; directions: (K, N3, P) the
    reported (or ideal) per-user precoder directions.
    """
    from ..baselines.ideal import rzf, zf
    from ..metrics.se import mu_rate

    K, N3 = directions.shape[:2]
    W = np.zeros((K, N3, directions.shape[2], 1), dtype=complex)
    for t in range(N3):
        D = directions[:, t, :].conj()  # (K, P) "channel" seen by the gNB
        F = zf(D) if xi is None else rzf(D, xi)  # (P, K)
        F = F / np.linalg.norm(F, axis=0, keepdims=True)
        for k in range(K):
            W[k, t, :, 0] = F[:, k]
    rates = mu_rate(H_users, W, rho / K)  # equal power split across users
    return float(np.sum(rates))
