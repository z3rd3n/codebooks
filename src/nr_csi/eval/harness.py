"""Evaluation harness: codebooks (or ML schemes) vs. ideal beamforming.

Any object implementing ``CodebookScheme`` (``select``/``precoder``/
``overhead_bits``) can be evaluated -- this is the drop-in point for a
learned CSI feedback algorithm: wrap the encoder/decoder pair in the same
interface and pass it to ``evaluate`` together with the 3GPP codebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..baselines.ideal import capacity_upper_bound, eigen_precoder
from ..channel.base import ChannelSource
from ..codebooks.base import CodebookScheme
from ..metrics.se import su_rate, su_rate_mmse
from ..metrics.similarity import sgcs, subspace_sgcs


@dataclass
class EvalResult:
    scheme: str
    snr_db: list[float]
    se: list[float]  # mean SE per SNR point (bits/s/Hz)
    se_upper_bound: list[float]  # per-interval eigen beamforming (equal power, achievable)
    sgcs: float  # mean SGCS vs per-interval eigen targets
    overhead_bits: float  # mean feedback bits per report
    per_drop_sgcs: list[float] = field(default_factory=list)
    subspace_sgcs: float = 0.0  # rotation-invariant companion of sgcs
    capacity_upper_bound: list[float] = field(default_factory=list)  # true waterfilling supremum
    se_mmse: list[float] = field(default_factory=list)  # per-layer linear MMSE receiver (no joint decoding)
    per_drop_rank: list[int] = field(default_factory=list)  # rank used per drop (varies with rank="auto")


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
    rank: int | str = 1,
    n_drops: int = 50,
    n_slots: int = 1,
    rng: np.random.Generator | None = None,
    feedback_delay_slots: int = 0,
    measurement_snr_db: float | None = None,
    measurement_slots: int | None = None,
    delay_aware: bool = False,
    select_snr_db: float = 10.0,
    auto_ranks: tuple[int, ...] | None = None,
) -> EvalResult:
    """Monte-Carlo evaluation of one scheme over channel drops.

    The scheme sees the channel for ``n_slots`` slot intervals (1 for
    non-Doppler codebooks, N4 for the R18 codebook); SE and SGCS are scored
    per interval against the true channel of that interval.

    ``rank="auto"`` enables per-drop rank adaptation (auto-RI): each drop
    reports the rank in ``auto_ranks`` (default 1..min(Nr, 8)) whose precoder
    maximizes SE on the *measured* channel at ``select_snr_db`` -- the
    decision a real UE makes -- and every reference (eigen bound, capacity
    bound, SGCS target) is computed at that drop's chosen rank.  The chosen
    ranks are recorded in ``EvalResult.per_drop_rank``.

    Realism knobs:
    * ``feedback_delay_slots``: the PMI is selected on the first ``n_slots``
      intervals but scored on the window shifted that many intervals into
      the future (CSI aging -- this is what the R18 codebook compensates).
    * ``measurement_snr_db``: complex Gaussian estimation noise added to the
      channel the scheme sees; scoring still uses the true channel.
    * ``measurement_slots``: the UE observes this many consecutive intervals
      ending at the report instant (default ``n_slots``).  Multi-interval
      schemes consume the most recent ``n_slots`` of the (noisy) observation;
      single-interval schemes its time-average -- a longer CSI-RS observation,
      so noise averaging is a level playing field with R18's N4-slot window.
      Static + noiseless => identical to the default for any value.
    * ``delay_aware``: the gNB applies the report knowing its age -- scoring
      interval ``j`` uses the predicted interval ``feedback_delay_slots + j``
      (clamped to the last reported interval) instead of interval ``j``.
      No-op for single-interval reports or zero delay.
    """
    rng = rng or np.random.default_rng(0)
    if isinstance(rank, str) and rank != "auto":
        raise ValueError(f"rank must be an int or 'auto', got {rank!r}")
    snr_db = list(np.atleast_1d(snr_db))
    rhos = [10 ** (s / 10) for s in snr_db]
    m = n_slots if measurement_slots is None else measurement_slots
    if m < n_slots:
        raise ValueError(f"measurement_slots {m} < n_slots {n_slots}")
    se = np.zeros(len(rhos))
    se_mmse = np.zeros(len(rhos))
    se_ub = np.zeros(len(rhos))
    se_cap_ub = np.zeros(len(rhos))
    sgcs_vals = []
    subspace_vals = []
    bits = []
    drop_ranks: list[int] = []
    for _ in range(n_drops):
        total = m + feedback_delay_slots
        H_full = channel.generate(n_slots=total, rng=rng)
        H_meas = H_full[:m]  # UE observation, (m, N3, Nr, P)
        if measurement_snr_db is not None:
            H_meas = _add_measurement_noise(H_meas, measurement_snr_db, rng)
        H_in = H_meas[m - n_slots:] if n_slots > 1 else H_meas.mean(0, keepdims=True)
        H = H_full[total - n_slots:]  # scoring window
        if rank == "auto":
            ranks = auto_ranks or tuple(range(1, min(H_in.shape[-2], 8) + 1))
            r_used, pmi, W, _ = select_rank(
                scheme, H_in, rho=10 ** (select_snr_db / 10), ranks=ranks
            )
        else:
            r_used = int(rank)
            pmi = scheme.select(H_in, rank=r_used)
            W = scheme.precoder(pmi)  # (S_out, N3, P, v)
        drop_ranks.append(r_used)
        S_out = W.shape[0]
        if delay_aware:
            # report interval s is absolute slot (m - n_slots) + s, scoring
            # index j absolute slot (m - n_slots) + feedback_delay_slots + j
            idx = np.minimum(np.arange(H.shape[0]) + feedback_delay_slots, S_out - 1)
            W_all = W[idx]
        elif S_out == H.shape[0]:
            W_all = W
        else:
            # non-Doppler schemes report once; replicate across intervals
            W_all = np.repeat(W[-1:], H.shape[0], axis=0)
        W_ref = eigen_precoder(H, rank=r_used)
        sgcs_vals.append(sgcs(W_ref, W_all))
        subspace_vals.append(subspace_sgcs(W_ref, W_all))
        bits.append(scheme.total_overhead_bits(pmi))
        for i, rho in enumerate(rhos):
            se[i] += su_rate(H, W_all, rho)
            se_mmse[i] += su_rate_mmse(H, W_all, rho)
            se_ub[i] += su_rate(H, W_ref, rho)
            se_cap_ub[i] += capacity_upper_bound(H, r_used, rho)
    return EvalResult(
        scheme=scheme.name,
        snr_db=snr_db,
        se=list(se / n_drops),
        se_upper_bound=list(se_ub / n_drops),
        sgcs=float(np.mean(sgcs_vals)),
        overhead_bits=float(np.mean(bits)),
        per_drop_sgcs=[float(x) for x in sgcs_vals],
        subspace_sgcs=float(np.mean(subspace_vals)),
        capacity_upper_bound=list(se_cap_ub / n_drops),
        se_mmse=list(se_mmse / n_drops),
        per_drop_rank=drop_ranks,
    )


def delay_sweep(
    scheme: CodebookScheme,
    channel: ChannelSource,
    delays: tuple[int, ...] = (0, 1, 2, 4, 8),
    seed: int = 0,
    **evaluate_kwargs,
) -> dict[int, EvalResult]:
    """``evaluate`` the scheme at each CSI feedback delay, matched drops.

    Every delay re-seeds the rng with the same ``seed``; ray-based channels
    draw their per-drop parameters independently of the requested slot count,
    so all delays see the *same* channel realizations and the curve isolates
    pure CSI aging -- the axis the R18 Doppler/predicted codebooks exist for
    (pass ``delay_aware=True`` and ``n_slots=scheme.N4`` to let them use it).

    Returns ``{delay: EvalResult}`` in the order given.
    """
    if "feedback_delay_slots" in evaluate_kwargs or "rng" in evaluate_kwargs:
        raise ValueError("delay_sweep sets feedback_delay_slots/rng itself; "
                         "pass delays=... and seed=... instead")
    return {
        int(d): evaluate(
            scheme, channel, feedback_delay_slots=int(d),
            rng=np.random.default_rng(seed), **evaluate_kwargs,
        )
        for d in delays
    }


def select_rank(
    scheme: CodebookScheme,
    H: np.ndarray,
    rho: float = 10.0,
    ranks: tuple[int, ...] = (1, 2, 3, 4),
):
    """Auto-RI: pick the rank whose reported precoder maximizes SE on ``H``.

    Ranks the scheme refuses (unsupported, prohibited by a rank-restriction
    bitmap, or beyond the channel rank) are skipped.  Returns
    (rank, pmi, W, se).
    """
    best = None
    for rank in ranks:
        try:
            pmi = scheme.select(H, rank=rank)
        except ValueError:
            continue
        W = scheme.precoder(pmi)
        W_all = W if W.shape[0] == H.shape[0] else np.repeat(W[-1:], H.shape[0], axis=0)
        se = su_rate(H, W_all, rho)
        if best is None or se > best[3]:
            best = (rank, pmi, W, se)
    if best is None:
        raise ValueError("no admissible rank for this scheme/channel")
    return best


@dataclass
class MuEvalResult:
    scheme: str
    snr_db: list[float]
    sum_rate: list[float]  # mean ZF/RZF sum rate from reported PMIs
    sum_rate_full_csi: list[float]  # same precoding from true eigen directions
    n_users: int
    rank: int = 1  # layers (streams) per user
    # (n_drops, n_snr, K) per-user rates -- the raw samples behind sum_rate,
    # kept so callers can compute cell-edge percentiles and CIs over drops
    per_drop_user_rates: list = field(default_factory=list)
    per_drop_user_rates_full_csi: list = field(default_factory=list)
    overhead_bits: float = 0.0  # mean feedback bits per user report


def evaluate_mu(
    scheme: CodebookScheme,
    channel: ChannelSource,
    n_users: int = 2,
    snr_db: list[float] | np.ndarray = (0.0, 10.0, 20.0),
    n_drops: int = 20,
    rng: np.random.Generator | None = None,
    regularization: float | None = None,
    rank: int = 1,
) -> MuEvalResult:
    """MU-MIMO evaluation (paper eq. 2): each user reports a rank-``rank`` PMI
    on its own channel drop; the gNB zero-forces across *all* reported
    precoder directions (RZF with ``regularization`` if given) so each of the
    K*rank streams nulls the others, and the per-user rates include the
    residual inter-user (and inter-stream) interference.  A user's rate sums
    its ``rank`` layers; transmit power ``rho`` is split equally across the
    K*rank streams (``rank=1`` reproduces the single-stream-per-user model
    exactly).

    The full-CSI reference applies the same cross-stream precoding to the true
    per-user eigenvectors, so the gap isolates the feedback quantization
    loss -- the comparison Type II codebooks (and ML schemes) exist for.
    """
    from ..baselines.ideal import eigen_precoder as eig

    rng = rng or np.random.default_rng(0)
    snr_db = list(np.atleast_1d(snr_db))
    rhos = [10 ** (s / 10) for s in snr_db]
    sums = np.zeros(len(rhos))
    sums_full = np.zeros(len(rhos))
    drop_rates: list[list[list[float]]] = []
    drop_rates_full: list[list[list[float]]] = []
    bits: list[int] = []
    for _ in range(n_drops):
        Hs = [channel.generate(n_slots=1, rng=rng) for _ in range(n_users)]
        reported = []
        ideal = []
        for H in Hs:
            pmi = scheme.select(H, rank=rank)
            W = scheme.precoder(pmi)  # (1, N3, P, rank)
            bits.append(scheme.total_overhead_bits(pmi))
            reported.append(W[0])  # (N3, P, rank)
            ideal.append(eig(H, rank=rank)[0])  # (N3, P, rank)
        H_users = np.stack([H[0] for H in Hs])  # (K, N3, Nr, P)
        rates_snr, rates_snr_full = [], []
        for i, rho in enumerate(rhos):
            rates = _zf_user_rates(H_users, np.stack(reported), rho, regularization)
            rates_full = _zf_user_rates(H_users, np.stack(ideal), rho, regularization)
            sums[i] += rates.sum()
            sums_full[i] += rates_full.sum()
            rates_snr.append([float(r) for r in rates])
            rates_snr_full.append([float(r) for r in rates_full])
        drop_rates.append(rates_snr)
        drop_rates_full.append(rates_snr_full)
    return MuEvalResult(
        scheme=scheme.name,
        snr_db=snr_db,
        sum_rate=list(sums / n_drops),
        sum_rate_full_csi=list(sums_full / n_drops),
        n_users=n_users,
        rank=rank,
        per_drop_user_rates=drop_rates,
        per_drop_user_rates_full_csi=drop_rates_full,
        overhead_bits=float(np.mean(bits)),
    )


def _zf_sum_rate(
    H_users: np.ndarray, directions: np.ndarray, rho: float, xi: float | None
) -> float:
    """Sum of ``_zf_user_rates`` (kept for callers that only need the total)."""
    return float(np.sum(_zf_user_rates(H_users, directions, rho, xi)))


def _zf_user_rates(
    H_users: np.ndarray, directions: np.ndarray, rho: float, xi: float | None
) -> np.ndarray:
    """ZF/RZF across all reported stream-directions, scored with paper eq. (2).

    H_users: (K, N3, Nr, P) true channels; directions: (K, N3, P, v) the
    reported (or ideal) per-user precoder directions (v = layers per user).
    The gNB treats every one of the K*v layers as a stream to be isolated:
    it zero-forces across the stacked K*v directions, normalizes each beam,
    and splits ``rho`` equally over the K*v streams.  The per-beam direction
    is invariant to the reported columns' magnitudes (row scaling of the
    stacked matrix cancels under the pseudo-inverse + per-column norm), so
    v = 1 reproduces the single-stream model bit-for-bit.

    Returns the per-user rates, shape (K,).
    """
    from ..baselines.ideal import rzf, zf
    from ..metrics.se import mu_rate

    K, N3, P, v = directions.shape
    W = np.zeros((K, N3, P, v), dtype=complex)
    for t in range(N3):
        # stack the K*v reported directions, row k*v + j = user k, layer j
        D = np.transpose(directions[:, t].conj(), (0, 2, 1)).reshape(K * v, P)
        F = zf(D) if xi is None else rzf(D, xi)  # (P, K*v)
        F = F / np.linalg.norm(F, axis=0, keepdims=True)
        W[:, t] = F.T.reshape(K, v, P).transpose(0, 2, 1)  # (K, P, v)
    return mu_rate(H_users, W, rho / (K * v))  # equal power split across streams
