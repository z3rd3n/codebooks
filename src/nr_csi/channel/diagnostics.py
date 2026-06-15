"""Codebook-independent channel diagnostics on the PMI grid.

Every metric here is a pure-NumPy function of a channel tensor

    H[..., N3, rx, port]   (typically (slot, N3, rx, port), or a stack of drops)

i.e. the channel exactly as the codebooks see it (after the CDL adapter has
averaged subcarriers into ``N3`` PMI frequency units).  They characterise the
three physical axes the codebooks compress:

* delay / frequency  -> :func:`power_delay_profile`, :func:`rms_delay_spread`,
  :func:`freq_correlation`
* time / Doppler      -> :func:`time_correlation`
* space / MIMO rank   -> :func:`singular_value_spectrum`

Because the synthetic ``SyntheticRayChannel`` places rays at *known* delays,
Dopplers, and angles, these functions have closed-form expected values there
(a ray at delay ``d`` taps is a PDP impulse at tap ``d``), which is what the
unit tests assert -- validating both the diagnostics and the channel's
frequency/time wiring.
"""

from __future__ import annotations

import numpy as np

_FREQ_AXIS = -3  # H[..., N3, rx, port]: the N3 (frequency-unit) axis
_SLOT_AXIS = -4  # H[slot, N3, rx, port]: the slot (time) axis


def _other_axes(ndim: int, keep: int) -> tuple[int, ...]:
    keep %= ndim
    return tuple(i for i in range(ndim) if i != keep)


def power_delay_profile(H: np.ndarray) -> np.ndarray:
    """Average power per delay tap (length ``N3``), normalised to sum 1.

    The impulse response is the inverse DFT of ``H`` along the frequency-unit
    axis; for a causal, well-resolved physical channel (CDL) the energy sits at
    low taps.  Note the framework's synthetic rays use the conjugate
    (fft-not-ifft) tap convention, so a synthetic ray at delay ``d`` appears at
    circular tap ``(N3 - d) % N3`` here -- the delay *spread* (below) is
    invariant to this reversal.
    """
    cir = np.fft.ifft(H, axis=_FREQ_AXIS)
    power = np.abs(cir) ** 2
    pdp = power.mean(axis=_other_axes(H.ndim, _FREQ_AXIS))
    total = pdp.sum()
    return pdp / total if total > 0 else pdp


def rms_delay_spread(H: np.ndarray) -> float:
    """RMS delay spread in tap units, computed with circular statistics.

    Linear moments of a power-delay profile are wrong when energy wraps the
    circular tap axis (and ambiguous under the fft/ifft sign convention); the
    circular standard deviation is reversal- and shift-invariant and matches the
    ordinary std for compact, low-delay profiles.  Zero for a single tap.
    """
    pdp = power_delay_profile(H)
    n = pdp.size
    theta = 2 * np.pi * np.arange(n) / n
    r = abs((pdp * np.exp(1j * theta)).sum())  # mean resultant length
    if r >= 1.0 - 1e-15:
        return 0.0
    if r < 1e-12:  # ~uniform power across taps
        return float(n / np.sqrt(12))
    return float(np.sqrt(-2.0 * np.log(r)) * n / (2 * np.pi))


def taps_to_seconds(n_taps: float, fft_size: int, subcarrier_spacing: float) -> float:
    """Convert a delay in PMI-grid taps to seconds.

    The ``N3`` units span the OFDM band ``fft_size * subcarrier_spacing``, so
    one tap of the ``N3``-point inverse DFT is ``1 / (fft_size * scs)`` seconds.
    """
    return n_taps / (fft_size * subcarrier_spacing)


def _lag_correlation(X: np.ndarray) -> np.ndarray:
    """|rho(lag)| for lag = 0..L-1 averaged over rows, X shape (R, L)."""
    L = X.shape[-1]
    power = np.mean(np.abs(X) ** 2)
    if power == 0:
        return np.ones(L)
    rho = np.empty(L)
    for lag in range(L):
        prod = X[:, : L - lag] * np.conj(X[:, lag:])
        rho[lag] = np.abs(prod.mean()) / power
    return rho


def freq_correlation(H: np.ndarray) -> np.ndarray:
    """Magnitude of the frequency correlation vs lag (length ``N3``).

    ``rho(0) = 1``; it decays faster for larger delay spread.  The lag where it
    crosses 0.5 is a coherence-bandwidth proxy (see :func:`coherence_lag`).
    """
    X = np.moveaxis(H, _FREQ_AXIS, -1).reshape(-1, H.shape[_FREQ_AXIS])
    return _lag_correlation(X)


def time_correlation(H: np.ndarray) -> np.ndarray:
    """Magnitude of the temporal correlation vs slot lag (length ``n_slots``).

    Requires ``H`` to carry a slot axis (``H[slot, N3, rx, port]``).  Decays
    faster at higher UE speed (Doppler); ``rho(0) = 1``.
    """
    if H.ndim < 4 or H.shape[_SLOT_AXIS] < 2:
        return np.ones(1)
    X = np.moveaxis(H, _SLOT_AXIS, -1).reshape(-1, H.shape[_SLOT_AXIS])
    return _lag_correlation(X)


def singular_value_spectrum(H: np.ndarray) -> np.ndarray:
    """Mean normalised *instantaneous* eigenvalue spectrum of ``H[rx, port]``
    (length ``min(rx, port)``), ordered descending and summing to 1 -- the
    per-snapshot MIMO rank as seen by the ``rx`` receive antennas.
    """
    s = np.linalg.svd(H, compute_uv=False)  # (..., min(rx, port))
    e = s ** 2
    e = e / e.sum(axis=-1, keepdims=True)
    return e.reshape(-1, e.shape[-1]).mean(axis=0)


def spatial_covariance_spectrum(H: np.ndarray) -> np.ndarray:
    """Eigenspectrum (length ``port``) of the transmit-side spatial covariance
    ``R = E[h h^H]`` over the CSI-RS ports, accumulated over all realisations
    and ordered descending, summing to 1.

    This is the proper angular-richness / line-of-sight indicator: a near-LoS
    channel (CDL-D/E) concentrates its energy in one spatial direction (a large
    first eigenvalue), while a rich NLoS channel (CDL-A/B/C) spreads it across
    many beams (a flatter spectrum).  Unlike :func:`singular_value_spectrum` it
    averages outer products *before* the eigendecomposition, so it can reveal up
    to ``port`` spatial degrees of freedom regardless of ``n_rx``.
    """
    X = H.reshape(-1, H.shape[-1])  # each row: a port-vector for one (drop, slot, freq, rx)
    cov = (X.conj().T @ X) / X.shape[0]
    ev = np.clip(np.linalg.eigvalsh(cov)[::-1].real, 0.0, None)
    total = ev.sum()
    return ev / total if total > 0 else ev


def coherence_lag(rho: np.ndarray, threshold: float = 0.5) -> float:
    """First lag (fractional, linearly interpolated) where ``rho`` drops below
    ``threshold``; ``len(rho)`` if it never does (fully coherent)."""
    below = np.where(rho < threshold)[0]
    if below.size == 0:
        return float(len(rho))
    k = int(below[0])
    if k == 0:
        return 0.0
    # linear interpolation between lag k-1 (>=thr) and k (<thr)
    hi, lo = rho[k - 1], rho[k]
    frac = (hi - threshold) / (hi - lo) if hi > lo else 0.0
    return float(k - 1 + frac)
