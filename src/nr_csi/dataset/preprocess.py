"""Preprocessing for raw-H CSI datasets: the angular-delay representation,
power normalization, and a train-time AWGN helper.

The stored ground truth is the raw complex channel ``H[..., n_rx, P, n_freq]``
(antenna x subcarrier image per RX).  ML CSI-feedback networks (CsiNet and
successors) operate on its *angular-delay* transform, where energy concentrates
in a few early delay taps so the matrix is sparse and compressible:

* **delay**  -- inverse DFT across the ``n_freq`` axis (the same freq->delay
  transform as :func:`nr_csi.channel.diagnostics.power_delay_profile`), then
  keep the first ``n_delay`` taps.
* **angular** -- a unitary per-polarization 2D DFT across the ``(N1, N2)`` port
  grid (port ordering ``pol*N1*N2 + n1*N2 + n2``, vertical ``n2`` fastest, as
  produced by the Sionna adapter and assumed by :mod:`nr_csi.utils.dft`).

All transforms here are unitary (``norm="ortho"``); with ``n_delay == n_freq``
they round-trip exactly (see the tests), so the dataset stays lossless and the
truncation length is a *training-time* choice, not baked into storage.
"""

from __future__ import annotations

import numpy as np

from ..config import AntennaConfig


def power_normalize(
    H: np.ndarray, axes: tuple[int, ...] = (-3, -2, -1)
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample power normalization.

    Returns ``(H / ||H||, ||H||)`` where the Frobenius norm is taken over
    ``axes`` (default the ``(rx, port, freq)`` channel axes).  The norm is
    returned with those axes kept (size 1) so it broadcasts; squeeze it for
    per-sample storage.  Separating the scale from the normalized channel is the
    standard "power-spherical" trick that keeps weak channels visible to the net.
    """
    norm = np.sqrt(np.sum(np.abs(H) ** 2, axis=axes, keepdims=True))
    safe = np.where(norm == 0, 1.0, norm)
    return H / safe, norm


def to_delay(H: np.ndarray, n_delay: int | None = None) -> np.ndarray:
    """Frequency -> delay (unitary IDFT on the last axis), truncated to
    ``n_delay`` taps (default: keep all ``n_freq``)."""
    Hd = np.fft.ifft(H, axis=-1, norm="ortho")
    if n_delay is not None:
        if n_delay > H.shape[-1]:
            raise ValueError(f"n_delay={n_delay} exceeds n_freq={H.shape[-1]}")
        Hd = Hd[..., :n_delay]
    return Hd


def from_delay(H_delay: np.ndarray, n_freq: int) -> np.ndarray:
    """Inverse of :func:`to_delay`: zero-pad to ``n_freq`` taps, then DFT back.

    Exact only when ``to_delay`` was called without truncation; otherwise the
    truncated taps are treated as zero (the lossy CsiNet assumption)."""
    n_delay = H_delay.shape[-1]
    if n_delay < n_freq:
        pad = list(H_delay.shape)
        pad[-1] = n_freq - n_delay
        H_delay = np.concatenate([H_delay, np.zeros(pad, dtype=H_delay.dtype)], axis=-1)
    return np.fft.fft(H_delay, axis=-1, norm="ortho")


def spatial_dft(H: np.ndarray, antenna: AntennaConfig, inverse: bool = False) -> np.ndarray:
    """Unitary per-polarization 2D DFT across the port (antenna) axis ``-2``.

    ``H[..., n_rx, P, n_freq]`` with ``P = 2*N1*N2`` -> same shape in the angular
    (beam) domain.  ``inverse=True`` undoes it."""
    N1, N2, P = antenna.N1, antenna.N2, antenna.P
    if H.shape[-2] != P:
        raise ValueError(f"port axis is {H.shape[-2]}, expected P={P}")
    lead = H.shape[:-2]
    n_freq = H.shape[-1]
    G = H.reshape(*lead, 2, N1, N2, n_freq)  # split pol / horizontal / vertical
    func = np.fft.ifft if inverse else np.fft.fft
    G = func(G, axis=-3, norm="ortho")  # N1 (horizontal) axis
    G = func(G, axis=-2, norm="ortho")  # N2 (vertical) axis
    return G.reshape(*lead, P, n_freq)


def to_angular_delay(
    H: np.ndarray, antenna: AntennaConfig, n_delay: int | None = None
) -> np.ndarray:
    """CsiNet input: spatial 2D DFT then freq->delay, truncated to ``n_delay``."""
    return to_delay(spatial_dft(H, antenna), n_delay=n_delay)


def from_angular_delay(H_ad: np.ndarray, antenna: AntennaConfig, n_freq: int) -> np.ndarray:
    """Inverse of :func:`to_angular_delay` (exact iff no delay truncation)."""
    return spatial_dft(from_delay(H_ad, n_freq), antenna, inverse=True)


def stack_real_imag(H: np.ndarray) -> np.ndarray:
    """Complex ``(...)`` -> real ``(2, ...)`` (channel-first, CNN-friendly)."""
    return np.stack([H.real, H.imag], axis=0).astype(np.float32)


def complex_from_real_imag(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`stack_real_imag`."""
    return x[0] + 1j * x[1]


def apply_awgn(H: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN at ``snr_db`` (per-entry), matching the eval harness'
    noise convention.  Use as a train-time augmentation on the clean stored H."""
    sigma = np.sqrt(np.mean(np.abs(H) ** 2) / 10 ** (snr_db / 10))
    noise = rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape)
    return H + sigma * noise / np.sqrt(2)
