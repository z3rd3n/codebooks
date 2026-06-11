"""Amplitude/phase quantization tables of the R15-R18 Type II codebooks.

Transcribed from the paper's tables:

* ``R15_WB_AMP``  -- Table tabk1:  k^(1) in 0..7  -> p^(1)   (3-bit wideband)
* ``R15_SB_AMP``  -- Table tabk2:  k^(2) in {0,1} -> p^(2)   (1-bit subband)
* ``R16_REF_AMP`` -- Table tabmapkuan: k^(1) in 1..15 -> p^(1) = 2^(-(15-k)/4)
                     (4-bit per-polarization reference amplitude; k=0 reserved)
* ``R16_DIFF_AMP``-- Table tabmapzhai: k^(2) in 0..7 -> p^(2) = 2^(-(7-k)/2)
                     (3-bit differential amplitude)

Phase: phi = exp(j*2*pi*c / N_PSK), N_PSK in {4, 8} for R15 and 16 for R16+.
"""

from __future__ import annotations

import numpy as np

R15_WB_AMP = np.array([0.0] + [np.sqrt(2.0 ** -(7 - k)) for k in range(1, 8)])
# = [0, sqrt(1/64), sqrt(1/32), sqrt(1/16), sqrt(1/8), sqrt(1/4), sqrt(1/2), 1]

R15_SB_AMP = np.array([np.sqrt(0.5), 1.0])

# k = 0 is "reserved" in the spec; index it as NaN so accidental use is loud.
R16_REF_AMP = np.array([np.nan] + [2.0 ** (-(15 - k) / 4) for k in range(1, 16)])

R16_DIFF_AMP = np.array([2.0 ** (-(7 - k) / 2) for k in range(8)])


def quantize_amplitude(value: float | np.ndarray, table: np.ndarray) -> np.ndarray:
    """Nearest-neighbor amplitude quantization; returns integer indices."""
    vals = np.nan_to_num(table, nan=-np.inf)  # never select reserved entries
    value = np.asarray(value)
    return np.abs(value[..., None] - vals).argmin(axis=-1)


def quantize_phase(angle: float | np.ndarray, n_psk: int) -> np.ndarray:
    """Quantize an angle (radians) to c in {0..N_PSK-1}, phi = exp(j*2*pi*c/N_PSK)."""
    c = np.round(np.asarray(angle) * n_psk / (2 * np.pi)).astype(int)
    return np.mod(c, n_psk)


def phase_value(c: int | np.ndarray, n_psk: int) -> np.ndarray:
    """phi_c = exp(j*2*pi*c / N_PSK)."""
    return np.exp(2j * np.pi * np.asarray(c) / n_psk)
