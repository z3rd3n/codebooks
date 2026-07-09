"""Fixed scalar quantizer for GLIMPSE measurements.

Because the projected measurements are near-Gaussian for any channel
distribution (see :mod:`.projection`), the spec can freeze ONE scalar
quantizer: the B-bit Lloyd-Max quantizer of the standard normal, applied
independently to the real and imaginary parts of each (unit-RMS-normalized)
measurement.  Levels are computed deterministically at import time by Lloyd
iteration on the exact Gaussian conditional means -- these are constants a
standard would tabulate, not tuned parameters.

``pack_indices``/``unpack_indices`` produce the actual '0'/'1' feedback
bitstream in the style of :mod:`nr_csi.codebooks.serialize`, so
``len(pack(...)) == overhead bits`` is enforced by construction.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.stats import norm


@functools.lru_cache(maxsize=None)
def lloyd_max(bits: int, iterations: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max quantizer of N(0, 1): ``(levels[2^B], boundaries[2^B - 1])``.

    Deterministic fixed-point iteration: boundaries are level midpoints,
    levels are exact Gaussian conditional means within each cell.
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    n = 1 << bits
    # start from equiprobable-cell centroids
    edges = norm.ppf(np.linspace(0.0, 1.0, n + 1))
    levels = _cell_means(edges)
    for _ in range(iterations):
        edges = np.concatenate(([-np.inf], (levels[:-1] + levels[1:]) / 2, [np.inf]))
        levels = _cell_means(edges)
    return levels, (levels[:-1] + levels[1:]) / 2


def _cell_means(edges: np.ndarray) -> np.ndarray:
    """E[X | a < X < b] for consecutive edge pairs, X ~ N(0,1)."""
    pdf = norm.pdf(edges)
    cdf = norm.cdf(edges)
    num = pdf[:-1] - pdf[1:]
    den = np.maximum(cdf[1:] - cdf[:-1], 1e-300)
    return num / den


def quantizer_mse(bits: int) -> float:
    """Mean squared distortion of the B-bit Lloyd-Max quantizer on N(0,1)."""
    levels, bounds = lloyd_max(bits)
    edges = np.concatenate(([-np.inf], bounds, [np.inf]))
    pdf = norm.pdf(edges)
    cdf = norm.cdf(edges)
    p = cdf[1:] - cdf[:-1]
    ex = pdf[:-1] - pdf[1:]  # E[X; cell]
    # E[(X-l)^2; cell] = p*(1+l^2) - 2*l*E[X; cell]  (E[X^2; cell] = p + [a pdf(a) - b pdf(b)])
    a, b = edges[:-1], edges[1:]
    # x*pdf(x) -> 0 as x -> +-inf; guard the inf*0 = nan at the outer edges
    apa = np.where(np.isfinite(a), np.where(np.isfinite(a), a, 0.0) * pdf[:-1], 0.0)
    bpb = np.where(np.isfinite(b), np.where(np.isfinite(b), b, 0.0) * pdf[1:], 0.0)
    ex2 = p + apa - bpb
    return float(np.sum(ex2 - 2 * levels * ex + levels**2 * p))


def allocate_bits(
    sigma: np.ndarray, m: int, total_bits: int, max_bits: int = 7
) -> np.ndarray:
    """Reverse water-filling bit allocation over the ``2m`` real dimensions.

    The ``k``-th KLT coordinate has population variance ``sigma_k^2``; its real
    and imaginary parts are each ~N(0, sigma_k^2/2).  Optimal transform coding
    (Gersho-Gray) allocates bits ``b_j = max(0, 1/2 log2(v_j / theta))`` to
    real dimension ``j`` of variance ``v_j``, with ``theta`` chosen so
    ``sum_j b_j = total_bits`` -- so high-variance (early) coordinates get more
    bits and negligible ones get none.  Returns an integer vector of length
    ``2m`` (real, imag interleaved per coordinate), clipped to ``[0, max_bits]``
    and rounded to sum to ``total_bits`` (largest-fractional-part rounding).

    This is a *published constant* function of the standardization variances,
    computed identically at the UE and the gNB -- no side information is sent.
    """
    v = np.repeat(np.asarray(sigma, float)[:m] ** 2, 2) / 2.0  # (2m,) real-dim var
    v = np.maximum(v, 1e-30)
    lo, hi = 1e-30, float(v.max())
    for _ in range(64):  # bisection on the water level theta
        theta = (lo + hi) / 2
        b = np.maximum(0.5 * np.log2(v / theta), 0.0)
        if b.sum() > total_bits:
            lo = theta
        else:
            hi = theta
    b = np.maximum(0.5 * np.log2(v / ((lo + hi) / 2)), 0.0)
    bi = np.floor(np.clip(b, 0, max_bits)).astype(int)
    rem = int(total_bits - bi.sum())
    if rem > 0:  # hand out the remaining bits by largest fractional part
        headroom = bi < max_bits
        order = np.argsort(-(b - bi) * headroom)
        for j in order[:rem]:
            if bi[j] < max_bits:
                bi[j] += 1
    return bi.astype(np.uint8)


def quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """Real array -> level indices (uint8), nearest-level rule."""
    if bits <= 0:
        return np.zeros_like(np.asarray(x), dtype=np.uint8)
    _, bounds = lloyd_max(bits)
    return np.searchsorted(bounds, np.asarray(x, dtype=float)).astype(np.uint8)


def dequantize(idx: np.ndarray, bits: int) -> np.ndarray:
    levels, _ = lloyd_max(bits)
    return levels[np.asarray(idx, dtype=int)]


def quantize_complex(y: np.ndarray, bits: int) -> np.ndarray:
    """Complex ``(..., m)`` -> indices ``(..., 2m)`` (Re block then Im block)."""
    return np.concatenate([quantize(y.real, bits), quantize(y.imag, bits)], axis=-1)


def dequantize_complex(idx: np.ndarray, bits: int) -> np.ndarray:
    m = idx.shape[-1] // 2
    x = dequantize(idx, bits)
    return x[..., :m] + 1j * x[..., m:]


def pack_indices(idx: np.ndarray, bits: int) -> str:
    """Indices -> feedback bitstream ('0'/'1' string, ``len == idx.size * bits``)."""
    return "".join(format(int(i), f"0{bits}b") for i in np.asarray(idx).reshape(-1))


def unpack_indices(bitstream: str, bits: int) -> np.ndarray:
    if len(bitstream) % bits:
        raise ValueError(f"bitstream length {len(bitstream)} not a multiple of {bits}")
    vals = [int(bitstream[i : i + bits], 2) for i in range(0, len(bitstream), bits)]
    return np.asarray(vals, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Water-filled (variable per-coordinate) quantization
# --------------------------------------------------------------------------- #
def _interleave_ri(u: np.ndarray) -> np.ndarray:
    """Complex ``(m,)`` -> real ``(2m,)`` interleaved [Re0, Im0, Re1, Im1, ...],
    matching the ordering of :func:`allocate_bits`."""
    out = np.empty(2 * u.shape[-1], float)
    out[0::2] = u.real
    out[1::2] = u.imag
    return out


def _deinterleave_ri(x: np.ndarray) -> np.ndarray:
    return x[0::2] + 1j * x[1::2]


def quantize_alloc(u: np.ndarray, bits_vec: np.ndarray) -> str:
    """Quantize one standardized complex report ``u[m]`` under a per-real-dim
    bit-allocation ``bits_vec[2m]`` -> feedback bitstream.  Coordinates with 0
    allocated bits contribute nothing (reconstructed as the mean, 0)."""
    x = _interleave_ri(np.asarray(u))
    parts = []
    for xj, bj in zip(x, np.asarray(bits_vec, int)):
        if bj > 0:
            parts.append(format(int(quantize(np.array([xj]), bj)[0]), f"0{bj}b"))
    return "".join(parts)


def dequantize_alloc(bitstream: str, bits_vec: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_alloc`: bitstream + allocation -> ``u_hat[m]``."""
    bits_vec = np.asarray(bits_vec, int)
    x = np.zeros(len(bits_vec), float)
    pos = 0
    for j, bj in enumerate(bits_vec):
        if bj > 0:
            idx = int(bitstream[pos : pos + bj], 2)
            x[j] = dequantize(np.array([idx]), bj)[0]
            pos += bj
    if pos != len(bitstream):
        raise ValueError(f"bitstream length {len(bitstream)} != allocated {pos} bits")
    return _deinterleave_ri(x)
