"""Reproduction of the paper's Fig. f1 experiment (single-polarization study).

The paper isolates *spatial* compression: a ULA with N antennas and a single
polarization, single or dual stream.  Ideal feedback = dominant eigenvector;
Type I = best beam of the oversampled DFT grid; Type II = best orthogonal
group, OMP selection of L=4 bases, least-squares weights, 3-bit wideband
amplitude (subband amplitude disabled).

Erratum handled here: the paper says "phase quantized to 3 bits, i.e.,
N_PSK = 4" -- contradictory (N_PSK=4 is 2 bits).  ``n_psk`` is configurable;
8 (= 3 bits) is the default used for the reproduction.

The channel is not specified in the paper; we use a sparse multipath ULA
channel with unit total path power so that E||h||^2 = N, which matches the
upper-bound curve of the figure (log2(1 + rho*N) at high SNR).

The paper also does not describe its 2-stream Type I procedure (its text
focuses on single-stream).  We restrict the second beam to the i_{1,3}
offsets of Table tabmap (spec-faithful); an unrestricted grid search would
land closer to the paper's 2-stream Type I curve, bracketing it from above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import quantization as qt


def ula_steering(N: int, sin_theta: float | np.ndarray) -> np.ndarray:
    """Half-wavelength ULA response, e^{j*pi*k*sin(theta)}."""
    k = np.arange(N)
    return np.exp(1j * np.pi * np.outer(np.atleast_1d(sin_theta), k))


def multipath_channel(
    N: int, n_rx: int, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """H (n_rx, N) = sum of n_paths rays, unit total power: E||H||_F^2 = n_rx*N."""
    sin_aod = rng.uniform(-1, 1, n_paths)
    gains = (rng.standard_normal(n_paths) + 1j * rng.standard_normal(n_paths)) / np.sqrt(
        2 * n_paths
    )
    A = ula_steering(N, sin_aod)  # (n_paths, N)
    rx = (rng.standard_normal((n_rx, n_paths)) + 1j * rng.standard_normal((n_rx, n_paths))) / np.sqrt(2)
    if n_rx == 1:
        rx = np.ones((1, n_paths))
    return (rx * gains) @ A.conj()


def dft_grid(N: int, O: int) -> np.ndarray:
    """Oversampled DFT beam grid, shape (N*O, N), unit-norm rows."""
    idx = np.arange(N * O)
    k = np.arange(N)
    return np.exp(2j * np.pi * np.outer(idx, k) / (N * O)) / np.sqrt(N)


def type1_precoder(H: np.ndarray, grid: np.ndarray, n_streams: int) -> np.ndarray:
    """Best beam(s) of the grid by received power ||H w||^2; greedy for 2 streams."""
    energy = np.sum(np.abs(H @ grid.T) ** 2, axis=0)  # w = grid[i], not conjugated
    if n_streams == 1:
        return grid[int(np.argmax(energy))][:, None]
    first = int(np.argmax(energy))
    # rank-2 Type I: the second beam is restricted to the i_{1,3} offsets
    # {O, 2O, 3O} of Table tabmap (N2=1 row); choose by the joint rate metric
    n_beams = grid.shape[0]
    O = n_beams // (grid.shape[1])
    rho = 100.0
    best = (-np.inf, None)
    for k in (1, 2, 3):
        i = (first + k * O) % n_beams
        W = np.stack([grid[first], grid[i]], axis=1) / np.sqrt(2)
        G = (H @ W).conj().T @ (H @ W)
        metric = np.log2(np.linalg.det(np.eye(2) + rho * G).real)
        if metric > best[0]:
            best = (metric, W)
    return best[1]


def type2_precoder(
    H: np.ndarray,
    N: int,
    O: int,
    L: int,
    n_streams: int,
    n_psk: int = 8,
) -> np.ndarray:
    """Paper's Type II procedure: group selection, OMP, LS, quantization."""
    targets = np.linalg.svd(H, full_matrices=False)[2].conj().T[:, :n_streams]  # (N, v)
    # 1) orthogonal group with the most energy captured by its best L beams
    best = (-1.0, 0)
    for q in range(O):
        Bg = ula_group(N, O, q)
        e = np.sort(np.sum(np.abs(Bg.conj() @ targets) ** 2, axis=1))[::-1][:L].sum()
        if e > best[0]:
            best = (float(e), q)
    Bg = ula_group(N, O, best[1])  # (N beams, N)
    W = np.zeros((N, n_streams), dtype=complex)
    for s in range(n_streams):
        t = targets[:, s]
        # 2) OMP over the orthogonal group (orthogonal bases: pick top-L)
        proj = Bg.conj() @ t  # (N,)
        sel = np.sort(np.argsort(np.abs(proj))[::-1][:L])
        c = proj[sel]  # LS solution on orthonormal bases
        # 3) quantize: 3-bit wideband amplitude, n_psk phase, strongest = (1, 0)
        i_star = int(np.argmax(np.abs(c)))
        c = c / c[i_star]
        amp = qt.R15_WB_AMP[qt.quantize_amplitude(np.minimum(np.abs(c), 1.0), qt.R15_WB_AMP)]
        phase = qt.phase_value(qt.quantize_phase(np.angle(c), n_psk), n_psk)
        cq = amp * phase
        w = Bg[sel].T @ cq
        W[:, s] = w / np.linalg.norm(w)
    return W / np.sqrt(n_streams)


def ula_group(N: int, O: int, q: int) -> np.ndarray:
    """Orthogonal beam group q of the oversampled ULA grid, unit-norm rows."""
    idx = O * np.arange(N) + q
    k = np.arange(N)
    return np.exp(2j * np.pi * np.outer(idx, k) / (N * O)) / np.sqrt(N)


@dataclass
class F1Curves:
    snr_db: np.ndarray
    upper_bound: np.ndarray
    type1: np.ndarray
    type2: np.ndarray


def run_f1_case(
    N: int,
    n_streams: int,
    snr_db: np.ndarray,
    n_drops: int = 300,
    n_paths: int = 4,
    O: int = 4,
    L: int = 4,
    n_psk: int = 8,
    seed: int = 0,
) -> F1Curves:
    rng = np.random.default_rng(seed)
    rhos = 10 ** (snr_db / 10)
    acc = {k: np.zeros(len(rhos)) for k in ("ub", "t1", "t2")}
    grid = dft_grid(N, O)
    n_rx = max(n_streams, 1)
    for _ in range(n_drops):
        H = multipath_channel(N, n_rx, n_paths, rng)
        U, S, Vh = np.linalg.svd(H, full_matrices=False)
        W_ub = Vh.conj().T[:, :n_streams] / np.sqrt(n_streams)
        W_t1 = type1_precoder(H, grid, n_streams)
        W_t2 = type2_precoder(H, N, O, L, n_streams, n_psk)
        for i, rho in enumerate(rhos):
            for key, W in (("ub", W_ub), ("t1", W_t1), ("t2", W_t2)):
                G = (H @ W).conj().T @ (H @ W)
                acc[key][i] += np.log2(
                    np.linalg.det(np.eye(n_streams) + rho * G).real
                )
    return F1Curves(
        snr_db=snr_db,
        upper_bound=acc["ub"] / n_drops,
        type1=acc["t1"] / n_drops,
        type2=acc["t2"] / n_drops,
    )
