"""Shared UE-side helpers: target precoders and spatial basis selection.

These are implementation choices (the UE algorithm is not standardized);
they follow the procedure the paper describes for its Fig. f1 evaluation:
per-subband eigen targets, strongest orthogonal beam group, top-L bases.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import AntennaConfig
from ..utils import combinatorics as cb
from ..utils import dft


def aligned_eigen_targets(H: np.ndarray, rank: int) -> np.ndarray:
    """Per-frequency-unit dominant eigenvectors with phases aligned along t.

    H: (N3, Nr, P) -> targets (N3, P, v), unit-norm columns.  Sequential
    phase alignment removes the per-subband phase ambiguity of the SVD so
    that delay-domain (DFT) compression of the targets is meaningful.
    """
    from ..baselines.ideal import eigen_precoder

    n_rx = H.shape[-2]
    if rank > min(n_rx, H.shape[-1]):
        raise ValueError(f"rank {rank} exceeds channel rank bound min(Nr={n_rx}, P)")
    targets = eigen_precoder(H, rank=rank) * np.sqrt(rank)  # unit columns
    for t in range(1, targets.shape[0]):
        inner = np.sum(targets[t - 1].conj() * targets[t], axis=0)  # (v,)
        targets[t] *= np.exp(-1j * np.angle(np.where(inner == 0, 1.0, inner)))
    return targets


def split_polarizations(targets: np.ndarray) -> np.ndarray:
    """(N3, P, v) -> (N3, pol, P/2, v)."""
    half = targets.shape[1] // 2
    return np.stack([targets[:, :half, :], targets[:, half:, :]], axis=1)


def select_group_and_beams(
    antenna: AntennaConfig, targets: np.ndarray, L: int
) -> tuple[int, int, int]:
    """Pick (q1, q2, i12): the orthogonal group whose best L beams capture the
    most target energy, and the combinatorial index of those beams."""
    pols = split_polarizations(targets)
    best = (-1.0, None)
    for q1 in range(antenna.O1):
        for q2 in range(antenna.O2):
            Bg = dft.orthogonal_group(antenna, q1, q2)  # (N1N2, P/2)
            proj = np.einsum("bp,nopv->nvob", Bg.conj(), pols)
            be = np.sum(np.abs(proj) ** 2, axis=(0, 1, 2))
            e = float(np.sort(be)[::-1][:L].sum())
            if e > best[0]:
                best = (e, (q1, q2, be))
    q1, q2, beam_energy = best[1]
    top = np.sort(np.argsort(beam_energy)[::-1][:L])
    n1 = [int(n % antenna.N1) for n in top]
    n2 = [int(n // antenna.N1) for n in top]
    return q1, q2, cb.encode_beam_combination(n1, n2, antenna.N1, antenna.N2)


def select_ps_initial(antenna: AntennaConfig, targets: np.ndarray, L: int, d: int) -> int:
    """Pick i_{1,1} for consecutive-port selection (R15/R16 PS variants)."""
    half = antenna.P // 2
    energy = np.sum(np.abs(targets) ** 2, axis=(0, 2))
    port_energy = energy[:half] + energy[half:]
    n_init = math.ceil(half / d)
    scores = [
        sum(port_energy[(j * d + i) % half] for i in range(L)) for j in range(n_init)
    ]
    return int(np.argmax(scores))


def basis_regular(antenna: AntennaConfig, q1: int, q2: int, i12: int, L: int) -> np.ndarray:
    """Selected DFT beams, shape (L, P/2)."""
    n1, n2 = cb.decode_beam_combination(i12, antenna.N1, antenna.N2, L)
    return np.stack(
        [
            dft.spatial_beam(antenna, antenna.O1 * a + q1, antenna.O2 * b + q2)
            for a, b in zip(n1, n2)
        ]
    )


def basis_ps(antenna: AntennaConfig, i11: int, L: int, d: int) -> np.ndarray:
    """Consecutive-port standard basis vectors, shape (L, P/2)."""
    half = antenna.P // 2
    B = np.zeros((L, half))
    for i in range(L):
        B[i, (i11 * d + i) % half] = 1.0
    return B


def ls_coefficients(B: np.ndarray, targets: np.ndarray, scale: float) -> np.ndarray:
    """Least-squares combination coefficients on both polarizations.

    B: (L, P/2) orthogonal-up-to-`scale` bases; targets (N3, P, v).
    Returns (v, N3, 2L).
    """
    half = targets.shape[1] // 2
    L = B.shape[0]
    N3, _, v = targets.shape
    coeff = np.empty((v, N3, 2 * L), dtype=complex)
    proj_a = np.einsum("bp,npv->vnb", B.conj(), targets[:, :half, :]) / scale
    proj_b = np.einsum("bp,npv->vnb", B.conj(), targets[:, half:, :]) / scale
    coeff[:, :, :L] = proj_a
    coeff[:, :, L:] = proj_b
    return coeff
