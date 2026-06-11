"""DFT bases used by the 3GPP codebooks.

Spatial basis (paper eq. vmm / beamv):
    v_{m1,m2} = a_{m1} (x) u_{m2}        with
    a_{m1}[k] = exp(j*2*pi*m1*k / (O1*N1)),  k = 0..N1-1   (horizontal)
    u_{m2}[k] = exp(j*2*pi*m2*k / (O2*N2)),  k = 0..N2-1   (vertical)

so a beam vector has length N1*N2 with the vertical index running fastest,
and squared norm N1*N2 (unit-modulus entries).

Frequency basis (eq. b46):  y_f[t]   = exp(j*2*pi*t*n3 / N3),  t = 0..N3-1
Temporal basis (eq. b88):   z_tau[i] = exp(j*2*pi*i*n4 / N4),  i = 0..N4-1
"""

from __future__ import annotations

import numpy as np

from ..config import AntennaConfig


def steering(N: int, O: int, idx: int | np.ndarray) -> np.ndarray:
    """Oversampled 1D DFT steering vector(s) of length N for grid index ``idx``.

    Returns shape (N,) for scalar idx, else (len(idx), N).
    """
    idx_arr = np.atleast_1d(np.asarray(idx))
    k = np.arange(N)
    vecs = np.exp(2j * np.pi * np.outer(idx_arr, k) / (O * N))
    return vecs[0] if np.isscalar(idx) or np.ndim(idx) == 0 else vecs


def spatial_beam(cfg: AntennaConfig, m1: int, m2: int) -> np.ndarray:
    """Single-polarization beam v_{m1,m2} of length N1*N2 (vertical fastest)."""
    a = steering(cfg.N1, cfg.O1, m1)
    u = steering(cfg.N2, cfg.O2, m2)
    return np.kron(a, u)


def spatial_grid(cfg: AntennaConfig) -> np.ndarray:
    """All oversampled beams, shape (N1*O1, N2*O2, N1*N2)."""
    G1, G2 = cfg.n_beams
    grid = np.empty((G1, G2, cfg.n_ports_per_pol), dtype=complex)
    a = steering(cfg.N1, cfg.O1, np.arange(G1))  # (G1, N1)
    u = steering(cfg.N2, cfg.O2, np.arange(G2))  # (G2, N2)
    for m1 in range(G1):
        for m2 in range(G2):
            grid[m1, m2] = np.kron(a[m1], u[m2])
    return grid


def beam_index(cfg: AntennaConfig, q1: int, q2: int, n1: int | np.ndarray, n2: int | np.ndarray):
    """Orthogonal-group indexing m = O*n + q (paper eq. a32)."""
    return cfg.O1 * np.asarray(n1) + q1, cfg.O2 * np.asarray(n2) + q2


def orthogonal_group(cfg: AntennaConfig, q1: int, q2: int) -> np.ndarray:
    """The N1*N2 orthogonal beams of group (q1, q2), shape (N1*N2 beams, N1*N2 ports).

    Row order: n = n2 * N1 + n1 (matching Algorithm 1's n1 = n mod N1 convention).
    """
    beams = np.empty((cfg.N1 * cfg.N2, cfg.n_ports_per_pol), dtype=complex)
    for n in range(cfg.N1 * cfg.N2):
        n1 = n % cfg.N1
        n2 = n // cfg.N1
        m1, m2 = cfg.O1 * n1 + q1, cfg.O2 * n2 + q2
        beams[n] = spatial_beam(cfg, m1, m2)
    return beams


def freq_basis(N3: int, n3: int | np.ndarray) -> np.ndarray:
    """Frequency-domain (delay-tap) DFT basis y over t = 0..N3-1.

    Returns shape (N3,) for scalar n3, else (len(n3), N3).
    """
    n3_arr = np.atleast_1d(np.asarray(n3))
    t = np.arange(N3)
    vecs = np.exp(2j * np.pi * np.outer(n3_arr, t) / N3)
    return vecs[0] if np.isscalar(n3) or np.ndim(n3) == 0 else vecs


def time_basis(N4: int, n4: int | np.ndarray) -> np.ndarray:
    """Temporal (Doppler-shift) DFT basis z over iota = 0..N4-1."""
    return freq_basis(N4, n4)
