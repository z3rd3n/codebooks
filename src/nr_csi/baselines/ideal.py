"""Full-CSI beamforming baselines (paper Appendix A, Tables SUMIMO/MUMIMO).

``eigen_precoder`` is the per-subband SVD beamformer used as the
"ideal feedback" upper bound in the paper's Fig. f1.
"""

from __future__ import annotations

import numpy as np


def eigen_precoder(H: np.ndarray, rank: int = 1) -> np.ndarray:
    """Per-frequency-unit dominant right singular vectors.

    H: (..., N3, Nr, P)  ->  W: (..., N3, P, rank), unit-norm columns scaled
    by 1/sqrt(rank) so tr(W^H W) = 1 (same convention as the codebooks).
    """
    H = np.asarray(H)
    _, _, Vh = np.linalg.svd(H, full_matrices=False)
    W = np.swapaxes(Vh, -2, -1).conj()[..., :rank]
    return W / np.sqrt(rank)


def svd_precoder(H: np.ndarray, n_streams: int) -> np.ndarray:
    """SVD beamforming for a single (Nr, Nt) channel: W = [V]_{:,1:Ns}."""
    _, _, Vh = np.linalg.svd(H, full_matrices=False)
    return Vh.conj().T[:, :n_streams]


def mrt(H: np.ndarray) -> np.ndarray:
    """Maximum ratio transmission: W = H^H (unnormalized)."""
    return H.conj().T


def zf(H: np.ndarray) -> np.ndarray:
    """Zero forcing: W = H^H (H H^H)^{-1}."""
    return H.conj().T @ np.linalg.inv(H @ H.conj().T)


def rzf(H: np.ndarray, xi: float) -> np.ndarray:
    """Regularized zero forcing: W = H^H (H H^H + xi I)^{-1}."""
    n = H.shape[0]
    return H.conj().T @ np.linalg.inv(H @ H.conj().T + xi * np.eye(n))


def mmse(H: np.ndarray, snr: float) -> np.ndarray:
    """MMSE beamforming: RZF with xi = sigma^2 / P_t = 1/snr."""
    return rzf(H, 1.0 / snr)
