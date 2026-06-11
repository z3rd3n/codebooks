"""Spectral efficiency metrics (paper eqs. mu-re-sig / ar-per-ue).

Power convention: noise power 1, total transmit power per frequency unit
``rho`` (linear SNR).  Precoders follow the framework convention
tr(W^H W) = 1, so ``sqrt(rho) W`` carries the full transmit power and the
spec's equal power split across layers is already encoded in W.
"""

from __future__ import annotations

import numpy as np


def su_rate(H: np.ndarray, W: np.ndarray, rho: float) -> float:
    """Mean single-user rate over all leading axes (slots, frequency units).

    H: (..., Nr, P), W: (..., P, v)  ->  mean of log2 det(I + rho (HW)(HW)^H)
    in bits/s/Hz, treating inter-layer interference exactly (joint decoding
    across the v layers of one user).
    """
    H, W = np.asarray(H), np.asarray(W)
    HW = H @ W  # (..., Nr, v)
    v = W.shape[-1]
    G = np.swapaxes(HW, -2, -1).conj() @ HW  # (..., v, v)
    eye = np.eye(v)
    rates = np.log2(np.linalg.det(eye + rho * G).real)
    return float(np.mean(rates))


def mu_rate(H: np.ndarray, W: np.ndarray, rho: float) -> np.ndarray:
    """Per-user rates with inter-user interference (paper eq. ar-per-ue).

    H: (K, ..., Nr, P) channels of K users; W: (K, ..., P, v) their precoders
    (each with tr(W_k^H W_k) = 1, i.e. equal power split among users is the
    caller's choice of rho).  Returns the mean rate per user, shape (K,).
    """
    H, W = np.asarray(H), np.asarray(W)
    K = H.shape[0]
    Nr = H.shape[-2]
    eye = np.eye(Nr)
    rates = np.zeros(K)
    for k in range(K):
        Hk = H[k]
        Sk = Hk @ W[k]  # (..., Nr, v)
        signal = rho * (Sk @ np.swapaxes(Sk, -2, -1).conj())
        interf = np.zeros_like(signal)
        for i in range(K):
            if i == k:
                continue
            Si = Hk @ W[i]
            interf = interf + rho * (Si @ np.swapaxes(Si, -2, -1).conj())
        Q = np.linalg.inv(eye + interf)
        rates[k] = np.mean(np.log2(np.linalg.det(eye + Q @ signal).real))
    return rates
