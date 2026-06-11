"""Precoder similarity metrics for codebook / ML-scheme comparison.

SGCS (squared generalized cosine similarity) is the standard metric of the
3GPP AI/ML CSI feedback study: per layer
    SGCS_l = |w_ref^H w_hat|^2 / (||w_ref||^2 ||w_hat||^2),
averaged over layers and all leading axes.  NMSE is reported after the
optimal per-column phase alignment (precoders are phase-ambiguous).
"""

from __future__ import annotations

import numpy as np


def sgcs(W_ref: np.ndarray, W_hat: np.ndarray) -> float:
    """W_*: (..., P, v). Returns mean SGCS in [0, 1]."""
    W_ref, W_hat = np.asarray(W_ref), np.asarray(W_hat)
    inner = np.abs(np.sum(W_ref.conj() * W_hat, axis=-2)) ** 2
    norms = (np.sum(np.abs(W_ref) ** 2, axis=-2) * np.sum(np.abs(W_hat) ** 2, axis=-2))
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(norms > 0, inner / norms, 0.0)
    return float(np.mean(vals))


def nmse(W_ref: np.ndarray, W_hat: np.ndarray) -> float:
    """Phase-aligned column-wise NMSE, averaged (linear, not dB)."""
    W_ref, W_hat = np.asarray(W_ref), np.asarray(W_hat)
    inner = np.sum(W_ref.conj() * W_hat, axis=-2)
    phase = np.exp(-1j * np.angle(inner))
    err = np.sum(np.abs(W_hat * phase[..., None, :] - W_ref) ** 2, axis=-2)
    ref = np.sum(np.abs(W_ref) ** 2, axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(ref > 0, err / ref, 0.0)
    return float(np.mean(vals))
