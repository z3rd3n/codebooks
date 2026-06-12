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


def subspace_sgcs(W_ref: np.ndarray, W_hat: np.ndarray) -> float:
    """Fraction of each reference column's energy captured by span(W_hat),
    mean over columns/leading axes.

    Equals ``sgcs`` at v = 1; >= ``sgcs`` always (each W_hat column lies in
    the span); invariant to W_hat -> W_hat @ U for unitary U.  Complements
    the 3GPP column-wise SGCS at rank > 1, which also penalizes rotations
    within the reported subspace that do not affect the log-det rate (e.g.
    Type I's rigid rank-2 pair).
    """
    W_ref, W_hat = np.asarray(W_ref), np.asarray(W_hat)
    Q, R = np.linalg.qr(W_hat)
    # mask Q columns from (near-)zero/degenerate W_hat columns so they cannot
    # spuriously capture energy; all-zero W_hat -> 0.0, matching sgcs
    diag = np.abs(np.diagonal(R, axis1=-2, axis2=-1))  # (..., v_hat)
    keep = diag > 1e-9 * np.max(diag, axis=-1, keepdims=True)
    Q = np.where(keep[..., None, :], Q, 0.0)
    proj = np.swapaxes(Q, -2, -1).conj() @ W_ref  # (..., v_hat, v_ref)
    num = np.sum(np.abs(proj) ** 2, axis=-2)
    den = np.sum(np.abs(W_ref) ** 2, axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(den > 0, num / den, 0.0)
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
