"""Compact matrix/tensor models of the codebooks (paper Sec. "Compact Model").

These are *independent* second implementations of the precoder structure,

    R15:  w = W_s_hat  @ w_c                                   (P,)
    R16:  W = W_s_hat  @ W_c @ W_f_hat^T                       (P, N3)
    R18:  vec(W) = (W_f_hat kron W_s_hat) @ W_c @ W_t_hat^T    (P*N3, N4)
          == Tucker:  Wc x1 W_s_hat x2 W_f_hat x3 W_t_hat      (P, N3, N4)

used by the tests to cross-validate the spec-table reconstructions.  The
compact model deliberately omits the normalization (the paper "sets aside"
scaling), so comparisons are direction-wise per column.
"""

from __future__ import annotations

import numpy as np

from ..utils import dft


def dual_block(B: np.ndarray) -> np.ndarray:
    """W_s_hat = blockdiag(B^T, B^T): (P, 2L) from per-pol bases B (L, P/2)."""
    L, half = B.shape
    Ws = np.zeros((2 * half, 2 * L), dtype=complex)
    Ws[:half, :L] = B.T
    Ws[half:, L:] = B.T
    return Ws


def compact_r15(B: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """w = W_s_hat @ w_c for one layer/subband; coeffs shape (2L,)."""
    return dual_block(B) @ coeffs


def compact_r16(B: np.ndarray, Wc: np.ndarray, taps: list[int], N3: int) -> np.ndarray:
    """W = W_s_hat @ W_c @ W_f_hat^T -> (P, N3); Wc shape (2L, Mv)."""
    Wf = dft.freq_basis(N3, np.array(taps)).T  # (N3, Mv)
    return dual_block(B) @ Wc @ Wf.T


def compact_r18(
    B: np.ndarray, Wc3: np.ndarray, taps: list[int], shifts: list[int], N3: int, N4: int
) -> np.ndarray:
    """Kronecker form: (W_f_hat kron W_s_hat) vec-stacked -> (N4, N3, P).

    Wc3: coefficient tensor (2L, Mv, Q).
    """
    Ws = dual_block(B)  # (P, 2L)
    Wf = dft.freq_basis(N3, np.array(taps)).T  # (N3, Mv)
    Wt = dft.time_basis(N4, np.array(shifts)).T  # (N4, Q)
    twoL, Mv, Q = Wc3.shape
    Wc = Wc3.reshape(twoL * Mv, Q, order="F")  # vec over (2L, Mv), column-major
    out = np.kron(Wf, Ws) @ Wc @ Wt.T  # (P*N3, N4)
    P = Ws.shape[0]
    return out.reshape(P, N3, N4, order="F").transpose(2, 1, 0)


def tucker_r18(
    B: np.ndarray, Wc3: np.ndarray, taps: list[int], shifts: list[int], N3: int, N4: int
) -> np.ndarray:
    """Tucker form (paper eq. tucker): Wc x1 Ws x2 Wf x3 Wt -> (N4, N3, P)."""
    Ws = dual_block(B)
    Wf = dft.freq_basis(N3, np.array(taps)).T
    Wt = dft.time_basis(N4, np.array(shifts)).T
    T = np.einsum("pa,afq->pfq", Ws, Wc3)  # mode-1
    T = np.einsum("tf,pfq->ptq", Wf, T)  # mode-2
    T = np.einsum("iq,ptq->pti", Wt, T)  # mode-3
    return T.transpose(2, 1, 0)
