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
    """Zero forcing: W = H^H (H H^H)^{-1}, computed as the right pseudo-inverse.

    ``pinv`` is identical for full row rank but stays finite when rows are
    colinear (e.g. two users reporting the same PMI direction): the colinear
    users then share the minimum-norm direction and fully interfere -- the
    physically right outcome -- instead of an ``inv`` blow-up.
    """
    return np.linalg.pinv(H)


def rzf(H: np.ndarray, xi: float) -> np.ndarray:
    """Regularized zero forcing: W = H^H (H H^H + xi I)^{-1}."""
    n = H.shape[0]
    return H.conj().T @ np.linalg.inv(H @ H.conj().T + xi * np.eye(n))


def mmse(H: np.ndarray, snr: float) -> np.ndarray:
    """MMSE beamforming: RZF with xi = sigma^2 / P_t = 1/snr."""
    return rzf(H, 1.0 / snr)


def gmd(H: np.ndarray, n_streams: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geometric mean decomposition H_K = Q R P^H (paper Table SUMIMO).

    R is upper triangular with all diagonal entries equal to the geometric
    mean of the K largest singular values; W = P[:, :K] is the beamformer.
    H_K denotes the rank-K truncation of H.  Returns (Q, R, P).

    Algorithm: Jiang/Hager/Li 2005 -- start from the truncated SVD and apply
    pairwise Givens-like rotations that equalize the diagonal.
    """
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    K = n_streams
    if K > len(s):
        raise ValueError(f"n_streams {K} exceeds rank bound {len(s)}")
    Q = U[:, :K].copy()
    P = Vh[:K].conj().T.copy()
    R = np.diag(s[:K]).astype(complex)
    sbar = float(np.exp(np.mean(np.log(s[:K]))))
    for k in range(K - 1):
        # permute so that diag[k] and diag[k+1] straddle the geometric mean
        d = np.real(np.diag(R))
        rest = d[k + 1:]
        if d[k] >= sbar:
            cand = [j for j in range(k + 1, K) if d[j] <= sbar]
            j = cand[0] if cand else int(np.argmin(rest)) + k + 1
        else:
            cand = [j for j in range(k + 1, K) if d[j] >= sbar]
            j = cand[0] if cand else int(np.argmax(rest)) + k + 1
        if j != k + 1:  # symmetric permutation of rows/cols (and Q/P columns)
            perm = list(range(K))
            perm[k + 1], perm[j] = perm[j], perm[k + 1]
            R = R[np.ix_(perm, perm)]
            Q = Q[:, perm]
            P = P[:, perm]
        d1, d2 = float(np.real(R[k, k])), float(np.real(R[k + 1, k + 1]))
        if abs(d1 - d2) < 1e-12:
            c, t = 1.0, 0.0
        else:
            c = np.sqrt(min(max((sbar**2 - d2**2) / (d1**2 - d2**2), 0.0), 1.0))
            t = np.sqrt(1.0 - c**2)
        G1 = np.eye(K, dtype=complex)  # right rotation (acts on P)
        G1[k, k], G1[k, k + 1] = c, -t
        G1[k + 1, k], G1[k + 1, k + 1] = t, c
        Rg = R @ G1
        # left rotation zeroing the (k+1, k) entry and putting sbar at (k, k)
        a, b = Rg[k, k], Rg[k + 1, k]
        n = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
        G2 = np.eye(K, dtype=complex)
        G2[k, k], G2[k, k + 1] = np.conj(a) / n, np.conj(b) / n
        G2[k + 1, k], G2[k + 1, k + 1] = -b / n, a / n
        R = G2 @ Rg
        Q = Q @ G2.conj().T
        P = P @ G1
    return Q, R, P


def ezf(H_users: np.ndarray, n_streams: int = 1, xi: float = 0.0) -> np.ndarray:
    """Eigen zero-forcing (paper Table MUMIMO): per-user dominant
    eigenvectors stacked and jointly (regularized) zero-forced.

    H_users: (K, Nr, Nt) -> W (Nt, K*n_streams), columns grouped per user.
    At xi = 0 the pseudo-inverse keeps W finite even when users' dominant
    eigendirections are colinear (cf. ``zf``).
    """
    vs = []
    for Hk in H_users:
        _, _, Vh = np.linalg.svd(Hk, full_matrices=False)
        vs.append(Vh.conj().T[:, :n_streams])
    V_eff = np.concatenate(vs, axis=1)  # (Nt, K*v)
    if xi == 0:
        return np.linalg.pinv(V_eff.conj().T)
    n = V_eff.shape[1]
    return V_eff @ np.linalg.inv(V_eff.conj().T @ V_eff + xi * np.eye(n))


def bd(H_users: np.ndarray, n_streams: int = 1) -> np.ndarray:
    """Block diagonalization (paper Table MUMIMO): each user's beamformer
    lives in the null space of all other users' channels.

    H_users: (K, Nr, Nt), requires K*Nr <= Nt.  Returns W (Nt, K*n_streams).
    """
    K, Nr, Nt = H_users.shape
    if K * Nr > Nt:
        raise ValueError(f"BD requires K*Nr <= Nt (got {K}*{Nr} > {Nt})")
    cols = []
    for k in range(K):
        H_bar = np.concatenate([H_users[j] for j in range(K) if j != k])  # ((K-1)Nr, Nt)
        _, s, Vh = np.linalg.svd(H_bar, full_matrices=True)
        V0 = Vh.conj().T[:, H_bar.shape[0]:]  # null-space basis
        H_eff = H_users[k] @ V0
        _, _, Vh_eff = np.linalg.svd(H_eff, full_matrices=False)
        cols.append(V0 @ Vh_eff.conj().T[:, :n_streams])
    return np.concatenate(cols, axis=1)


def wmmse(
    H_users: np.ndarray,
    snr: float,
    n_streams: int = 1,
    n_iter: int = 20,
    priorities: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """WMMSE beamforming (paper Algorithm 'WMMSE'), noise power 1, P_t = snr.

    H_users: (K, Nr, Nt) -> W (Nt, K*n_streams) with tr(W W^H) = P_t.
    The iteration is monotone in the weighted sum rate.
    """
    K, Nr, Nt = H_users.shape
    D = n_streams
    Pt = snr
    chi = np.ones(K) if priorities is None else np.asarray(priorities, float)
    rng = rng or np.random.default_rng(0)
    W = rng.standard_normal((Nt, K * D)) + 1j * rng.standard_normal((Nt, K * D))
    W *= np.sqrt(Pt / np.trace(W @ W.conj().T).real)
    Wk = [W[:, k * D:(k + 1) * D] for k in range(K)]
    for _ in range(n_iter):
        W = np.concatenate(Wk, axis=1)
        gamma1 = np.trace(W @ W.conj().T).real / Pt  # sigma_k^2 = 1
        C, B = [], []
        for k in range(K):
            Hk = H_users[k]
            cov = Hk @ W @ W.conj().T @ Hk.conj().T + gamma1 * np.eye(Nr)
            Ck = np.linalg.solve(cov, Hk @ Wk[k])  # (Nr, D)
            Bk = np.linalg.inv(np.eye(D) - Wk[k].conj().T @ Hk.conj().T @ Ck)
            C.append(Ck)
            B.append(Bk)
        gamma2 = sum(
            chi[k] * np.trace(C[k] @ B[k] @ C[k].conj().T).real / Pt for k in range(K)
        )
        A = sum(
            chi[k] * H_users[k].conj().T @ C[k] @ B[k] @ C[k].conj().T @ H_users[k]
            for k in range(K)
        ) + gamma2 * np.eye(Nt)
        Wk = [
            chi[k] * np.linalg.solve(A, H_users[k].conj().T @ C[k] @ B[k])
            for k in range(K)
        ]
    W = np.concatenate(Wk, axis=1)
    return W * np.sqrt(Pt / np.trace(W @ W.conj().T).real)


def water_filling(gains: np.ndarray, p_total: float) -> np.ndarray:
    """Water-filling power allocation: P_i = (mu - 1/lambda_i^2)^+.

    gains: subchannel SNR factors lambda_i^2 (paper's notation).
    """
    gains = np.asarray(gains, float)
    if np.any(gains <= 0) or p_total <= 0:
        raise ValueError("gains and total power must be positive")
    inv = 1.0 / gains
    order = np.argsort(inv)
    for n_active in range(len(gains), 0, -1):
        active = order[:n_active]
        mu = (p_total + inv[active].sum()) / n_active
        if mu > inv[active].max():
            break
    p = np.maximum(mu - inv, 0.0)
    p[order[n_active:]] = 0.0
    return p


def harmonic_mean_allocation(gains: np.ndarray, p_total: float) -> np.ndarray:
    """Harmonic-mean power allocation: P_i = beta / lambda_i (paper eq.)."""
    lam = np.sqrt(np.asarray(gains, float))
    beta = p_total / np.sum(1.0 / lam)
    return beta / lam
