"""UE-side GLIMPSE encoder core: the fixed measurement projection.

The report is ``y = A_m vec(G)`` where ``G`` is the angle-delay transform of
the phase-aligned per-subband eigenvectors and ``A_m`` are the first ``m``
rows of a *fixed, published* semi-unitary matrix.  Two basis choices, both
parameter-free at the UE (one ``m x D`` matrix multiply):

* ``basis="klt"`` (default, headline) -- the Karhunen-Loeve basis of the
  channel eigenvector distribution, fitted once offline and published as a
  constant.  Principal directions are ordered by variance, so ``A_m`` captures
  the maximum energy of any ``m``-dimensional linear sketch, and truncation
  drops the least informative measurement first (an *optimally* ordered
  rateless report).  This is the classical DFT->KLT upgrade of the codebook's
  fixed DFT dictionary: the same idea, but the basis is matched to the channel
  statistics instead of assumed.
* ``basis="random"`` (ablation) -- a seeded semi-unitary matrix.  Its rows are
  generic, so the measurement entries are near-Gaussian for *any* channel
  distribution (a JL/rotation argument) and every prefix is equally
  informative; the price is that it is distribution-*blind* and needs many
  more measurements for the same fidelity.

Both are fixed constants: nothing is learned or adapted at the UE.  Because
the spatial DFT, the delay IDFT, and ``A`` are all linear, the whole encoder
collapses to one ``m x D`` complex matrix multiply on the eigenvector the UE
already computes for CQI/RI.

Per-coordinate standardization: coordinate ``k`` has population std
``sigma_k`` (``sqrt`` of the KLT eigenvalue; ~constant for the random basis).
The UE reports ``u = y / sigma`` -- unit-variance coordinates a single fixed
Lloyd-Max Gaussian quantizer digitizes near-optimally -- and the gNB rescales
by the published ``sigma``.  The eigenvector target is unit-norm, so no
per-report scale is signalled.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

from ..config import AntennaConfig
from ..dataset.preprocess import from_delay, spatial_dft, to_delay


def measurement_matrix(D: int, m_max: int, seed: int = 0) -> np.ndarray:
    """First ``m_max`` rows of a seeded ``D x D`` unitary matrix (random basis).

    Drawn once from a complex Ginibre ensemble and orthonormalized by QR;
    columns of Q are phase-fixed against the QR sign ambiguity so the matrix
    is reproducible.  Rows are orthonormal: ``A A^H = I_m`` for every prefix.
    """
    if not 1 <= m_max <= D:
        raise ValueError(f"m_max must be in [1, {D}], got {m_max}")
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
    Q, R = np.linalg.qr(Z)
    d = np.diagonal(R)
    Q = Q * (d.conj() / np.abs(d))  # phase-fix each column
    return Q.conj().T[:m_max].astype(np.complex128)  # rows = fixed directions


def fit_klt(g: np.ndarray, m_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit the KLT basis from angle-delay eigenvector vectors ``g[n, D]``.

    Returns ``(A[m_max, D], sigma[m_max])``: the top-``m_max`` principal
    directions (rows, orthonormal, variance-ordered) and their per-coordinate
    population std ``sigma_k = sqrt(lambda_k)``.  Computed from the (uncentered)
    second-moment matrix -- the eigenvectors are phase-random, so their mean is
    ~0 and the second moment is the natural covariance.
    """
    g = np.asarray(g)
    D = g.shape[1]
    if not 1 <= m_max <= D:
        raise ValueError(f"m_max must be in [1, {D}], got {m_max}")
    # covariance E[g g^H] (rows of g are samples): its eigenvectors u_k are the
    # principal directions, and the report coordinate is y_k = u_k^H g, which
    # the codec computes as g @ A^T with A row k = u_k^H = conj(u_k).
    C = (g.T @ g.conj()) / g.shape[0]  # (D, D) Hermitian second moment
    w, U = np.linalg.eigh(C)  # ascending eigenvalues
    U = U[:, ::-1]  # principal first
    w = np.clip(w[::-1], 0.0, None)
    A = U[:, :m_max].conj().T.astype(np.complex128)  # rows = principal directions
    sigma = np.sqrt(w[:m_max]).astype(np.float64)
    return A, sigma


@dataclass(frozen=True)
class GlimpseCodec:
    """The fixed (spec-side) part of GLIMPSE for one ``(antenna, N3)`` grid.

    Handles target <-> angle-delay <-> vector conversions, the projection, and
    per-coordinate standardization.  ``m_max`` bounds the largest report; any
    ``m <= m_max`` uses the prefix ``A[:m]`` (the rateless property).

    Provide a fitted ``basis_matrix``/``sigma`` (from :func:`fit_klt`, saved
    via :meth:`save`/:meth:`load`) for the KLT codec, or leave them ``None``
    for the seeded random basis.
    """

    antenna: AntennaConfig
    N3: int
    m_max: int = 64
    seed: int = 0
    basis: str = "random"
    basis_matrix: np.ndarray | None = None
    sigma: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.basis_matrix is not None:
            A = np.asarray(self.basis_matrix)
            if A.shape != (self.m_max, self.D):
                raise ValueError(
                    f"basis_matrix has shape {A.shape}, expected ({self.m_max}, {self.D})"
                )
            sigma = (np.ones(self.m_max) if self.sigma is None
                     else np.asarray(self.sigma, float))
            object.__setattr__(self, "basis", "klt")
        else:
            A = measurement_matrix(self.D, self.m_max, self.seed)
            sigma = np.ones(self.m_max)  # random rows: ~uniform coordinate variance
            object.__setattr__(self, "basis", "random")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "_sigma", sigma)

    @property
    def D(self) -> int:
        return self.antenna.P * self.N3

    # -------------------------------------------------------------- persistence
    def save(self, path: str | pathlib.Path) -> None:
        """Write the published constant (``A``, ``sigma``) as ``<path>.npz``."""
        path = pathlib.Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, A=self.A, sigma=self._sigma,
                            N1=self.antenna.N1, N2=self.antenna.N2, N3=self.N3,
                            m_max=self.m_max, basis=self.basis)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "GlimpseCodec":
        with np.load(pathlib.Path(path).with_suffix(".npz"), allow_pickle=True) as d:
            ant = AntennaConfig.standard(int(d["N1"]), int(d["N2"]))
            return cls(ant, int(d["N3"]), m_max=int(d["m_max"]),
                       basis_matrix=d["A"], sigma=d["sigma"])

    # ---------------------------------------------------------------- targets
    def targets_to_vec(self, V: np.ndarray) -> np.ndarray:
        """Aligned eigen targets ``V[N3, P, v]`` (or ``[N3, P]``) to
        angle-delay vectors ``g[v, D]`` (or ``[D]``)."""
        V = np.asarray(V)
        single = V.ndim == 2
        if single:
            V = V[..., None]
        if V.shape[:2] != (self.N3, self.antenna.P):
            raise ValueError(
                f"targets have shape {V.shape}, expected ({self.N3}, {self.antenna.P}, v)"
            )
        X = V.transpose(2, 1, 0)  # (v, P, N3): port x frequency
        G = to_delay(spatial_dft(X, self.antenna))  # unitary, (v, P, N3)
        g = G.reshape(V.shape[2], self.D)
        return g[0] if single else g

    def vec_to_targets(self, g: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`targets_to_vec`: ``g[..., D]`` -> ``V[N3, P, v]``
        (or ``[N3, P]`` for a single vector)."""
        g = np.asarray(g)
        single = g.ndim == 1
        G = g.reshape(-1, self.antenna.P, self.N3)
        X = spatial_dft(from_delay(G, self.N3), self.antenna, inverse=True)
        V = X.transpose(2, 1, 0)  # (N3, P, v)
        return V[..., 0] if single else V

    # ------------------------------------------------------------ projection
    def project(self, g: np.ndarray, m: int) -> np.ndarray:
        """Raw measurements ``y = A_m g`` for ``g[..., D]`` -> ``y[..., m]``."""
        if not 1 <= m <= self.m_max:
            raise ValueError(f"m must be in [1, {self.m_max}], got {m}")
        return g @ self.A[:m].T

    def standardize(self, y: np.ndarray) -> np.ndarray:
        """Divide each coordinate by its population std -> unit-variance report
        (what the fixed Gaussian quantizer digitizes)."""
        m = y.shape[-1]
        return y / self._sigma[:m]

    def destandardize(self, u: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`standardize`: back to physical measurements."""
        m = u.shape[-1]
        return u * self._sigma[:m]

    def adjoint(self, y: np.ndarray, m: int) -> np.ndarray:
        """Back-projection ``A_m^H y`` (exact pseudo-inverse; rows orthonormal).

        Consumes *physical* measurements (destandardize first)."""
        return y @ self.A[:m].conj()


# --------------------------------------------------------------------------- #
# Complexity accounting (UE side), in complex multiply-accumulate operations.
# Shared prerequisites (per-subband eigen targets, needed for CQI whichever
# scheme reports PMI) are excluded from both counts.
# --------------------------------------------------------------------------- #
def encoder_flops(antenna: AntennaConfig, N3: int, m: int) -> dict[str, int]:
    """GLIMPSE UE encode cost with the composed one-GEMM implementation."""
    D = antenna.P * N3
    return {
        "projection": m * D,  # y = (A T) vec(V)
        "standardize_quantize": 2 * m,  # per-coord scale + table lookups
    }


def type2_select_flops(antenna: AntennaConfig, N3: int, L: int, Mv: int) -> dict[str, int]:
    """R16 eType II ``select`` cost with the repo's UE algorithm
    (:mod:`nr_csi.codebooks._spatial` / ``etype2_r16``), complex MACs.

    * orthogonal-group scan: O1*O2 groups x N1*N2 beams x P/2 ports x N3
      subbands x 2 polarizations;
    * LS combination coefficients on the L selected beams;
    * FD DFT across N3 per selected beam column (as a dense N3 x N3 product,
      matching the reference implementation);
    * per-coefficient amplitude/phase quantization (2 ops per coefficient).
    """
    half = antenna.P // 2
    n_beams = antenna.N1 * antenna.N2
    group_scan = antenna.O1 * antenna.O2 * n_beams * half * N3 * 2
    ls_coeff = L * half * N3 * 2
    fd_dft = 2 * L * N3 * N3
    quant = 2 * (2 * L * Mv)
    return {
        "orthogonal_group_scan": group_scan,
        "ls_coefficients": ls_coeff,
        "fd_dft": fd_dft,
        "quantization": quant,
    }
