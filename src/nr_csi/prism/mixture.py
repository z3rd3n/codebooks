"""PRISM mixture codec: a published dictionary of K KLT sketch bases.

``fit_mixture`` runs a Lloyd-style K-subspaces alternation on angle-delay
eigen-target vectors ``g[n, D]``:

* **assign** each sample to the basis whose top-``m_ref`` rows capture the
  most of its energy (the same criterion the UE applies per report);
* **refit** each cluster's KLT (:func:`nr_csi.ml.projection.fit_klt`) on its
  assigned samples.

The captured-energy objective is monotone non-decreasing under both steps, so
the alternation converges; we run multiple seeded restarts and keep the best.
This is Lloyd's algorithm lifted from scalar levels to subspaces -- the same
design language as the scheme's Lloyd-Max coordinate quantizer, and the same
object 3GPP already standardizes: a codebook, here a codebook *of transforms*.

``PrismCodec`` packages the K fitted ``(A_k, sigma_k)`` pairs with the shared
angle-delay transform. ``K = 1`` is exactly a single-KLT GLIMPSE codec fitted
on the pooled mix (the "broad GLIMPSE" ablation).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np

from ..config import AntennaConfig
from ..ml.projection import GlimpseCodec, fit_klt


def captured_energy(A: np.ndarray, g: np.ndarray, m: int) -> np.ndarray:
    """Energy of ``g[n, D]`` captured by the top-``m`` rows of ``A``:
    ``||A[:m] g||^2`` per sample (rows orthonormal, so this is the energy of
    the projection onto the sketch subspace)."""
    return np.sum(np.abs(g @ A[:m].T) ** 2, axis=-1)


def fit_mixture(
    g: np.ndarray,
    n_components: int,
    m_max: int,
    m_ref: int = 16,
    n_iter: int = 30,
    n_restarts: int = 4,
    seed: int = 0,
    tol: float = 1e-6,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, float]:
    """K-subspaces fit of ``n_components`` KLT bases on ``g[n, D]``.

    Returns ``(bases, sigmas, assign, objective)`` where ``objective`` is the
    mean captured-energy fraction at ``m_ref`` (targets are unit-norm, so this
    is directly comparable to GLIMPSE's single-basis capture figure).
    """
    g = np.asarray(g)
    n, D = g.shape
    if not 1 <= n_components:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if n_components == 1:  # degenerate case: plain pooled KLT, no restarts
        A, sigma = fit_klt(g, m_max)
        obj = float(np.mean(captured_energy(A, g, m_ref)))
        return [A], [sigma], np.zeros(n, dtype=int), obj

    best: tuple | None = None
    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + restart)
        # -- residual-greedy seeding (k-means++ analogue for subspaces).
        # Random-half initialization is degenerate: every random half shares
        # the pooled covariance, so all components start identical and the
        # argmax never separates them.  Instead: seed component 0 on a random
        # subset, then seed each next component on the samples the current
        # dictionary captures WORST.
        n_seed = min(max(4 * m_max, n // (4 * n_components)), n)
        bases = [fit_klt(g[rng.choice(n, size=n_seed, replace=False)], m_max)[0]]
        for _ in range(1, n_components):
            covered = np.max(
                np.stack([captured_energy(A, g, m_ref) for A in bases], 1), 1)
            worst = np.argsort(covered)[:n_seed]
            bases.append(fit_klt(g[worst], m_max)[0])
        energy = np.stack([captured_energy(A, g, m_ref) for A in bases], axis=1)
        assign = np.argmax(energy, axis=1)
        prev_obj = -np.inf
        for _ in range(n_iter):
            bases, sigmas = [], []
            for k in range(n_components):
                members = g[assign == k]
                if len(members) < 2 * m_max:  # collapsed cluster: reseed it
                    members = g[rng.choice(n, size=n_seed, replace=False)]
                A, sigma = fit_klt(members, m_max)
                bases.append(A)
                sigmas.append(sigma)
            energy = np.stack(
                [captured_energy(A, g, m_ref) for A in bases], axis=1)  # (n, K)
            assign = np.argmax(energy, axis=1)
            obj = float(np.mean(np.max(energy, axis=1)))
            if obj - prev_obj < tol:
                break
            prev_obj = obj
        # refit sigmas on the final assignment (population stds per cluster)
        for k in range(n_components):
            members = g[assign == k]
            if len(members) >= 2 * m_max:
                bases[k], sigmas[k] = fit_klt(members, m_max)
        if best is None or obj > best[3]:
            best = (bases, sigmas, assign, obj)
    return best


@dataclass(frozen=True)
class PrismCodec:
    """The published constant of PRISM: K KLT bases + stds for one grid.

    Delegates the (basis-independent) angle-delay transforms to an internal
    :class:`~nr_csi.ml.projection.GlimpseCodec`, and adds per-report basis
    selection.  ``index_bits = ceil(log2 K)`` is the selection overhead.
    """

    antenna: AntennaConfig
    N3: int
    bases: tuple = ()  # K arrays (m_max, D)
    sigmas: tuple = ()  # K arrays (m_max,)
    _transform: GlimpseCodec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.bases:
            raise ValueError("PrismCodec needs at least one fitted basis")
        if len(self.bases) != len(self.sigmas):
            raise ValueError("bases and sigmas must pair up")
        shape = self.bases[0].shape
        for A in self.bases:
            if A.shape != shape:
                raise ValueError("all bases must share (m_max, D) shape")
        if shape[1] != self.antenna.P * self.N3:
            raise ValueError(f"basis D={shape[1]} != P*N3={self.antenna.P * self.N3}")
        object.__setattr__(
            self, "_transform", GlimpseCodec(self.antenna, self.N3, m_max=shape[0]))

    @property
    def n_components(self) -> int:
        return len(self.bases)

    @property
    def m_max(self) -> int:
        return self.bases[0].shape[0]

    @property
    def D(self) -> int:
        return self.bases[0].shape[1]

    @property
    def index_bits(self) -> int:
        return max(int(np.ceil(np.log2(self.n_components))), 0)

    # ------------------------------------------------------------- transforms
    def targets_to_vec(self, V: np.ndarray) -> np.ndarray:
        return self._transform.targets_to_vec(V)

    def vec_to_targets(self, g: np.ndarray) -> np.ndarray:
        return self._transform.vec_to_targets(g)

    # -------------------------------------------------------------- selection
    def select_basis(self, g: np.ndarray, m: int) -> int:
        """The UE rule: index of the basis capturing the most energy of
        ``g[..., D]`` (summed over leading axes, i.e. over layers) in its
        top-``m`` sketch."""
        g2 = np.atleast_2d(g)
        energies = [float(np.sum(captured_energy(A, g2, m))) for A in self.bases]
        return int(np.argmax(energies))

    # ------------------------------------------------------- per-basis pieces
    def project(self, g: np.ndarray, m: int, k: int) -> np.ndarray:
        if not 1 <= m <= self.m_max:
            raise ValueError(f"m must be in [1, {self.m_max}], got {m}")
        return g @ self.bases[k][:m].T

    def standardize(self, y: np.ndarray, k: int) -> np.ndarray:
        return y / self.sigmas[k][: y.shape[-1]]

    def destandardize(self, u: np.ndarray, k: int) -> np.ndarray:
        return u * self.sigmas[k][: u.shape[-1]]

    def adjoint(self, y: np.ndarray, m: int, k: int) -> np.ndarray:
        """Least-squares (min-norm) inverse ``A_k[:m]^H y`` -- rows are
        orthonormal, so this is the exact pseudo-inverse."""
        return y @ self.bases[k][:m].conj()

    # ------------------------------------------------------------ persistence
    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"N1": self.antenna.N1, "N2": self.antenna.N2, "N3": self.N3,
                  "n_components": self.n_components}
        for k, (A, s) in enumerate(zip(self.bases, self.sigmas)):
            arrays[f"A{k}"] = A
            arrays[f"sigma{k}"] = s
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "PrismCodec":
        with np.load(pathlib.Path(path).with_suffix(".npz")) as d:
            ant = AntennaConfig.standard(int(d["N1"]), int(d["N2"]))
            K = int(d["n_components"])
            bases = tuple(d[f"A{k}"] for k in range(K))
            sigmas = tuple(d[f"sigma{k}"] for k in range(K))
            return cls(ant, int(d["N3"]), bases=bases, sigmas=sigmas)


# --------------------------------------------------------------------------- #
# UE complexity (complex MACs), mirroring nr_csi.ml.projection.encoder_flops.
# --------------------------------------------------------------------------- #
def prism_encoder_flops(antenna: AntennaConfig, N3: int, m: int,
                        n_components: int, m_sel: int | None = None) -> dict[str, int]:
    """PRISM UE encode cost.

    Direct rule: all K sketches at length ``m`` (the winner's sketch is then
    already computed): ``K m D`` MACs.  Two-stage rule (``m_sel`` short
    selection prefixes, then the winner's full sketch):
    ``K m_sel D + m D``.
    """
    D = antenna.P * N3
    if m_sel is None:
        proj = n_components * m * D
    else:
        proj = n_components * m_sel * D + m * D
    return {"projection": proj, "standardize_quantize": 2 * m, "argmax": n_components}
