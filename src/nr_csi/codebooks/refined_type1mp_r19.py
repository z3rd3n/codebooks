"""Release-19 Refined Type I Multi-Panel codebook, TS 38.214 5.2.2.2.2a.

'typeI-MultiPanel-r19' for the aggregated arrays of Table 5.2.2.2.2a-1
(48/64/128 ports over Ng in {2, 4} CSI-RS resources, O1 = O2 = 4), ranks 1-4.

Structure (Table 5.2.2.2.2a-3): the aggregated precoder stacks per-panel
blocks [phi_{p_{j-1}} v_j ; +-phi_{n_j} phi_{p_{j-1}} v_j] (panel-major, the
same port layout as the Release-15 multi-panel codebook), where

* each panel j selects its *own* beam (l_j, m_j) = (i_{1,1,j}, i_{1,2,j})
  over the full oversampled grid -- the Release-19 refinement vs. the R15
  codebook's single common beam;
* ranks 2-4 add a per-panel companion beam (l_j + k_{1,j}, m_j + k_{2,j})
  through the per-panel offset index i_{1,3,j} (Table 5.2.2.2.2a-2, the same
  offsets as Table 5.2.2.2.1a-3 for ranks 2-4);
* i_{1,4,q}, q = 1..Ng-1, are wideband inter-panel QPSK phases phi_{p_q};
* i_{2,j} is the per-panel polarization co-phasing n_j, per PMI frequency
  unit (QPSK for rank 1, BPSK-of-QPSK {0,1} for ranks 2-4);
* the layer families W^{1,Ng} / W^{2,Ng} pair columns as
  v=1: [W1(b)]; v=2: [W1(b), W2(b')]; v=3: [W1(b), W1(b'), W2(b)];
  v=4: [W1(b), W1(b'), W2(b), W2(b')], with b' the companion beams.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..config import SUPPORTED_NG_N1N2_R19, AntennaConfig
from ..utils import dft
from .base import CodebookScheme
from .refined_type1_r19 import i13_lowrank


@dataclass
class RefinedType1MPPMI:
    rank: int
    i11: tuple[int, ...]  # per-panel horizontal beam index l_j
    i12: tuple[int, ...]  # per-panel vertical beam index m_j
    i14: tuple[int, ...]  # Ng-1 inter-panel co-phasing indices p_q
    i2: np.ndarray  # (N3, Ng) per-panel polarization co-phasing n_j
    i13: tuple[int, ...] | None = None  # per-panel companion offset, ranks 2-4


class RefinedType1MultiPanelCodebook(CodebookScheme):
    """TS 38.214 5.2.2.2.2a 'typeI-MultiPanel-r19', ranks 1-4."""

    name = "R19 Refined Type I multi-panel"

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int = 1,
        beam_restriction: np.ndarray | None = None,
        ri_restriction: np.ndarray | None = None,
        selection_snr_db: float = 10.0,
    ) -> None:
        if (antenna.Ng, antenna.N1, antenna.N2) not in SUPPORTED_NG_N1N2_R19:
            raise ValueError(
                f"(Ng,N1,N2)=({antenna.Ng},{antenna.N1},{antenna.N2}) is not a "
                f"Release-19 multi-panel configuration (Table 5.2.2.2.2a-1: "
                f"{sorted(SUPPORTED_NG_N1N2_R19)})"
            )
        if N3 < 1:
            raise ValueError("N3 must be positive")
        self.antenna = antenna
        self.N3 = N3
        # ng-n1-n2-cbsr-r19: one bit a_{N2*O2*l + m} per grid beam v_{l,m},
        # shared by all panels.
        n_bits = math.prod(antenna.n_beams)
        if beam_restriction is None:
            beam_restriction = np.ones(n_bits, dtype=bool)
        self.beam_restriction = np.asarray(beam_restriction, dtype=bool)
        if self.beam_restriction.shape != (n_bits,):
            raise ValueError(f"ng-n1-n2-cbsr-r19 bitmap must have {n_bits} bits")
        # ri-Restriction-r19: 4-bit bitmap [r0..r3].
        if ri_restriction is None:
            ri_restriction = np.ones(4, dtype=bool)
        self.ri_restriction = np.asarray(ri_restriction, dtype=bool)
        if self.ri_restriction.shape != (4,):
            raise ValueError("ri-Restriction-r19 must have 4 bits [r0..r3]")
        self.selection_rho = 10 ** (selection_snr_db / 10)

    # -- helpers -------------------------------------------------------------

    def _beam_allowed(self, l: int, m: int) -> bool:
        G1, G2 = self.antenna.n_beams
        return bool(self.beam_restriction[(m % G2) + G2 * (l % G1)])

    def _companion(self, pmi_rank: int, l: int, m: int, i13: int) -> tuple[int, int]:
        a = self.antenna
        k1, k2 = i13_lowrank(pmi_rank, i13, a.O1, a.O2)
        return l + k1, m + k2

    def _w_columns(
        self,
        rank: int,
        base: list[np.ndarray],
        comp: list[np.ndarray] | None,
        i14: tuple[int, ...],
        n_state: tuple[int, ...],
    ) -> np.ndarray:
        """One (P, rank) precoding matrix for the given beams and phases."""
        a = self.antenna

        def stack(beams: list[np.ndarray], sign: float) -> np.ndarray:
            panel_phases = [1.0] + [np.exp(1j * np.pi * p / 2) for p in i14]
            blocks = []
            for b, pp, n in zip(beams, panel_phases, n_state):
                phi_n = np.exp(1j * np.pi * n / 2)
                blocks.extend([pp * b, sign * phi_n * pp * b])
            return np.concatenate(blocks) / math.sqrt(a.P)

        if rank == 1:
            columns = [stack(base, 1.0)]
        elif rank == 2:
            columns = [stack(base, 1.0), stack(comp, -1.0)]
        elif rank == 3:
            columns = [stack(base, 1.0), stack(comp, 1.0), stack(base, -1.0)]
        else:
            columns = [
                stack(base, 1.0),
                stack(comp, 1.0),
                stack(base, -1.0),
                stack(comp, -1.0),
            ]
        return np.stack(columns, axis=1) / math.sqrt(rank)

    def _beams(self, pmi: RefinedType1MPPMI) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
        a = self.antenna
        base = [dft.spatial_beam(a, l, m) for l, m in zip(pmi.i11, pmi.i12)]
        if pmi.rank == 1:
            return base, None
        comp = [
            dft.spatial_beam(a, *self._companion(pmi.rank, l, m, i13))
            for l, m, i13 in zip(pmi.i11, pmi.i12, pmi.i13)
        ]
        return base, comp

    # -- gNB side ------------------------------------------------------------

    def precoder(self, pmi: RefinedType1MPPMI) -> np.ndarray:
        self._validate(pmi)
        base, comp = self._beams(pmi)
        W = np.empty((1, self.N3, self.antenna.P, pmi.rank), dtype=complex)
        for t in range(self.N3):
            W[0, t] = self._w_columns(
                pmi.rank, base, comp, pmi.i14, tuple(int(x) for x in pmi.i2[t])
            )
        return W

    # -- UE side -------------------------------------------------------------

    def select(self, H: np.ndarray, rank: int = 1) -> RefinedType1MPPMI:
        from ._spatial import aligned_eigen_targets

        if not 1 <= rank <= 4:
            raise ValueError("Refined Type I multi-panel supports ranks 1-4")
        if not self.ri_restriction[rank - 1]:
            raise ValueError(
                f"rank {rank} prohibited by ri-Restriction-r19 (r{rank - 1}=0)"
            )
        H = np.asarray(H)
        if H.ndim != 4:
            raise ValueError("H must be [slot, t, rx, port]")
        Ht = H[-1]
        if Ht.shape[0] != self.N3:
            raise ValueError(f"channel has {Ht.shape[0]} frequency units, expected {self.N3}")
        if Ht.shape[-1] != self.antenna.P:
            raise ValueError(f"channel has {Ht.shape[-1]} ports, expected {self.antenna.P}")
        a = self.antenna
        Ng, npp = a.Ng, a.n_ports_per_pol
        targets = aligned_eigen_targets(Ht, rank)  # (N3, P, v)

        # per-panel beam energy over the oversampled grid (both polarizations)
        grid = dft.spatial_grid(a)  # (G1, G2, npp)
        G1, G2 = a.n_beams
        allowed = np.array(
            [[self._beam_allowed(l, m) for m in range(G2)] for l in range(G1)]
        )
        if not allowed.any():
            raise RuntimeError("ng-n1-n2-cbsr-r19 prohibits every beam")
        i11, i12, i13 = [], [], []
        for j in range(Ng):
            seg = targets[:, 2 * j * npp : 2 * (j + 1) * npp, :]
            pa = np.einsum("xyp,npv->xynv", grid.conj(), seg[:, :npp, :])
            pb = np.einsum("xyp,npv->xynv", grid.conj(), seg[:, npp:, :])
            energy = (np.abs(pa) ** 2 + np.abs(pb) ** 2).sum(axis=(2, 3))  # (G1, G2)
            masked = np.where(allowed, energy, -np.inf)
            l, m = np.unravel_index(int(np.argmax(masked)), masked.shape)
            i11.append(int(l))
            i12.append(int(m))
            if rank > 1:
                # per-panel companion offset: strongest allowed companion beam
                best = (-1.0, None)
                for off in range(4):
                    lc, mc = self._companion(rank, int(l), int(m), off)
                    if not self._beam_allowed(lc, mc):
                        continue
                    e = float(energy[lc % G1, mc % G2])
                    if e > best[0]:
                        best = (e, off)
                if best[1] is None:
                    raise RuntimeError(
                        "ng-n1-n2-cbsr-r19 prohibits every companion beam offset"
                    )
                i13.append(best[1])
        pmi = RefinedType1MPPMI(
            rank,
            tuple(i11),
            tuple(i12),
            (0,) * (Ng - 1),
            np.zeros((self.N3, Ng), dtype=int),
            tuple(i13) if rank > 1 else None,
        )

        # joint search over the wideband inter-panel phases i14 and the
        # per-subband panel co-phasing states n, by spectral efficiency
        base, comp = self._beams(pmi)
        n_states = list(
            itertools.product(range(4 if rank == 1 else 2), repeat=Ng)
        )
        eye = np.eye(rank, dtype=complex)
        best = None
        for i14 in itertools.product(range(4), repeat=Ng - 1):
            Ws = np.stack(
                [self._w_columns(rank, base, comp, i14, ns) for ns in n_states]
            )  # (S, P, v)
            HW = np.einsum("tnp,spv->stnv", Ht, Ws)
            gram = np.einsum("...nv,...nw->...vw", HW.conj(), HW)
            _, logdet = np.linalg.slogdet(eye + self.selection_rho * gram)
            rates = np.real(logdet)  # (S, N3)
            metric = float(rates.max(axis=0).sum())
            if best is None or metric > best[0]:
                best = (metric, i14, rates.argmax(axis=0))
        _, i14, state_idx = best
        pmi.i14 = tuple(int(x) for x in i14)
        pmi.i2 = np.array([n_states[int(s)] for s in state_idx], dtype=int)
        return pmi

    # -- validation ----------------------------------------------------------

    def _validate(self, pmi: RefinedType1MPPMI) -> None:
        a = self.antenna
        G1, G2 = a.n_beams
        if not 1 <= pmi.rank <= 4:
            raise ValueError(f"rank {pmi.rank} not in 1..4")
        for name, vec, hi in (("i11", pmi.i11, G1), ("i12", pmi.i12, G2)):
            if len(vec) != a.Ng or any(not 0 <= x < hi for x in vec):
                raise ValueError(f"{name} needs {a.Ng} entries in [0, {hi})")
        if len(pmi.i14) != a.Ng - 1 or any(not 0 <= p < 4 for p in pmi.i14):
            raise ValueError(f"i14 needs {a.Ng - 1} QPSK entries")
        if pmi.rank == 1:
            if pmi.i13 is not None:
                raise ValueError("i13 must not be reported for rank 1")
        else:
            if pmi.i13 is None or len(pmi.i13) != a.Ng or any(
                not 0 <= x < 4 for x in pmi.i13
            ):
                raise ValueError(f"i13 needs {a.Ng} entries in 0..3")
        i2 = np.asarray(pmi.i2)
        hi = 4 if pmi.rank == 1 else 2
        if i2.shape != (self.N3, a.Ng) or i2.min() < 0 or i2.max() >= hi:
            raise ValueError(
                f"i2 must be ({self.N3}, {a.Ng}) with values in [0, {hi})"
            )

    # -- overhead ------------------------------------------------------------

    def overhead_bits(self, pmi: RefinedType1MPPMI) -> dict[str, int]:
        a = self.antenna
        G1, G2 = a.n_beams
        bits = {
            "i11": a.Ng * math.ceil(math.log2(G1)),
            "i12": a.Ng * math.ceil(math.log2(G2)),
            "i14": 2 * (a.Ng - 1),
            "i2": self.N3 * a.Ng * (2 if pmi.rank == 1 else 1),
        }
        if pmi.rank > 1:
            bits["i13"] = 2 * a.Ng
        return bits
