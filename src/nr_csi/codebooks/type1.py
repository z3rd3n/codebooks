"""R15 Type I single-panel regular codebook, Modes 1 and 2, ranks 1-2.

Reconstruction follows the paper's Tables tabmode1/tabmode2 (TS 38.214
Tables 5.2.2.2.1-5/-6):

    rank 1:  W = 1/sqrt(P) [v_{l,m}; phi_n v_{l,m}]
    rank 2:  W = 1/sqrt(2P) [[v_{l,m}, v_{l',m'}], [phi_n v_{l,m}, -phi_n v_{l',m'}]]

with phi_n = exp(j*pi*n/2) and (l', m') = (l + k1, m + k2) given by i_{1,3}
(Table tabmap).  Mode 1 reports a wideband beam and per-subband co-phasing;
Mode 2 reports a wideband group of 4 beams and per-subband beam + co-phasing
(supported for N2 > 1, as detailed in the paper).

Beam restriction follows the bit sequence a = vec(V^T), i.e. bit N2*O2*l + m
governs beam v_{l,m} (paper eq. type1map).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import AntennaConfig
from ..utils import dft
from .base import CodebookScheme

# Mode-2 beam-in-group offsets, in i2 order (paper Table tabmode2).
_MODE2_OFFSETS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def i13_offsets(N1: int, N2: int, O1: int, O2: int) -> list[tuple[int, int]]:
    """(k1, k2) as a function of i_{1,3} (paper Table tabmap + spec N1=2,N2=1 case)."""
    if N1 > N2 > 1:
        return [(0, 0), (O1, 0), (0, O2), (2 * O1, 0)]
    if N1 == N2:
        return [(0, 0), (O1, 0), (0, O2), (O1, O2)]
    if N1 > 2 and N2 == 1:
        return [(0, 0), (O1, 0), (2 * O1, 0), (3 * O1, 0)]
    if N1 == 2 and N2 == 1:
        return [(0, 0), (O1, 0)]
    raise ValueError(f"unsupported (N1,N2)=({N1},{N2}) for rank-2 Type I")


@dataclass
class Type1PMI:
    rank: int
    mode: int
    i11: int
    i12: int
    i2: np.ndarray  # per-subband, shape (N3,)
    i13: int | None = None  # rank 2 only


class Type1Codebook(CodebookScheme):
    name = "R15 Type I"

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int = 1,
        mode: int = 1,
        beam_restriction: np.ndarray | None = None,
        selection_snr_db: float = 10.0,
    ) -> None:
        if mode not in (1, 2):
            raise ValueError("codebookMode must be 1 or 2")
        if mode == 2 and antenna.N2 == 1:
            raise ValueError("codebook Mode 2 requires N2 > 1 (paper Table tabmode2)")
        self.antenna = antenna
        self.N3 = N3
        self.mode = mode
        n_bits = antenna.N1 * antenna.O1 * antenna.N2 * antenna.O2
        if beam_restriction is None:
            beam_restriction = np.ones(n_bits, dtype=bool)
        self.beam_restriction = np.asarray(beam_restriction, dtype=bool)
        if self.beam_restriction.shape != (n_bits,):
            raise ValueError(f"beam restriction bitmap must have {n_bits} bits")
        self.selection_rho = 10 ** (selection_snr_db / 10)

    # -- helpers -----------------------------------------------------------

    def _beam_allowed(self, l: int, m: int) -> bool:
        G1, G2 = self.antenna.n_beams
        return bool(self.beam_restriction[(m % G2) + G2 * (l % G1)])

    def _w_rank1(self, l: int, m: int, n: int) -> np.ndarray:
        a = self.antenna
        v = dft.spatial_beam(a, l, m)
        phi = np.exp(1j * np.pi * n / 2)
        return np.concatenate([v, phi * v])[:, None] / math.sqrt(a.P)

    def _w_rank2(self, l: int, lp: int, m: int, mp: int, n: int) -> np.ndarray:
        a = self.antenna
        v1 = dft.spatial_beam(a, l, m)
        v2 = dft.spatial_beam(a, lp, mp)
        phi = np.exp(1j * np.pi * n / 2)
        w1 = np.concatenate([v1, phi * v1])
        w2 = np.concatenate([v2, -phi * v2])
        return np.stack([w1, w2], axis=1) / math.sqrt(2 * a.P)

    def _beam_and_phase(self, pmi: Type1PMI, t: int) -> tuple[int, int, int]:
        """(l, m, n) for frequency unit t (resolving Mode 2's i2 split)."""
        i2 = int(pmi.i2[t])
        if pmi.mode == 1:
            return pmi.i11, pmi.i12, i2
        n_phases = 4 if pmi.rank == 1 else 2
        k1p, k2p = _MODE2_OFFSETS[i2 // n_phases]
        return 2 * pmi.i11 + k1p, 2 * pmi.i12 + k2p, i2 % n_phases

    # -- gNB side ----------------------------------------------------------

    def precoder(self, pmi: Type1PMI) -> np.ndarray:
        from .validate import validate_type1

        validate_type1(self, pmi)
        a = self.antenna
        W = np.empty((1, self.N3, a.P, pmi.rank), dtype=complex)
        offsets = i13_offsets(a.N1, a.N2, a.O1, a.O2) if pmi.rank == 2 else None
        for t in range(self.N3):
            l, m, n = self._beam_and_phase(pmi, t)
            if pmi.rank == 1:
                W[0, t] = self._w_rank1(l, m, n)
            else:
                k1, k2 = offsets[pmi.i13]
                W[0, t] = self._w_rank2(l, l + k1, m, m + k2, n)
        return W

    # -- UE side -----------------------------------------------------------

    def select(self, H: np.ndarray, rank: int = 1) -> Type1PMI:
        if rank not in (1, 2):
            raise ValueError("Type I implementation supports ranks 1-2")
        H = np.asarray(H)
        if H.ndim != 4:
            raise ValueError("H must be [slot, t, rx, port]")
        Ht = H[-1]  # most recent slot, shape (N3, Nr, P)
        if Ht.shape[0] != self.N3:
            raise ValueError(f"channel has {Ht.shape[0]} frequency units, expected {self.N3}")

        a = self.antenna
        G1, G2 = a.n_beams
        if self.mode == 1:
            cand_wb = [(i11, i12) for i11 in range(G1) for i12 in range(G2)]
        else:
            cand_wb = [(i11, i12) for i11 in range(G1 // 2) for i12 in range(max(G2 // 2, 1))]

        offsets = i13_offsets(a.N1, a.N2, a.O1, a.O2) if rank == 2 else [None]
        n_i2 = self._n_i2(rank)

        best = (-np.inf, None)
        for i11, i12 in cand_wb:
            for i13_idx, _ in enumerate(offsets):
                metric = 0.0
                i2_per_t = np.zeros(self.N3, dtype=int)
                feasible = True
                for t in range(self.N3):
                    rates = np.full(n_i2, -np.inf)
                    for i2 in range(n_i2):
                        pmi_t = Type1PMI(rank, self.mode, i11, i12,
                                         np.array([i2]), i13_idx if rank == 2 else None)
                        l, m, n = self._beam_and_phase(pmi_t, 0)
                        if not self._beam_allowed(l, m):
                            continue
                        if rank == 1:
                            Wc = self._w_rank1(l, m, n)
                        else:
                            k1, k2 = offsets[i13_idx]
                            if not self._beam_allowed(l + k1, m + k2):
                                continue
                            Wc = self._w_rank2(l, l + k1, m, m + k2, n)
                        rates[i2] = _su_logdet(Ht[t], Wc, self.selection_rho)
                    if np.isinf(rates).all():
                        feasible = False
                        break
                    i2_per_t[t] = int(rates.argmax())
                    metric += rates.max()
                if feasible and metric > best[0]:
                    best = (metric, Type1PMI(rank, self.mode, i11, i12, i2_per_t,
                                             i13_idx if rank == 2 else None))
        if best[1] is None:
            raise RuntimeError("no feasible Type I PMI under the configured restriction")
        return best[1]

    def _n_i2(self, rank: int) -> int:
        if self.mode == 1:
            return 4 if rank == 1 else 2
        return 16 if rank == 1 else 8

    # -- overhead ----------------------------------------------------------

    def overhead_bits(self, pmi: Type1PMI) -> dict[str, int]:
        a = self.antenna
        G1, G2 = a.n_beams
        div = 2 if self.mode == 2 else 1
        bits = {
            "i11": math.ceil(math.log2(max(G1 // div, 1))),
            "i12": math.ceil(math.log2(max(G2 // div, 1))),
            "i2": self.N3 * math.ceil(math.log2(self._n_i2(pmi.rank))),
        }
        if pmi.rank == 2:
            n_off = len(i13_offsets(a.N1, a.N2, a.O1, a.O2))
            bits["i13"] = math.ceil(math.log2(n_off))
        return bits


def _su_logdet(Ht: np.ndarray, W: np.ndarray, rho: float) -> float:
    """log2 det(I + rho/v * (HW)(HW)^H) for one frequency unit."""
    HW = Ht @ W
    v = W.shape[1]
    G = HW.conj().T @ HW  # (v, v)
    return float(np.log2(np.linalg.det(np.eye(v) + (rho / v) * G).real))
