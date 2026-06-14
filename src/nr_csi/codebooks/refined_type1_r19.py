"""Release-19 Refined Type I Single-Panel codebook, TS 38.214 5.2.2.2.1a.

'typeI-SinglePanel-r19' for the large arrays of Table 5.2.2.2.1a-1 (48/64/128
ports, O1 = O2 = 4).  This module implements **codebookMode 'modeA'**, ranks
1-8.

The per-layer precoding matrices W^(v) (Table 5.2.2.2.1a-4) reuse the *same*
column patterns as the Release-15 Type I codebook (clause 5.2.2.2.1):

    phi_n = e^{j pi n / 2},  v_{l,m} the length-N1*N2 oversampled DFT beam,
    and each column is [b ; +-phi_n b] for a selected beam b.

The Release-19 *refinement* is the beam selection, not the column pattern:

* ranks 2-4 use a fixed i_{1,3} -> (k1,k2) table (Table 5.2.2.2.1a-3) and a
  single companion beam (i_{1,1}+k1, i_{1,2}+k2);
* ranks 5-8 select the 2 or 3 companion beams *independently* via the indices
  i_{1,1,j}, i_{1,2,j}, mapped through the i_{1,3} -> (o1,k1),(o2,k2) row of
  Table 5.2.2.2.1a-3:
      l^(j) = o1 * i_{1,1,j} + k1,   m^(j) = o2 * i_{1,2,j} + k2.

'modeB' (a per-layer / combinatorial beam selection) and the Refined Type I
Multi-Panel codebook (5.2.2.2.2a) are not implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import SUPPORTED_N1N2_R19, AntennaConfig
from ..utils import dft
from .base import CodebookScheme


def i13_lowrank(rank: int, i13: int, O1: int, O2: int) -> tuple[int, int]:
    """Table 5.2.2.2.1a-3, the (k1,k2) companion-beam offset for ranks 2-4."""
    if rank == 2:
        table = [(0, 0), (O1, 0), (0, O2), (2 * O1, 0)]
    else:  # ranks 3, 4
        table = [(O1, 0), (0, O2), (O1, O2), (2 * O1, 0)]
    return table[i13]


def i13_highrank(
    i13: int, i11: int, i12: int, O1: int, O2: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Table 5.2.2.2.1a-3, the ((o1,k1),(o2,k2)) mapping for ranks 5-8."""
    if i13 == 0:
        return (O1, i11 % O1), (1, 0)
    return (1, 0), (O2, i12 % O2)


@dataclass
class RefinedType1PMI:
    rank: int
    i11: int
    i12: int
    i2: np.ndarray  # (N3,) co-phasing index per PMI frequency unit
    i13: int | None = None  # ranks 2-8
    i112: tuple[int, ...] = field(default_factory=tuple)  # ranks 5-8: i_{1,1,j}
    i122: tuple[int, ...] = field(default_factory=tuple)  # ranks 5-8: i_{1,2,j}


# Column sign patterns of Table 5.2.2.2.1a-4, as (beam_index, top_sign, phi,
# bottom_sign, phi).  Each column is [s_t * (phi?phi_n:1) * b ; ...].  Encoded as
# (beam, phi_top, sign_top, phi_bot, sign_bot) where phi_* selects phi_n.
# beam index refers to position in the ordered beam list [b0, b1, b2, b3].
_COLUMN_PATTERNS: dict[int, list[tuple[int, bool, int, bool, int]]] = {
    1: [(0, False, 1, True, 1)],
    2: [(0, False, 1, True, 1), (1, False, 1, True, -1)],
    3: [(0, False, 1, True, 1), (1, False, 1, True, 1), (0, False, 1, True, -1)],
    4: [
        (0, False, 1, True, 1), (1, False, 1, True, 1),
        (0, False, 1, True, -1), (1, False, 1, True, -1),
    ],
    5: [
        (0, False, 1, True, 1), (0, False, 1, True, -1),
        (1, False, 1, False, 1), (1, False, 1, False, -1),
        (2, False, 1, False, 1),
    ],
    6: [
        (0, False, 1, True, 1), (0, False, 1, True, -1),
        (1, False, 1, True, 1), (1, False, 1, True, -1),
        (2, False, 1, False, 1), (2, False, 1, False, -1),
    ],
    7: [
        (0, False, 1, True, 1), (0, False, 1, True, -1),
        (1, False, 1, True, 1),
        (2, False, 1, False, 1), (2, False, 1, False, -1),
        (3, False, 1, False, 1), (3, False, 1, False, -1),
    ],
    8: [
        (0, False, 1, True, 1), (0, False, 1, True, -1),
        (1, False, 1, True, 1), (1, False, 1, True, -1),
        (2, False, 1, False, 1), (2, False, 1, False, -1),
        (3, False, 1, False, 1), (3, False, 1, False, -1),
    ],
}

#: number of companion beams selected independently (ranks 5-8).
_N_EXTRA = {5: 2, 6: 2, 7: 3, 8: 3}


class RefinedType1SinglePanelCodebook(CodebookScheme):
    """TS 38.214 5.2.2.2.1a 'typeI-SinglePanel-r19', codebookMode 'modeA'."""

    name = "R19 Refined Type I single-panel (modeA)"

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int = 1,
        selection_snr_db: float = 10.0,
    ) -> None:
        if (antenna.N1, antenna.N2) not in SUPPORTED_N1N2_R19:
            raise ValueError(
                f"(N1,N2)=({antenna.N1},{antenna.N2}) is not a Release-19 large-array "
                f"configuration (Table 5.2.2.2.1a-1: {sorted(SUPPORTED_N1N2_R19)})"
            )
        if antenna.Ng != 1:
            raise ValueError("Refined Type I single-panel requires Ng=1")
        if N3 < 1:
            raise ValueError("N3 must be positive")
        self.antenna = antenna
        self.N3 = N3
        self.selection_rho = 10 ** (selection_snr_db / 10)

    # -- beam list ----------------------------------------------------------

    def _beams(self, pmi: RefinedType1PMI) -> list[tuple[int, int]]:
        a = self.antenna
        l, m = pmi.i11, pmi.i12
        if pmi.rank == 1:
            return [(l, m)]
        if pmi.rank in (2, 3, 4):
            k1, k2 = i13_lowrank(pmi.rank, pmi.i13, a.O1, a.O2)
            return [(l, m), (l + k1, m + k2)]
        (o1, k1), (o2, k2) = i13_highrank(pmi.i13, l, m, a.O1, a.O2)
        beams = [(l, m)]
        for a1, a2 in zip(pmi.i112, pmi.i122, strict=True):
            beams.append((o1 * a1 + k1, o2 * a2 + k2))
        return beams

    def _n_i2(self, rank: int) -> int:
        return 4 if rank == 1 else 2

    # -- gNB side -----------------------------------------------------------

    def _w_at(self, beams: list[np.ndarray], rank: int, n: int) -> np.ndarray:
        a = self.antenna
        phi = np.exp(1j * np.pi * n / 2)
        columns = []
        for bi, phi_top, s_top, phi_bot, s_bot in _COLUMN_PATTERNS[rank]:
            b = beams[bi]
            top = s_top * (phi if phi_top else 1.0) * b
            bot = s_bot * (phi if phi_bot else 1.0) * b
            columns.append(np.concatenate([top, bot]))
        return np.stack(columns, axis=1) / math.sqrt(rank * a.P)

    def precoder(self, pmi: RefinedType1PMI) -> np.ndarray:
        self._validate(pmi)
        a = self.antenna
        beam_idx = self._beams(pmi)
        beams = [dft.spatial_beam(a, l, m) for l, m in beam_idx]
        W = np.empty((1, self.N3, a.P, pmi.rank), dtype=complex)
        for t in range(self.N3):
            W[0, t] = self._w_at(beams, pmi.rank, int(pmi.i2[t]))
        return W

    # -- UE side ------------------------------------------------------------

    def select(self, H: np.ndarray, rank: int = 1) -> RefinedType1PMI:
        from ._spatial import aligned_eigen_targets

        if not 1 <= rank <= min(8, self.antenna.P):
            raise ValueError(f"Refined Type I rank {rank} unsupported")
        H = np.asarray(H)
        if H.ndim != 4:
            raise ValueError("H must be [slot, t, rx, port]")
        Ht = H[-1]
        if Ht.shape[0] != self.N3:
            raise ValueError(f"channel has {Ht.shape[0]} frequency units, expected {self.N3}")
        if Ht.shape[-1] != self.antenna.P:
            raise ValueError(f"channel has {Ht.shape[-1]} ports, expected {self.antenna.P}")
        a = self.antenna
        targets = aligned_eigen_targets(Ht, rank)  # (N3, P, rank)
        energy = self._beam_energy(targets)  # (G1, G2)
        G1, G2 = a.n_beams

        i11, i12 = (int(x) for x in np.unravel_index(int(np.argmax(energy)), energy.shape))
        pmi = RefinedType1PMI(rank, i11, i12, np.zeros(self.N3, dtype=int))

        if rank in (2, 3, 4):
            pmi.i13 = max(
                range(4),
                key=lambda j: energy[
                    tuple(
                        (i + k) % G
                        for i, k, G in zip(
                            (i11, i12), i13_lowrank(rank, j, a.O1, a.O2), (G1, G2)
                        )
                    )
                ],
            )
        elif rank >= 5:
            pmi.i13, pmi.i112, pmi.i122 = self._select_highrank(energy, i11, i12, rank)

        # optimise the per-subband co-phasing i_2 by spectral efficiency.
        beams = [dft.spatial_beam(a, l, m) for l, m in self._beams(pmi)]
        n_i2 = self._n_i2(rank)
        eye = np.eye(rank, dtype=complex)
        for t in range(self.N3):
            best, best_se = 0, -np.inf
            for n in range(n_i2):
                W = self._w_at(beams, rank, n)
                hw = Ht[t] @ W
                _, logdet = np.linalg.slogdet(eye + self.selection_rho * (hw.conj().T @ hw))
                if np.real(logdet) > best_se:
                    best_se, best = float(np.real(logdet)), n
            pmi.i2[t] = best
        return pmi

    def _beam_energy(self, targets: np.ndarray) -> np.ndarray:
        a = self.antenna
        half = a.P // 2
        grid = dft.spatial_grid(a)  # (G1, G2, half), unit-modulus rows
        pa = np.einsum("xyp,npv->xynv", grid.conj(), targets[:, :half, :])
        pb = np.einsum("xyp,npv->xynv", grid.conj(), targets[:, half:, :])
        return (np.abs(pa) ** 2 + np.abs(pb) ** 2).sum(axis=(2, 3))

    def _select_highrank(self, energy, i11, i12, rank):
        """Greedy companion-beam selection for ranks 5-8.

        Beams are kept orthogonal (spec requirement, 5.2.2.2.1a) by giving each
        companion a *distinct* orthogonal-axis index: for i_{1,3}=0 the
        horizontal index i_{1,1,j} in {0..N1-1} must differ from the base and
        from each other (the (o1,k1)=(O1, i11 mod O1) parametrisation then makes
        all beams mutually orthogonal in the t dimension); symmetrically for
        i_{1,3}=1 in the vertical dimension.
        """
        a = self.antenna
        G1, G2 = a.n_beams
        n_extra = _N_EXTRA[rank]
        best = None
        for i13 in (0, 1):
            (o1, k1), (o2, k2) = i13_highrank(i13, i11, i12, a.O1, a.O2)
            if i13 == 0:  # distinct horizontal orth index; free vertical index
                base_orth, n_orth, n_free = i11 // a.O1, a.N1, a.N2 * a.O2
            else:  # distinct vertical orth index; free horizontal index
                base_orth, n_orth, n_free = i12 // a.O2, a.N2, a.N1 * a.O1
            orth_vals = [x for x in range(n_orth) if x != base_orth]
            if len(orth_vals) < n_extra:
                continue  # not enough orthogonal positions for this i13

            def grid_pos(x_orth, x_free, i13=i13, o1=o1, k1=k1, o2=o2, k2=k2):
                if i13 == 0:
                    return (o1 * x_orth + k1) % G1, (o2 * x_free + k2) % G2
                return (o1 * x_free + k1) % G1, (o2 * x_orth + k2) % G2

            per_orth = []  # (energy, x1, x2) of the best free index for each orth index
            for xo in orth_vals:
                xf = max(range(n_free), key=lambda f: energy[grid_pos(xo, f)])
                pos = grid_pos(xo, xf)
                x1, x2 = (xo, xf) if i13 == 0 else (xf, xo)
                per_orth.append((energy[pos], x1, x2))
            per_orth.sort(key=lambda p: p[0], reverse=True)
            picks = per_orth[:n_extra]
            total = sum(p[0] for p in picks)
            if best is None or total > best[0]:
                best = (total, i13, tuple(p[1] for p in picks), tuple(p[2] for p in picks))
        return best[1], best[2], best[3]

    # -- validation ---------------------------------------------------------

    def _validate(self, pmi: RefinedType1PMI) -> None:
        a = self.antenna
        G1, G2 = a.n_beams
        if not 1 <= pmi.rank <= 8:
            raise ValueError(f"rank {pmi.rank} not in 1..8")
        if not (0 <= pmi.i11 < G1 and 0 <= pmi.i12 < G2):
            raise ValueError("i11/i12 out of range")
        i2 = np.asarray(pmi.i2)
        if i2.shape != (self.N3,) or i2.min() < 0 or i2.max() >= self._n_i2(pmi.rank):
            raise ValueError(f"i2 must be ({self.N3},) with values in [0,{self._n_i2(pmi.rank)})")
        if pmi.rank == 1:
            return
        if pmi.rank in (2, 3, 4):
            if pmi.i13 is None or not 0 <= pmi.i13 < 4:
                raise ValueError("i13 must be in 0..3 for ranks 2-4")
            return
        if pmi.i13 is None or pmi.i13 not in (0, 1):
            raise ValueError("i13 must be in {0,1} for ranks 5-8")
        n_extra = _N_EXTRA[pmi.rank]
        if len(pmi.i112) != n_extra or len(pmi.i122) != n_extra:
            raise ValueError(f"ranks {pmi.rank} need {n_extra} companion beam indices")
        hi1 = a.N1 if pmi.i13 == 0 else a.N1 * a.O1
        hi2 = a.N2 * a.O2 if pmi.i13 == 0 else a.N2
        if any(not 0 <= x < hi1 for x in pmi.i112) or any(not 0 <= x < hi2 for x in pmi.i122):
            raise ValueError("companion beam indices out of range (Table 5.2.2.2.1a-2)")

    # -- overhead -----------------------------------------------------------

    def overhead_bits(self, pmi: RefinedType1PMI) -> dict[str, int]:
        a = self.antenna
        G1, G2 = a.n_beams
        bits = {
            "i11": math.ceil(math.log2(G1)),
            "i2": self.N3 * (2 if pmi.rank == 1 else 1),
        }
        if G2 > 1:
            bits["i12"] = math.ceil(math.log2(G2))
        if pmi.rank in (2, 3, 4):
            bits["i13"] = 2
        elif pmi.rank >= 5:
            bits["i13"] = 1
            n_extra = _N_EXTRA[pmi.rank]
            hi1 = a.N1 if pmi.i13 == 0 else a.N1 * a.O1
            hi2 = a.N2 * a.O2 if pmi.i13 == 0 else a.N2
            bits["i112"] = n_extra * math.ceil(math.log2(hi1))
            if hi2 > 1:
                bits["i122"] = n_extra * math.ceil(math.log2(hi2))
        return bits
