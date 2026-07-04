"""Release-19 Refined Type I Single-Panel codebook, TS 38.214 5.2.2.2.1a.

'typeI-SinglePanel-r19' for the large arrays of Table 5.2.2.2.1a-1 (48/64/128
ports, O1 = O2 = 4), ranks 1-8, both codebookMode-r19 values.

**modeA** -- the per-layer precoding matrices W^(v) (Table 5.2.2.2.1a-4) reuse
the *same* column patterns as the Release-15 Type I codebook (clause
5.2.2.2.1):

    phi_n = e^{j pi n / 2},  v_{l,m} the length-N1*N2 oversampled DFT beam,
    and each column is [b ; +-phi_n b] for a selected beam b.

The Release-19 *refinement* is the beam selection, not the column pattern:

* ranks 2-4 use a fixed i_{1,3} -> (k1,k2) table (Table 5.2.2.2.1a-3) and a
  single companion beam (i_{1,1}+k1, i_{1,2}+k2);
* ranks 5-8 select the 2 or 3 companion beams *independently* via the indices
  i_{1,1,j}, i_{1,2,j}, mapped through the i_{1,3} -> (o1,k1),(o2,k2) row of
  Table 5.2.2.2.1a-3:
      l^(j) = o1 * i_{1,1,j} + k1,   m^(j) = o2 * i_{1,2,j} + k2.

**modeB** -- every layer carries a single beam column [b_l ; phi_{c_l} b_l]
(Table 5.2.2.2.1a-7) with a common orthogonal group i_{1,1} = [q1 q2]:

* ranks 1-4 select one beam *per layer* inside the group, i_{1,2,l} with
  n^(l) = N1*N2 - 1 - i_{1,2,l} (reverse indexing), and a free per-layer
  QPSK co-phasing c_l = i_{2,l} in {0..3};
* ranks 5-8 select L_G in {3, 4} distinct beams via the clause-5.2.2.2.3
  combinatorial index i_{1,2}; layers pair up on the beams and each pair's
  co-phasings are (c, c+2) -- an orthogonal (+phi, -phi) pair -- indicated by
  a 1-bit i_{2,g} (2-bit for the unpaired layer), Table 5.2.2.2.1a-6.

The Refined Type I Multi-Panel codebook (5.2.2.2.2a) lives in
``refined_type1mp_r19``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from math import comb

import numpy as np

from ..config import SUPPORTED_N1N2_R19, AntennaConfig
from ..utils import combinatorics as cb
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


@dataclass
class RefinedType1ModeBPMI:
    """codebookMode-r19 = 'modeB' report (5.2.2.2.1a).

    ``i2`` holds one row per PMI frequency unit: ranks 1-4 carry the per-layer
    QPSK co-phasing i_{2,l} (v columns, values 0-3); ranks 5-8 carry the
    per-group fields i_{2,g} of Table 5.2.2.2.1a-6 (L_G columns, 1-bit for a
    (c, c+2) layer pair, 2-bit for the unpaired layer).
    """

    rank: int
    q1: int  # i_{1,1} = [q1 q2]
    q2: int
    i12l: tuple[int, ...] = field(default_factory=tuple)  # ranks 1-4: i_{1,2,l}
    i12: int | None = None  # ranks 5-8: combinatorial beam-set index i_{1,2}
    i2: np.ndarray | None = None


#: modeB ranks 5-8: i_{2,g} fields in reporting order as (group, n_phases);
#: n_phases = 2 encodes a (c, c+2) co-phasing pair carried by two layers on
#: beam ``group``, n_phases = 4 a free QPSK phase for a single layer
#: (Table 5.2.2.2.1a-6).
_MODEB_FIELDS: dict[int, list[tuple[int, int]]] = {
    5: [(0, 2), (1, 2), (2, 4)],
    6: [(0, 2), (1, 2), (2, 2)],
    7: [(0, 2), (1, 4), (2, 2), (3, 2)],
    8: [(0, 2), (1, 2), (2, 2), (3, 2)],
}

#: modeB ranks 5-8: number of distinct beams L_G selected by i_{1,2}.
_MODEB_LG = {5: 3, 6: 3, 7: 4, 8: 4}


def modeb_layer_fields(rank: int) -> list[tuple[int, int, int | None]]:
    """Per-layer (beam group g, i2 field index, phase offset) for ranks 5-8.

    ``offset`` is 0/2 for the two layers of a (c, c+2) pair and None for the
    free-phase layer (c_l = i_{2,g} directly).
    """
    layers: list[tuple[int, int, int | None]] = []
    for fi, (g, nph) in enumerate(_MODEB_FIELDS[rank]):
        if nph == 2:
            layers.extend([(g, fi, 0), (g, fi, 2)])
        else:
            layers.append((g, fi, None))
    return layers


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
    """TS 38.214 5.2.2.2.1a 'typeI-SinglePanel-r19' (codebookMode-r19
    'modeA' or 'modeB')."""

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int = 1,
        mode: str = "modeA",
        ri_restriction: np.ndarray | None = None,
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
        if mode not in ("modeA", "modeB"):
            raise ValueError("codebookMode-r19 must be 'modeA' or 'modeB'")
        self.antenna = antenna
        self.N3 = N3
        self.mode = mode
        # typeI-SinglePanelRI-Restriction-r19: 8-bit bitmap [r0..r7].
        if ri_restriction is None:
            ri_restriction = np.ones(8, dtype=bool)
        self.ri_restriction = np.asarray(ri_restriction, dtype=bool)
        if self.ri_restriction.shape != (8,):
            raise ValueError(
                "typeI-SinglePanelRI-Restriction-r19 must have 8 bits [r0..r7]"
            )
        self.name = f"R19 Refined Type I single-panel ({mode})"
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

    def precoder(self, pmi) -> np.ndarray:
        if self.mode == "modeB":
            return self._precoder_modeb(pmi)
        self._validate(pmi)
        a = self.antenna
        beam_idx = self._beams(pmi)
        beams = [dft.spatial_beam(a, l, m) for l, m in beam_idx]
        W = np.empty((1, self.N3, a.P, pmi.rank), dtype=complex)
        for t in range(self.N3):
            W[0, t] = self._w_at(beams, pmi.rank, int(pmi.i2[t]))
        return W

    # -- modeB --------------------------------------------------------------

    def _modeb_beams(self, pmi: RefinedType1ModeBPMI) -> list[tuple[int, int]]:
        """Selected beam grid positions: one per layer (ranks 1-4) or the L_G
        distinct beams in i_{1,2} order (ranks 5-8)."""
        a = self.antenna
        if pmi.rank <= 4:
            out = []
            for i12l in pmi.i12l:
                n = a.N1 * a.N2 - 1 - i12l  # n^(l) = N1*N2 - 1 - i_{1,2,l}
                n1, n2 = n % a.N1, n // a.N1
                out.append((a.O1 * n1 + pmi.q1, a.O2 * n2 + pmi.q2))
            return out
        n1, n2 = cb.decode_beam_combination(
            pmi.i12, a.N1, a.N2, _MODEB_LG[pmi.rank]
        )
        return [(a.O1 * x1 + pmi.q1, a.O2 * x2 + pmi.q2) for x1, x2 in zip(n1, n2)]

    def _modeb_layer_cs(self, pmi: RefinedType1ModeBPMI, t: int) -> list[tuple[int, int]]:
        """(beam list index, c_l) per layer at frequency unit t."""
        i2 = np.asarray(pmi.i2)
        if pmi.rank <= 4:
            return [(li, int(i2[t, li])) for li in range(pmi.rank)]
        out = []
        for g, fi, offset in modeb_layer_fields(pmi.rank):
            v = int(i2[t, fi])
            out.append((g, v if offset is None else v + offset))
        return out

    def _precoder_modeb(self, pmi: RefinedType1ModeBPMI) -> np.ndarray:
        self._validate_modeb(pmi)
        a = self.antenna
        beams = [dft.spatial_beam(a, l, m) for l, m in self._modeb_beams(pmi)]
        W = np.empty((1, self.N3, a.P, pmi.rank), dtype=complex)
        scale = math.sqrt(pmi.rank * a.P)
        for t in range(self.N3):
            for li, (bi, c) in enumerate(self._modeb_layer_cs(pmi, t)):
                phi = np.exp(1j * np.pi * c / 2)
                W[0, t, :, li] = np.concatenate([beams[bi], phi * beams[bi]]) / scale
        return W

    # -- UE side ------------------------------------------------------------

    def select(self, H: np.ndarray, rank: int = 1):
        from ._spatial import aligned_eigen_targets

        if not 1 <= rank <= min(8, self.antenna.P):
            raise ValueError(f"Refined Type I rank {rank} unsupported")
        if not self.ri_restriction[rank - 1]:
            raise ValueError(
                f"rank {rank} prohibited by typeI-SinglePanelRI-Restriction-r19 "
                f"(r{rank - 1}=0)"
            )
        H = np.asarray(H)
        if H.ndim != 4:
            raise ValueError("H must be [slot, t, rx, port]")
        Ht = H[-1]
        if Ht.shape[0] != self.N3:
            raise ValueError(f"channel has {Ht.shape[0]} frequency units, expected {self.N3}")
        if Ht.shape[-1] != self.antenna.P:
            raise ValueError(f"channel has {Ht.shape[-1]} ports, expected {self.antenna.P}")
        if self.mode == "modeB":
            return self._select_modeb(Ht, rank)
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

    # -- modeB UE side -------------------------------------------------------

    def _select_modeb(self, Ht: np.ndarray, rank: int) -> RefinedType1ModeBPMI:
        from ._spatial import aligned_eigen_targets

        a = self.antenna
        half = a.P // 2
        targets = aligned_eigen_targets(Ht, rank)  # (N3, P, rank)
        grid = dft.spatial_grid(a)  # (G1, G2, half)
        pa = np.einsum("xyp,npv->xynv", grid.conj(), targets[:, :half, :])
        pb = np.einsum("xyp,npv->xynv", grid.conj(), targets[:, half:, :])
        # per-layer energy on every grid beam, (G1, G2, v)
        energy_l = (np.abs(pa) ** 2 + np.abs(pb) ** 2).sum(axis=2)

        n_beams = a.N1 * a.N2
        best = (-1.0, None)
        for q1 in range(a.O1):
            for q2 in range(a.O2):
                # energies of the orthogonal group's beams, (N1*N2, v) with
                # row n = N1*n2 + n1
                rows = np.array(
                    [
                        energy_l[a.O1 * (n % a.N1) + q1, a.O2 * (n // a.N1) + q2]
                        for n in range(n_beams)
                    ]
                )
                if rank <= 4:
                    score = float(rows.max(axis=0).sum())  # per-layer free choice
                    picks = rows.argmax(axis=0)  # beam n per layer
                else:
                    L_G = _MODEB_LG[rank]
                    tot = rows.sum(axis=1)
                    top = np.sort(np.argsort(tot)[::-1][:L_G])
                    score = float(tot[top].sum())
                    picks = top
                if score > best[0]:
                    best = (score, (q1, q2, picks))
        q1, q2, picks = best[1]

        pmi = RefinedType1ModeBPMI(rank, q1, q2)
        if rank <= 4:
            pmi.i12l = tuple(int(n_beams - 1 - n) for n in picks)
            beam_of_layer = list(range(rank))
        else:
            n1 = [int(n % a.N1) for n in picks]
            n2 = [int(n // a.N1) for n in picks]
            pmi.i12 = cb.encode_beam_combination(n1, n2, a.N1, a.N2)
            beam_of_layer = None  # derived from modeb_layer_fields below

        beams = [dft.spatial_beam(a, l, m) for l, m in self._modeb_beams(pmi)]
        # matched-filter projections per (t, beam, layer): w^H u with
        # w = [b; phi b] gives z0 + conj(phi) z1
        z0 = np.einsum("bp,tpv->tbv", np.conj(beams), targets[:, :half, :])
        z1 = np.einsum("bp,tpv->tbv", np.conj(beams), targets[:, half:, :])
        phis = np.exp(1j * np.pi * np.arange(4) / 2)

        if rank <= 4:
            pmi.i2 = np.zeros((self.N3, rank), dtype=int)
            for t in range(self.N3):
                used: set[tuple[int, int]] = set()
                for li in range(rank):
                    bi = beam_of_layer[li]
                    metric = np.abs(z0[t, bi, li] + np.conj(phis) * z1[t, bi, li])
                    order = np.argsort(metric)[::-1]
                    beam_key = pmi.i12l[li]
                    c = next(
                        (int(x) for x in order if (beam_key, int(x)) not in used),
                        int(order[0]),
                    )
                    used.add((beam_key, c))
                    pmi.i2[t, li] = c
            return pmi

        fields = _MODEB_FIELDS[rank]
        pmi.i2 = np.zeros((self.N3, len(fields)), dtype=int)
        for t in range(self.N3):
            li = 0
            for fi, (g, nph) in enumerate(fields):
                if nph == 4:  # one free-phase layer on beam g
                    metric = np.abs(z0[t, g, li] + np.conj(phis) * z1[t, g, li])
                    pmi.i2[t, fi] = int(np.argmax(metric))
                    li += 1
                else:  # layer pair (c, c+2) on beam g
                    m = [
                        float(
                            np.abs(z0[t, g, li] + np.conj(phis[i]) * z1[t, g, li])
                            + np.abs(
                                z0[t, g, li + 1] + np.conj(phis[i + 2]) * z1[t, g, li + 1]
                            )
                        )
                        for i in (0, 1)
                    ]
                    pmi.i2[t, fi] = int(np.argmax(m))
                    li += 2
        return pmi

    def _validate_modeb(self, pmi: RefinedType1ModeBPMI) -> None:
        a = self.antenna
        n_beams = a.N1 * a.N2
        if not 1 <= pmi.rank <= 8:
            raise ValueError(f"rank {pmi.rank} not in 1..8")
        if not (0 <= pmi.q1 < a.O1 and 0 <= pmi.q2 < a.O2):
            raise ValueError("i11 = [q1 q2] out of range")
        i2 = np.asarray(pmi.i2) if pmi.i2 is not None else None
        if pmi.rank <= 4:
            if len(pmi.i12l) != pmi.rank:
                raise ValueError("i12l needs one entry per layer for ranks 1-4")
            if any(not 0 <= x < n_beams for x in pmi.i12l):
                raise ValueError(f"i_{{1,2,l}} must be in [0, {n_beams})")
            if i2 is None or i2.shape != (self.N3, pmi.rank):
                raise ValueError(f"i2 must be ({self.N3}, {pmi.rank})")
            if i2.min() < 0 or i2.max() > 3:
                raise ValueError("i_{2,l} values must be in 0..3")
            return
        L_G = _MODEB_LG[pmi.rank]
        hi = comb(n_beams, L_G)
        if pmi.i12 is None or not 0 <= pmi.i12 < hi:
            raise ValueError(f"i12 must be in [0, C({n_beams},{L_G}))")
        fields = _MODEB_FIELDS[pmi.rank]
        if i2 is None or i2.shape != (self.N3, len(fields)):
            raise ValueError(f"i2 must be ({self.N3}, {len(fields)})")
        for fi, (_, nph) in enumerate(fields):
            col = i2[:, fi]
            if col.min() < 0 or col.max() >= nph:
                raise ValueError(f"i_{{2,{fi + 1}}} values must be in [0, {nph})")

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

    def overhead_bits(self, pmi) -> dict[str, int]:
        a = self.antenna
        if self.mode == "modeB":
            n_beams = a.N1 * a.N2
            bits = {"i11": math.ceil(math.log2(a.O1)) + math.ceil(math.log2(a.O2))}
            if pmi.rank <= 4:
                bits["i12"] = pmi.rank * math.ceil(math.log2(n_beams))
                bits["i2"] = self.N3 * 2 * pmi.rank
            else:
                L_G = _MODEB_LG[pmi.rank]
                bits["i12"] = math.ceil(math.log2(comb(n_beams, L_G)))
                bits["i2"] = self.N3 * sum(
                    round(math.log2(nph)) for _, nph in _MODEB_FIELDS[pmi.rank]
                )
            return bits
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
