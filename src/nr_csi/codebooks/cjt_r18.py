"""R18 Enhanced Type II codebooks for coherent joint transmission (CJT).

TS 38.214 5.2.2.2.8 'typeII-CJT-r18' and 5.2.2.2.9 'typeII-CJT-PortSelection-
r18': N_TRP in {1,2,3,4} CSI-RS resources (one per transmission point), each
with P_CSI-RS = 2*N1*N2 ports, jointly precoded for one UE.

Reconstruction (Tables 5.2.2.2.8-4 / 5.2.2.2.9-4): the layer-l precoder
stacks one eType II block per selected resource j,

    w^l_t = 1/sqrt(s * gamma_{t,l}) *
        [ e^{j 2 pi t psi_j / N3} *
          [ sum_i v_{m^(i)_j} p1_{l,0} sum_f y_t^(f) p2_{l,i,f,j} phi_{l,i,f,j} ;
            sum_i v_{m^(i)_j} p1_{l,1} sum_f y_t^(f) p2_{l,i+L_j,f,j} phi ... ]
        ]_{j=1..N0}

with s = N1*N2 for the regular codebook (DFT beams per TRP, as in 5.2.2.2.5)
and s = 1 for the port-selection variant (free per-TRP port selection, as in
5.2.2.2.7).  gamma sums over all resources.  CJT-specific ingredients:

* per-TRP spatial bases: resource j selects its own orthogonal group
  (q_{1,j}, q_{2,j}) and L_{sigma_j} beams (regular), or its own
  K_{1,sigma_j} = alpha_{sigma_j} * P ports (PS);
* a common frequency (delay) basis across TRPs: M_v per-layer taps selected
  exactly as in 5.2.2.2.5 (regular) or the M in {1,2} windowed taps of
  5.2.2.2.7, common to all layers (PS);
* inter-TRP co-phasing/delay offsets psi_j = d_j / O3 (codebookMode 'mode1',
  d_j in {0..N3*O3-1}, O3 in {1,4} via *numberOfO3*), applied as a frequency
  ramp e^{j 2 pi t psi_j / N3} on resource j's block; 'mode2' fixes psi = 0;
* reference amplitudes p^(1)_{l,p} per polarization, common to all TRPs;
* CSI-RS resource selection: with *restrictedCMR-Selection* all N_TRP
  resources are used (N0 = N_TRP); otherwise the UE selects N0 resources and
  reports an N_TRP-bit bitmap.

Coefficients are stored on the concatenated beam index
I = 2*sum_{k<j} L_k + i (i in 0..2L_j-1, polarization-split at L_j inside
each resource), so arrays are (v, M, S) with S = sum_j 2*L_j.

Channel/precoder port convention: the aggregated array concatenates the
N_TRP resources' ports (2*N1*N2 each, [pol0; pol1] within a resource);
unselected resources' rows of W are zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb

import numpy as np

from ..config import AntennaConfig, m_v
from ..utils import combinatorics as cb
from ..utils import dft
from ..utils import quantization as qt
from . import _spatial
from .base import CodebookScheme
from .etype2_r16 import select_taps

# ---------------------------------------------------------------------------
# configuration tables
# ---------------------------------------------------------------------------

#: Table 5.2.2.2.8-1: paramCombination-CJT-L-r18 -> {L_1..L_NTRP} per N_TRP.
CJT_L_COMBOS: dict[int, dict[int, tuple[int, ...]]] = {
    1: {1: (2,), 2: (4,), 3: (6,)},
    2: {1: (2, 2), 2: (2, 4), 3: (4, 2), 4: (4, 4)},
    3: {1: (2, 2, 2), 2: (2, 2, 4), 3: (2, 4, 2), 4: (4, 2, 2), 5: (4, 4, 4)},
    4: {1: (2, 2, 2, 2), 2: (2, 2, 2, 4), 3: (2, 2, 4, 4), 4: (4, 4, 4, 4)},
}

#: Table 5.2.2.2.8-2: paramCombination-CJT-r18 -> (p_v12, p_v34, beta).
CJT_PV_BETA: dict[int, tuple[Fraction, Fraction, Fraction]] = {
    1: (Fraction(1, 8), Fraction(1, 16), Fraction(1, 4)),
    2: (Fraction(1, 8), Fraction(1, 16), Fraction(1, 2)),
    3: (Fraction(1, 4), Fraction(1, 8), Fraction(1, 4)),
    4: (Fraction(1, 4), Fraction(1, 8), Fraction(1, 2)),
    5: (Fraction(1, 4), Fraction(1, 4), Fraction(3, 4)),
    6: (Fraction(1, 2), Fraction(1, 4), Fraction(1, 2)),
    7: (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
}

#: Table 5.2.2.2.8-3: allowed paramCombination-CJT-r18 per (N_TRP, pcL).
CJT_ALLOWED: dict[int, dict[int, set[int]]] = {
    1: {1: {3, 4}, 2: {3, 4, 5, 6}, 3: {4, 5}},
    2: {1: {1}, 2: {1}, 3: {1}, 4: {2, 4, 7}},
    3: {1: {1, 2}, 2: {1, 2}, 3: {1, 2}, 4: {1, 2}, 5: {1, 2, 3, 4, 5, 7}},
    4: {1: {1}, 2: {1}, 3: {4, 5}, 4: {2, 4, 5}},
}

#: 5.2.2.2.8: paramCombination-CJT-L-r18 values barred at P_CSI-RS = 4.
CJT_L_BARRED_P4: dict[int, set[int]] = {
    1: {2, 3},
    2: {2, 3, 4},
    3: {2, 3, 4, 5},
    4: {2, 3, 4},
}

#: Table 5.2.2.2.9-1: paramCombination-CJT-PS-alpha-r18 -> {alpha_1..}.
CJT_PS_ALPHA_COMBOS: dict[int, dict[int, tuple[Fraction, ...]]] = {
    1: {1: (Fraction(1, 2),), 2: (Fraction(3, 4),), 3: (Fraction(1),)},
    2: {
        1: (Fraction(1, 2), Fraction(1, 2)),
        2: (Fraction(1, 2), Fraction(1)),
        3: (Fraction(1), Fraction(1, 2)),
        4: (Fraction(3, 4), Fraction(3, 4)),
        5: (Fraction(1), Fraction(1)),
    },
    3: {
        1: (Fraction(1, 2),) * 3,
        2: (Fraction(1, 2), Fraction(1, 2), Fraction(3, 4)),
        3: (Fraction(1, 2), Fraction(3, 4), Fraction(1, 2)),
        4: (Fraction(3, 4), Fraction(1, 2), Fraction(1, 2)),
        5: (Fraction(1, 2), Fraction(1, 2), Fraction(1)),
        6: (Fraction(1, 2), Fraction(1), Fraction(1, 2)),
        7: (Fraction(1), Fraction(1, 2), Fraction(1, 2)),
        8: (Fraction(1),) * 3,
    },
    4: {
        1: (Fraction(1, 2),) * 4,
        2: (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1)),
        3: (Fraction(1, 2), Fraction(1, 2), Fraction(1), Fraction(1)),
        4: (Fraction(1),) * 4,
    },
}

#: Table 5.2.2.2.9-2: paramCombination-CJT-PS-r18 -> (M, beta).
CJT_PS_M_BETA: dict[int, tuple[int, Fraction]] = {
    1: (1, Fraction(1, 2)),
    2: (1, Fraction(3, 4)),
    3: (1, Fraction(1)),
    4: (2, Fraction(1, 2)),
    5: (2, Fraction(3, 4)),
}

#: Table 5.2.2.2.9-3: allowed paramCombination-CJT-PS-r18 per (N_TRP, pcA).
CJT_PS_ALLOWED: dict[int, dict[int, set[int]]] = {
    1: {1: {4}, 2: {1, 4}, 3: {1, 2, 3, 4, 5}},
    2: {1: {1, 4}, 2: {1}, 3: {1}, 4: {2}, 5: {2, 4}},
    3: {1: {1, 4}, 2: {1}, 3: {1}, 4: {1}, 5: {2, 4}, 6: {2, 4}, 7: {2, 4}, 8: {2, 5}},
    4: {1: {1}, 2: {1}, 3: {2, 3, 4}, 4: {3}},
}

#: 5.2.2.2.9: paramCombination-CJT-PS-alpha-r18 barred at P in {4, 12}.
CJT_PS_ALPHA_BARRED_P4_12: dict[int, set[int]] = {1: {2}, 2: {4}, 3: {2, 3, 4}, 4: set()}

#: 5.2.2.2.9: pcA barred when pc-PS in {4, 5} and P = 32.
CJT_PS_ALPHA_BARRED_P32_M2: dict[int, set[int]] = {1: {3}, 2: {5}, 3: {8}, 4: {4}}

#: 5.2.2.2.9: pcA barred at P = 4 when a rank > 2 is allowed.
CJT_PS_ALPHA_BARRED_P4_RANK34: dict[int, set[int]] = {
    1: {1},
    2: {1, 2, 3},
    3: {1, 2, 3, 4, 5, 6, 7},
    4: {1, 2, 3},
}


@dataclass
class R18CJTPMI:
    """Report for both CJT variants; per-resource fields are ordered by the
    selected resources sigma_1 < ... < sigma_N0 (0-based indices)."""

    rank: int
    i_L: int = 0  # index of the selected {L_1..} / {alpha_1..} combination
    resources: tuple[int, ...] = ()  # selected CSI-RS resources (0-based)
    # regular: per-resource orthogonal group + beam combination
    q1: tuple[int, ...] = ()
    q2: tuple[int, ...] = ()
    i12: tuple[int | None, ...] = ()  # PS: port combination (None if alpha=1)
    # frequency part
    i15: int | None = None  # regular, N3 > 19 only
    i16: list[int] = field(default_factory=list)  # regular: per layer; PS: [offset]
    i19: tuple[int, ...] = ()  # mode1: d_j for j = 2..N0
    # coefficients on the concatenated index I (v, M, S)
    i17: np.ndarray | None = None
    i18: list[int] = field(default_factory=list)
    k1: np.ndarray | None = None  # (v, 2)
    k2: np.ndarray | None = None
    c: np.ndarray | None = None


class _CJTBase(CodebookScheme):
    """Shared machinery of the two CJT variants."""

    N_PSK = 16

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        n_trp: int,
        mode: str,
        O3: int,
        R: int,
        restricted_cmr: bool,
        n0: int | None,
        ri_restriction: np.ndarray | None,
    ) -> None:
        if n_trp not in (1, 2, 3, 4):
            raise ValueError("N_TRP must be in {1,2,3,4}")
        if antenna.Ng != 1:
            raise ValueError("CJT resources are single-panel (Ng=1) each")
        if N3 < 1:
            raise ValueError("N3 must be positive")
        if mode not in ("mode1", "mode2"):
            raise ValueError("codebookMode must be 'mode1' or 'mode2'")
        if O3 not in (1, 4):
            raise ValueError("numberOfO3 must be 1 or 4")
        if R not in (1, 2):
            raise ValueError("R must be 1 or 2")
        self.antenna = antenna
        self.N3 = N3
        self.n_trp = n_trp
        self.mode = mode
        self.O3 = O3
        self.R = R
        self.restricted_cmr = restricted_cmr
        if restricted_cmr:
            if n0 is not None and n0 != n_trp:
                raise ValueError("restrictedCMR-Selection fixes N0 = N_TRP")
            n0 = n_trp
        else:
            if n0 is None:
                n0 = n_trp
            if not 1 <= n0 <= n_trp:
                raise ValueError("N0 must satisfy 1 <= N0 <= N_TRP")
        self.n0 = n0
        if ri_restriction is None:
            ri_restriction = np.ones(4, dtype=bool)
        self.ri_restriction = np.asarray(ri_restriction, dtype=bool)
        if self.ri_restriction.shape != (4,):
            raise ValueError("RI restriction must have 4 bits [r0..r3]")

    # -- combination bookkeeping (subclasses define self.combos: list of
    #    per-resource L tuples over the N_TRP resources) ---------------------

    @property
    def N_L(self) -> int:
        return len(self.combos)

    def Ls(self, pmi: R18CJTPMI) -> list[int]:
        """L_{sigma_j} for the selected resources, in resource order."""
        combo = self.combos[pmi.i_L]
        return [combo[r] for r in pmi.resources]

    def _offsets(self, Ls: list[int]) -> list[int]:
        """Start of each resource's block on the concatenated index I."""
        out, acc = [], 0
        for L in Ls:
            out.append(acc)
            acc += 2 * L
        return out

    def _S(self, Ls: list[int]) -> int:
        return int(2 * sum(Ls))

    def _pol_of(self, Ls: list[int], I: int) -> int:
        for off, L in zip(self._offsets(Ls), Ls):
            if off <= I < off + 2 * L:
                return (I - off) // L
        raise ValueError("concatenated index out of range")

    def _ramp(self, pmi: R18CJTPMI) -> np.ndarray:
        """Inter-TRP phase ramps e^{j 2 pi t psi_j / N3}, shape (N0, N3)."""
        t = np.arange(self.N3)
        psi = np.zeros(self.n0_of(pmi))
        if self.mode == "mode1":
            psi[1:] = np.asarray(pmi.i19, dtype=float) / self.O3
        return np.exp(2j * np.pi * np.outer(psi, t) / self.N3)

    def n0_of(self, pmi: R18CJTPMI) -> int:
        return len(pmi.resources)

    # -- shared coefficient quantization (mirrors etype2_r16) ----------------

    def _quantize(
        self,
        pmi: R18CJTPMI,
        kept: list[np.ndarray],
        stars: list[tuple[int, int]],
        Ls: list[int],
        M: int,
    ) -> None:
        """Fill k1/k2/c from the pruned per-layer coefficients (S, M)."""
        S = self._S(Ls)
        for li in range(pmi.rank):
            x, bm = kept[li], pmi.i17[li].T  # (S, M)
            i_star, f_star = stars[li]
            p_star = self._pol_of(Ls, i_star)
            mags = np.abs(x)
            pols = np.array([self._pol_of(Ls, i) for i in range(S)])
            for pol in (0, 1):
                if pol == p_star:
                    pmi.k1[li, pol] = 15
                    continue
                rows = pols == pol
                sel = mags[rows][bm[rows]]
                ref = sel.max() if sel.size else 0.0
                pmi.k1[li, pol] = max(int(qt.quantize_amplitude(ref, qt.R16_REF_AMP)), 1)
            p1 = qt.R16_REF_AMP[pmi.k1[li]]
            for i in range(S):
                for f in range(M):
                    if not bm[i, f]:
                        continue
                    rel = mags[i, f] / p1[pols[i]]
                    pmi.k2[li, f, i] = int(
                        qt.quantize_amplitude(min(rel, 1.0), qt.R16_DIFF_AMP)
                    )
                    pmi.c[li, f, i] = int(
                        qt.quantize_phase(np.angle(x[i, f]), self.N_PSK)
                    )
            pmi.k2[li, f_star, i_star] = 7
            pmi.c[li, f_star, i_star] = 0

    def _enforce_total_budget(self, pmi, kept, stars, K0: int) -> None:
        budget = 2 * K0
        total = int(pmi.i17.sum())
        if total <= budget:
            return
        entries = []
        for li in range(pmi.rank):
            bm = pmi.i17[li].T
            i_star, f_star = stars[li]
            for idx in np.argwhere(bm):
                i, f = (int(v) for v in idx)
                if not (i == i_star and f == f_star):
                    entries.append((abs(kept[li][i, f]), li, i, f))
        entries.sort()
        for mag, li, i, f in entries:
            if total <= budget:
                break
            pmi.i17[li][f, i] = False
            kept[li][i, f] = 0.0
            total -= 1

    def _prune_layer(self, x: np.ndarray, K0: int) -> np.ndarray:
        """Per-layer bitmap of the K0 strongest coefficients, (S, M) -> bool."""
        mags = np.abs(x)
        order = np.argsort(mags.reshape(-1))[::-1]
        keep = order[: min(K0, mags.size)]
        bm = np.zeros(mags.shape, dtype=bool)
        bm[np.unravel_index(keep, mags.shape)] = True
        bm &= mags > qt.R16_DIFF_AMP[0] / 2
        return bm

    def _select_resources(self, targets: np.ndarray) -> tuple[int, ...]:
        """Strongest-N0 CSI-RS resource selection (all when restricted)."""
        if self.restricted_cmr or self.n0 == self.n_trp:
            return tuple(range(self.n_trp))
        P = self.antenna.P
        energy = [
            float(np.sum(np.abs(targets[:, r * P : (r + 1) * P, :]) ** 2))
            for r in range(self.n_trp)
        ]
        return tuple(sorted(np.argsort(energy)[::-1][: self.n0].tolist()))

    def _estimate_offsets(self, C_layers: list[np.ndarray], Ls: list[int]) -> tuple[int, ...]:
        """mode1: per-resource delay offsets d_j on the O3-oversampled grid,
        relative to the first selected resource (dominant fractional tap of
        each resource's coefficient block)."""
        n_fft = self.O3 * self.N3
        offsets = self._offsets(Ls)
        taps = []
        for off, L in zip(offsets, Ls):
            E = np.zeros(n_fft)
            for C in C_layers:  # (S, N3) per layer
                block = C[off : off + 2 * L]
                E += np.sum(np.abs(np.fft.fft(block, n=n_fft, axis=1)) ** 2, axis=0)
            taps.append(int(np.argmax(E)))
        return tuple((t - taps[0]) % n_fft for t in taps[1:])

    def _deramp(self, C: np.ndarray, pmi: R18CJTPMI, Ls: list[int]) -> np.ndarray:
        """Remove the reported inter-TRP ramps from coefficients (S, N3)."""
        out = C.copy()
        ramp = self._ramp(pmi)  # (N0, N3)
        for j, (off, L) in enumerate(zip(self._offsets(Ls), Ls)):
            out[off : off + 2 * L] *= np.conj(ramp[j])[None, :]
        return out


def _encode_i18(bitmap_l: np.ndarray, i_star: int, f_star: int, rank: int, S: int) -> int:
    """Strongest-coefficient indicator: rank 1 counts the nonzero f = f*-row
    bits up to I = i* (after remapping f* = 0 for the regular variant);
    ranks 2-4 report the flat index directly."""
    if rank == 1:
        bits = bitmap_l[f_star, :]
        return int(np.cumsum(bits)[i_star]) - 1
    return f_star * S + i_star


class R18CJTCodebook(_CJTBase):
    """TS 38.214 5.2.2.2.8 'typeII-CJT-r18' (eType II per TRP)."""

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        n_trp: int = 2,
        param_combination_L: int | list[int] | tuple[int, ...] = 1,
        param_combination: int = 1,
        mode: str = "mode1",
        O3: int = 4,
        R: int = 1,
        restricted_cmr: bool = True,
        n0: int | None = None,
        ri_restriction: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            antenna, N3, n_trp, mode, O3, R, restricted_cmr, n0, ri_restriction
        )
        pcls = (
            [param_combination_L]
            if isinstance(param_combination_L, int)
            else list(param_combination_L)
        )
        if len(pcls) not in (1, 2, 4):
            raise ValueError("N_L (configured L-combinations) must be 1, 2 or 4")
        rank34 = bool(self.ri_restriction[2] or self.ri_restriction[3])
        for pcl in pcls:
            if pcl not in CJT_L_COMBOS[n_trp]:
                raise ValueError(
                    f"paramCombination-CJT-L-r18={pcl} undefined for N_TRP={n_trp}"
                )
            if antenna.P == 4 and pcl in CJT_L_BARRED_P4[n_trp]:
                raise ValueError(
                    f"paramCombination-CJT-L-r18={pcl} is not supported at "
                    f"P_CSI-RS=4 for N_TRP={n_trp}"
                )
            if n_trp == 1 and pcl == 3:
                if antenna.P < 32:
                    raise ValueError(
                        "paramCombination-CJT-L-r18=3 (L=6) requires P_CSI-RS >= 32"
                    )
                if R == 2 or rank34:
                    raise ValueError(
                        "paramCombination-CJT-L-r18=3 requires R=1 and ranks 3-4 "
                        "disallowed (typeII-RI-Restriction-r18)"
                    )
            if param_combination not in CJT_ALLOWED[n_trp][pcl]:
                raise ValueError(
                    f"paramCombination-CJT-r18={param_combination} is not "
                    f"configurable with paramCombination-CJT-L-r18={pcl} for "
                    f"N_TRP={n_trp} (Table 5.2.2.2.8-3)"
                )
        if param_combination not in CJT_PV_BETA:
            raise ValueError(f"paramCombination-CJT-r18={param_combination} undefined")
        self.combos = [CJT_L_COMBOS[n_trp][pcl] for pcl in pcls]
        self.p_v12, self.p_v34, self.beta = CJT_PV_BETA[param_combination]
        for combo in self.combos:
            if max(combo) > antenna.n_ports_per_pol:
                raise ValueError(
                    f"L={max(combo)} beams cannot be drawn from an orthogonal "
                    f"group of N1*N2={antenna.n_ports_per_pol} beams"
                )
        self.name = f"R18 eType II CJT (N_TRP={n_trp})"

    def p_v(self, rank: int) -> Fraction:
        if rank in (1, 2):
            return self.p_v12
        if rank in (3, 4):
            return self.p_v34
        raise ValueError("rank must be 1..4")

    def Mv(self, rank: int) -> int:
        val = m_v(self.p_v(rank), self.N3, self.R)
        if val > self.N3:
            raise ValueError("M_v exceeds N3")
        return val

    def K0_of(self, Ls: list[int]) -> int:
        M1 = m_v(self.p_v12, self.N3, self.R)
        return math.ceil(2 * self.beta * M1 * sum(Ls))

    # -- gNB side ------------------------------------------------------------

    def _bases(self, pmi: R18CJTPMI, Ls: list[int]) -> list[np.ndarray]:
        return [
            _spatial.basis_regular(self.antenna, q1, q2, i12, L)
            for q1, q2, i12, L in zip(pmi.q1, pmi.q2, pmi.i12, Ls)
        ]

    def precoder(self, pmi: R18CJTPMI) -> np.ndarray:
        self._validate(pmi)
        a = self.antenna
        Ls = self.Ls(pmi)
        S = self._S(Ls)
        Mv = self.Mv(pmi.rank)
        bases = self._bases(pmi, Ls)
        ramp = self._ramp(pmi)  # (N0, N3)
        offsets = self._offsets(Ls)
        pols = np.array([self._pol_of(Ls, i) for i in range(S)])
        W = np.zeros((1, self.N3, self.n_trp * a.P, pmi.rank), dtype=complex)
        for li in range(pmi.rank):
            taps = cb.decode_taps(pmi.i16[li], self.N3, Mv, pmi.i15)
            Y = dft.freq_basis(self.N3, np.array(taps))  # (Mv, N3)
            p1 = qt.R16_REF_AMP[pmi.k1[li]]
            p2 = qt.R16_DIFF_AMP[pmi.k2[li]]  # (Mv, S)
            phi = qt.phase_value(pmi.c[li], self.N_PSK)
            x = (p2 * phi * pmi.i17[li]).T * p1[pols][:, None]  # (S, Mv)
            ct = x @ Y  # (S, N3)
            gamma = np.sum(np.abs(ct) ** 2, axis=0)
            gamma = np.where(gamma == 0, 1.0, gamma)
            for j, (r, off, L, B) in enumerate(
                zip(pmi.resources, offsets, Ls, bases)
            ):
                block = np.concatenate(
                    [B.T @ ct[off : off + L], B.T @ ct[off + L : off + 2 * L]]
                )  # (P, N3)
                block = block * ramp[j][None, :]
                W[0, :, r * a.P : (r + 1) * a.P, li] = (
                    block / np.sqrt(a.n_ports_per_pol * gamma)
                ).T
        return W / np.sqrt(pmi.rank)

    # -- UE side ---------------------------------------------------------------

    def select(self, H: np.ndarray, rank: int = 1) -> R18CJTPMI:
        if not 1 <= rank <= 4:
            raise ValueError("CJT eType II supports ranks 1-4")
        if not self.ri_restriction[rank - 1]:
            raise ValueError(
                f"rank {rank} prohibited by typeII-RI-Restriction-r18 (r{rank - 1}=0)"
            )
        a = self.antenna
        H = np.asarray(H)[-1]
        if H.shape[0] != self.N3:
            raise ValueError(f"channel has {H.shape[0]} frequency units, expected {self.N3}")
        if H.shape[-1] != self.n_trp * a.P:
            raise ValueError(
                f"channel has {H.shape[-1]} ports, expected N_TRP*P = {self.n_trp * a.P}"
            )
        targets = _spatial.aligned_eigen_targets(H, rank)  # (N3, N_TRP*P, v)
        Mv = self.Mv(rank)

        pmi = R18CJTPMI(rank=rank)
        pmi.resources = self._select_resources(targets)
        # combination selection: the configured combination with the most
        # spatial DoF over the selected resources (UE algorithm choice)
        pmi.i_L = max(
            range(self.N_L), key=lambda i: sum(self.combos[i][r] for r in pmi.resources)
        )
        Ls = self.Ls(pmi)
        S = self._S(Ls)
        K0 = self.K0_of(Ls)

        q1s, q2s, i12s, coeff_blocks = [], [], [], []
        for r, L in zip(pmi.resources, Ls):
            seg = targets[:, r * a.P : (r + 1) * a.P, :]
            q1, q2, i12 = _spatial.select_group_and_beams(a, seg, L)
            q1s.append(q1)
            q2s.append(q2)
            i12s.append(i12)
            B = _spatial.basis_regular(a, q1, q2, i12, L)
            coeff_blocks.append(
                _spatial.ls_coefficients(B, seg, float(a.n_ports_per_pol))
            )  # (v, N3, 2L)
        pmi.q1, pmi.q2, pmi.i12 = tuple(q1s), tuple(q2s), tuple(i12s)
        coeff = np.concatenate(coeff_blocks, axis=2)  # (v, N3, S)

        # per-layer global phase alignment (the eigenvector's per-subband
        # phase ambiguity is common to all resources)
        C_layers = []
        for li in range(rank):
            C = coeff[li].T  # (S, N3)
            iref = int(np.argmax(np.sum(np.abs(C) ** 2, axis=1)))
            C_layers.append(C * np.exp(-1j * np.angle(C[iref]))[None, :])

        # inter-TRP delay offsets (mode1), then de-ramp before tap selection
        if self.mode == "mode1" and len(pmi.resources) > 1:
            pmi.i19 = self._estimate_offsets(C_layers, Ls)
        C_layers = [self._deramp(C, pmi, Ls) for C in C_layers]

        pmi.i17 = np.zeros((rank, Mv, S), dtype=bool)
        pmi.k1 = np.ones((rank, 2), dtype=int)
        pmi.k2 = np.zeros((rank, Mv, S), dtype=int)
        pmi.c = np.zeros((rank, Mv, S), dtype=int)

        kept, stars = [], []
        m_init_common: int | None = None
        for li in range(rank):
            C = C_layers[li]
            Ctap = np.fft.fft(C, axis=1) / self.N3  # (S, N3)
            tap_energy = np.sum(np.abs(Ctap) ** 2, axis=0)
            n_star = int(np.argmax(tap_energy))
            Ctap = np.roll(Ctap, -n_star, axis=1)
            taps = select_taps(np.roll(tap_energy, -n_star), Mv, self.N3, m_init_common)
            i16, i15 = cb.encode_taps(taps, self.N3, Mv, m_initial=m_init_common)
            pmi.i16.append(i16)
            if i15 is not None and m_init_common is None:
                m_init_common = i15 if i15 == 0 else i15 - 2 * Mv
            Ct = Ctap[:, taps]  # (S, Mv)
            i_star = int(np.argmax(np.abs(Ct[:, 0])))
            stars.append((i_star, 0))  # strongest tap remapped to f = 0
            Ct = Ct / Ct[i_star, 0]
            bm = self._prune_layer(Ct, K0)
            bm[i_star, 0] = True
            pmi.i17[li] = bm.T
            kept.append(Ct * bm)
        if m_init_common is not None:
            pmi.i15 = m_init_common if m_init_common == 0 else m_init_common + 2 * Mv

        self._enforce_total_budget(pmi, kept, stars, K0)
        self._quantize(pmi, kept, stars, Ls, Mv)
        for li in range(rank):
            pmi.i18.append(_encode_i18(pmi.i17[li], stars[li][0], 0, rank, S))
        return pmi

    # -- validation ------------------------------------------------------------

    def _validate(self, pmi: R18CJTPMI) -> None:
        a = self.antenna
        if not 1 <= pmi.rank <= 4:
            raise ValueError(f"rank {pmi.rank} not in 1..4")
        if not 0 <= pmi.i_L < self.N_L:
            raise ValueError(f"i_L={pmi.i_L} not in [0, {self.N_L})")
        n0 = len(pmi.resources)
        if not 1 <= n0 <= self.n_trp or sorted(set(pmi.resources)) != list(pmi.resources):
            raise ValueError("resources must be distinct, increasing, within N_TRP")
        if self.restricted_cmr and n0 != self.n_trp:
            raise ValueError("restrictedCMR-Selection requires all resources selected")
        if any(r >= self.n_trp or r < 0 for r in pmi.resources):
            raise ValueError("resource index out of range")
        Ls = self.Ls(pmi)
        S = self._S(Ls)
        Mv = self.Mv(pmi.rank)
        if len(pmi.q1) != n0 or len(pmi.q2) != n0 or len(pmi.i12) != n0:
            raise ValueError("q1/q2/i12 need one entry per selected resource")
        for q1, q2, i12, L in zip(pmi.q1, pmi.q2, pmi.i12, Ls):
            if not (0 <= q1 < a.O1 and 0 <= q2 < a.O2):
                raise ValueError("per-resource (q1, q2) out of range")
            if i12 is None or not 0 <= i12 < comb(a.N1 * a.N2, L):
                raise ValueError("per-resource i12 out of range")
        if self.mode == "mode1":
            if len(pmi.i19) != n0 - 1 or any(
                not 0 <= d < self.N3 * self.O3 for d in pmi.i19
            ):
                raise ValueError(f"i19 needs {n0 - 1} offsets in [0, {self.N3 * self.O3})")
        elif pmi.i19:
            raise ValueError("i19 must not be reported in mode2")
        if len(pmi.i16) != pmi.rank:
            raise ValueError("i16 must have one entry per layer")
        if self.N3 > 19:
            if pmi.i15 is None or not 0 <= pmi.i15 < 2 * Mv:
                raise ValueError(f"i15={pmi.i15} not in [0, {2 * Mv})")
        elif pmi.i15 is not None:
            raise ValueError("i15 must not be reported for N3 <= 19")
        shape = (pmi.rank, Mv, S)
        for name, arr, lo, hi in (
            ("i17", pmi.i17, 0, 1),
            ("k2", pmi.k2, 0, 7),
            ("c", pmi.c, 0, self.N_PSK - 1),
        ):
            A = np.asarray(arr)
            if A.shape != shape:
                raise ValueError(f"{name} shape {A.shape} != {shape}")
            if A.dtype != bool and (A.min() < lo or A.max() > hi):
                raise ValueError(f"{name} values outside [{lo}, {hi}]")
        if np.asarray(pmi.k1).shape != (pmi.rank, 2):
            raise ValueError("k1 must be (v, 2)")
        for li in range(pmi.rank):
            i_star, f_star = self._decode_star(pmi, li, S)
            if not pmi.i17[li, f_star, i_star]:
                raise ValueError(f"layer {li}: strongest coefficient absent from bitmap")
            if pmi.k1[li, self._pol_of(Ls, i_star)] != 15:
                raise ValueError(f"layer {li}: strongest polarization must have k1=15")
            if pmi.k2[li, f_star, i_star] != 7 or pmi.c[li, f_star, i_star] != 0:
                raise ValueError(f"layer {li}: strongest coefficient must have k2=7, c=0")

    def _decode_star(self, pmi: R18CJTPMI, li: int, S: int) -> tuple[int, int]:
        if pmi.rank == 1:
            bits = pmi.i17[li][0, :]
            return int(np.argmax(np.cumsum(bits) == pmi.i18[li] + 1)), 0
        return pmi.i18[li] % S, pmi.i18[li] // S

    # -- overhead --------------------------------------------------------------

    def overhead_bits(self, pmi: R18CJTPMI) -> dict[str, int]:
        a = self.antenna
        Ls = self.Ls(pmi)
        S = self._S(Ls)
        Mv, v = self.Mv(pmi.rank), pmi.rank
        bits: dict[str, int] = {}
        if self.N_L > 1:
            bits["i_L"] = math.ceil(math.log2(self.N_L))
        if not self.restricted_cmr:
            bits["cmr"] = self.n_trp
        bits["i11"] = len(pmi.resources) * math.ceil(math.log2(a.O1 * a.O2))
        bits["i12"] = sum(
            math.ceil(math.log2(comb(a.N1 * a.N2, L))) for L in Ls
        )
        if self.N3 > 19:
            bits["i15"] = math.ceil(math.log2(2 * Mv))
            bits["i16"] = v * math.ceil(math.log2(comb(2 * Mv - 1, Mv - 1)))
        elif Mv > 1:
            bits["i16"] = v * math.ceil(math.log2(comb(self.N3 - 1, Mv - 1)))
        if self.mode == "mode1" and len(pmi.resources) > 1:
            bits["i19"] = (len(pmi.resources) - 1) * math.ceil(
                math.log2(self.N3 * self.O3)
            )
        bits["i17"] = v * Mv * S
        K_nz = int(pmi.i17.sum())
        if v == 1:
            bits["i18"] = math.ceil(math.log2(K_nz)) if K_nz > 1 else 0
        else:
            bits["i18"] = v * math.ceil(math.log2(S))
        bits["i23"] = 4 * v
        bits["i24"] = 3 * (K_nz - v)
        bits["i25"] = 4 * (K_nz - v)
        return bits


class R18CJTPortSelectionCodebook(_CJTBase):
    """TS 38.214 5.2.2.2.9 'typeII-CJT-PortSelection-r18' (feType II per TRP)."""

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        n_trp: int = 2,
        param_combination_alpha: int | list[int] | tuple[int, ...] = 1,
        param_combination: int = 1,
        N_window: int = 4,
        mode: str = "mode1",
        O3: int = 4,
        R: int = 1,
        restricted_cmr: bool = True,
        n0: int | None = None,
        ri_restriction: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            antenna, N3, n_trp, mode, O3, R, restricted_cmr, n0, ri_restriction
        )
        pcas = (
            [param_combination_alpha]
            if isinstance(param_combination_alpha, int)
            else list(param_combination_alpha)
        )
        if len(pcas) not in (1, 2, 4):
            raise ValueError("N_L (configured alpha-combinations) must be 1, 2 or 4")
        if param_combination not in CJT_PS_M_BETA:
            raise ValueError(
                f"paramCombination-CJT-PS-r18={param_combination} undefined"
            )
        rank34 = bool(self.ri_restriction[2] or self.ri_restriction[3])
        for pca in pcas:
            if pca not in CJT_PS_ALPHA_COMBOS[n_trp]:
                raise ValueError(
                    f"paramCombination-CJT-PS-alpha-r18={pca} undefined for "
                    f"N_TRP={n_trp}"
                )
            if antenna.P in (4, 12) and pca in CJT_PS_ALPHA_BARRED_P4_12[n_trp]:
                raise ValueError(
                    f"paramCombination-CJT-PS-alpha-r18={pca} is not supported at "
                    f"P_CSI-RS={antenna.P} for N_TRP={n_trp}"
                )
            if (
                antenna.P == 32
                and param_combination in (4, 5)
                and pca in CJT_PS_ALPHA_BARRED_P32_M2[n_trp]
            ):
                raise ValueError(
                    f"paramCombination-CJT-PS-alpha-r18={pca} is not supported at "
                    f"P_CSI-RS=32 with paramCombination-CJT-PS-r18 in {{4,5}}"
                )
            if (
                antenna.P == 4
                and rank34
                and pca in CJT_PS_ALPHA_BARRED_P4_RANK34[n_trp]
            ):
                raise ValueError(
                    f"paramCombination-CJT-PS-alpha-r18={pca} at P_CSI-RS=4 "
                    f"requires ranks 3-4 disallowed "
                    f"(typeII-PortSelectionRI-Restriction-r18)"
                )
            if param_combination not in CJT_PS_ALLOWED[n_trp][pca]:
                raise ValueError(
                    f"paramCombination-CJT-PS-r18={param_combination} is not "
                    f"configurable with paramCombination-CJT-PS-alpha-r18={pca} "
                    f"for N_TRP={n_trp} (Table 5.2.2.2.9-3)"
                )
        self.alpha_combos = [CJT_PS_ALPHA_COMBOS[n_trp][pca] for pca in pcas]
        self.M, self.beta = CJT_PS_M_BETA[param_combination]
        if N_window not in (2, 4):
            raise ValueError("valueOfN-CJT-r18 must be 2 or 4")
        self.N_window = N_window
        if self.M == 1 and R == 2:
            raise ValueError("R = 2 requires M = 2")
        if self.M > N3:
            raise ValueError(f"M={self.M} selected taps cannot exceed N3={N3}")
        # K1 = alpha * P must be a positive even integer for every resource
        self.combos = []
        for combo in self.alpha_combos:
            Ls = []
            for alpha in combo:
                K1 = alpha * antenna.P
                if K1.denominator != 1 or int(K1) % 2 != 0 or K1 <= 0:
                    raise ValueError(
                        f"alpha*P_CSI-RS = {alpha}*{antenna.P} must be a positive "
                        f"even integer"
                    )
                Ls.append(int(K1) // 2)
            self.combos.append(tuple(Ls))
        self.name = f"R18 feType II CJT PS (N_TRP={n_trp})"

    def alphas(self, pmi: R18CJTPMI) -> list[Fraction]:
        combo = self.alpha_combos[pmi.i_L]
        return [combo[r] for r in pmi.resources]

    def K0_of(self, Ls: list[int]) -> int:
        return math.ceil(self.beta * self.M * 2 * sum(Ls))

    def taps(self, pmi: R18CJTPMI) -> list[int]:
        if self.M == 1:
            return [0]
        if min(self.N_window, self.N3) == 2:
            return [0, 1]
        return [0, pmi.i16[0] + 1]

    # -- gNB side ------------------------------------------------------------

    def _bases(self, pmi: R18CJTPMI, Ls: list[int]) -> list[np.ndarray]:
        half = self.antenna.P // 2
        bases = []
        for i12, L in zip(pmi.i12, Ls):
            ports = list(range(L)) if i12 is None else cb.decode_ports(
                i12, self.antenna.P, L
            )
            B = np.zeros((L, half))
            for i, p in enumerate(ports):
                B[i, p] = 1.0
            bases.append(B)
        return bases

    def precoder(self, pmi: R18CJTPMI) -> np.ndarray:
        self._validate(pmi)
        a = self.antenna
        Ls = self.Ls(pmi)
        S = self._S(Ls)
        bases = self._bases(pmi, Ls)
        Y = dft.freq_basis(self.N3, np.array(self.taps(pmi)))  # (M, N3)
        ramp = self._ramp(pmi)
        offsets = self._offsets(Ls)
        pols = np.array([self._pol_of(Ls, i) for i in range(S)])
        W = np.zeros((1, self.N3, self.n_trp * a.P, pmi.rank), dtype=complex)
        for li in range(pmi.rank):
            p1 = qt.R16_REF_AMP[pmi.k1[li]]
            p2 = qt.R16_DIFF_AMP[pmi.k2[li]]  # (M, S)
            phi = qt.phase_value(pmi.c[li], self.N_PSK)
            x = (p2 * phi * pmi.i17[li]).T * p1[pols][:, None]  # (S, M)
            ct = x @ Y  # (S, N3)
            gamma = np.sum(np.abs(ct) ** 2, axis=0)
            gamma = np.where(gamma == 0, 1.0, gamma)
            for j, (r, off, L, B) in enumerate(
                zip(pmi.resources, offsets, Ls, bases)
            ):
                block = np.concatenate(
                    [B.T @ ct[off : off + L], B.T @ ct[off + L : off + 2 * L]]
                )
                block = block * ramp[j][None, :]
                W[0, :, r * a.P : (r + 1) * a.P, li] = (block / np.sqrt(gamma)).T
        return W / np.sqrt(pmi.rank)

    # -- UE side ---------------------------------------------------------------

    def select(self, H: np.ndarray, rank: int = 1) -> R18CJTPMI:
        if not 1 <= rank <= 4:
            raise ValueError("CJT feType II supports ranks 1-4")
        if not self.ri_restriction[rank - 1]:
            raise ValueError(
                f"rank {rank} prohibited by typeII-PortSelectionRI-Restriction-r18 "
                f"(r{rank - 1}=0)"
            )
        a = self.antenna
        H = np.asarray(H)[-1]
        if H.shape[0] != self.N3:
            raise ValueError(f"channel has {H.shape[0]} frequency units, expected {self.N3}")
        if H.shape[-1] != self.n_trp * a.P:
            raise ValueError(
                f"channel has {H.shape[-1]} ports, expected N_TRP*P = {self.n_trp * a.P}"
            )
        targets = _spatial.aligned_eigen_targets(H, rank)
        half = a.P // 2

        pmi = R18CJTPMI(rank=rank)
        pmi.resources = self._select_resources(targets)
        pmi.i_L = max(
            range(self.N_L), key=lambda i: sum(self.combos[i][r] for r in pmi.resources)
        )
        Ls = self.Ls(pmi)
        alphas = self.alphas(pmi)
        S = self._S(Ls)
        K0 = self.K0_of(Ls)

        i12s, coeff_blocks = [], []
        for r, L, alpha in zip(pmi.resources, Ls, alphas):
            seg = targets[:, r * a.P : (r + 1) * a.P, :]
            if alpha < 1:
                energy = np.sum(np.abs(seg) ** 2, axis=(0, 2))
                port_energy = energy[:half] + energy[half:]
                ports = sorted(np.argsort(port_energy)[::-1][:L].tolist())
                i12s.append(cb.encode_ports(ports, a.P))
            else:
                i12s.append(None)
            B = np.zeros((L, half))
            sel = (
                list(range(L))
                if i12s[-1] is None
                else cb.decode_ports(i12s[-1], a.P, L)
            )
            for i, p in enumerate(sel):
                B[i, p] = 1.0
            coeff_blocks.append(_spatial.ls_coefficients(B, seg, 1.0))
        pmi.i12 = tuple(i12s)
        coeff = np.concatenate(coeff_blocks, axis=2)  # (v, N3, S)

        C_layers = []
        for li in range(rank):
            C = coeff[li].T
            iref = int(np.argmax(np.sum(np.abs(C) ** 2, axis=1)))
            C_layers.append(C * np.exp(-1j * np.angle(C[iref]))[None, :])

        if self.mode == "mode1" and len(pmi.resources) > 1:
            pmi.i19 = self._estimate_offsets(C_layers, Ls)
        C_layers = [self._deramp(C, pmi, Ls) for C in C_layers]

        # common tap pair across layers and resources (window {0..N-1})
        if self.M == 2 and min(self.N_window, self.N3) > 2:
            e_tap = np.zeros(self.N3)
            for C in C_layers:
                e_tap += np.sum(np.abs(np.fft.fft(C, axis=1) / self.N3) ** 2, axis=0)
            window = e_tap[1 : min(self.N_window, self.N3)]
            pmi.i16 = [int(np.argmax(window))]
        taps = self.taps(pmi)
        Y = dft.freq_basis(self.N3, np.array(taps))  # (M, N3)

        pmi.i17 = np.zeros((rank, self.M, S), dtype=bool)
        pmi.k1 = np.ones((rank, 2), dtype=int)
        pmi.k2 = np.zeros((rank, self.M, S), dtype=int)
        pmi.c = np.zeros((rank, self.M, S), dtype=int)

        kept, stars = [], []
        for li in range(rank):
            C = C_layers[li]
            Ct = (C @ Y.conj().T) / self.N3  # (S, M)
            f_star = int(np.argmax(np.sum(np.abs(Ct) ** 2, axis=0)))
            i_star = int(np.argmax(np.abs(Ct[:, f_star])))
            stars.append((i_star, f_star))
            Ct = Ct / Ct[i_star, f_star]
            bm = self._prune_layer(Ct, K0)
            bm[i_star, f_star] = True
            pmi.i17[li] = bm.T
            kept.append(Ct * bm)

        self._enforce_total_budget(pmi, kept, stars, K0)
        self._quantize(pmi, kept, stars, Ls, self.M)
        for li in range(rank):
            i_star, f_star = stars[li]
            pmi.i18.append(f_star * S + i_star)
        return pmi

    # -- validation ------------------------------------------------------------

    def _validate(self, pmi: R18CJTPMI) -> None:
        a = self.antenna
        if not 1 <= pmi.rank <= 4:
            raise ValueError(f"rank {pmi.rank} not in 1..4")
        if not 0 <= pmi.i_L < self.N_L:
            raise ValueError(f"i_L={pmi.i_L} not in [0, {self.N_L})")
        n0 = len(pmi.resources)
        if not 1 <= n0 <= self.n_trp or sorted(set(pmi.resources)) != list(pmi.resources):
            raise ValueError("resources must be distinct, increasing, within N_TRP")
        if self.restricted_cmr and n0 != self.n_trp:
            raise ValueError("restrictedCMR-Selection requires all resources selected")
        Ls = self.Ls(pmi)
        alphas = self.alphas(pmi)
        S = self._S(Ls)
        if len(pmi.i12) != n0:
            raise ValueError("i12 needs one entry per selected resource")
        for i12, L, alpha in zip(pmi.i12, Ls, alphas):
            if alpha == 1:
                if i12 is not None:
                    raise ValueError("i12 must not be reported when alpha = 1")
            elif i12 is None or not 0 <= i12 < comb(a.P // 2, L):
                raise ValueError("per-resource i12 out of range")
        if self.mode == "mode1":
            if len(pmi.i19) != n0 - 1 or any(
                not 0 <= d < self.N3 * self.O3 for d in pmi.i19
            ):
                raise ValueError(f"i19 needs {n0 - 1} offsets in [0, {self.N3 * self.O3})")
        elif pmi.i19:
            raise ValueError("i19 must not be reported in mode2")
        if self.M == 2 and min(self.N_window, self.N3) > 2:
            if len(pmi.i16) != 1 or not 0 <= pmi.i16[0] < self.N_window - 1:
                raise ValueError(f"i16 must be one offset in [0, {self.N_window - 1})")
        elif pmi.i16:
            raise ValueError("i16 must not be reported when M=1 or N=2")
        shape = (pmi.rank, self.M, S)
        for name, arr, lo, hi in (
            ("i17", pmi.i17, 0, 1),
            ("k2", pmi.k2, 0, 7),
            ("c", pmi.c, 0, self.N_PSK - 1),
        ):
            A = np.asarray(arr)
            if A.shape != shape:
                raise ValueError(f"{name} shape {A.shape} != {shape}")
            if A.dtype != bool and (A.min() < lo or A.max() > hi):
                raise ValueError(f"{name} values outside [{lo}, {hi}]")
        if np.asarray(pmi.k1).shape != (pmi.rank, 2):
            raise ValueError("k1 must be (v, 2)")
        for li in range(pmi.rank):
            f_star, i_star = divmod(pmi.i18[li], S)
            if not pmi.i17[li, f_star, i_star]:
                raise ValueError(f"layer {li}: strongest coefficient absent from bitmap")
            if pmi.k1[li, self._pol_of(Ls, i_star)] != 15:
                raise ValueError(f"layer {li}: strongest polarization must have k1=15")
            if pmi.k2[li, f_star, i_star] != 7 or pmi.c[li, f_star, i_star] != 0:
                raise ValueError(f"layer {li}: strongest coefficient must have k2=7, c=0")

    # -- overhead --------------------------------------------------------------

    def overhead_bits(self, pmi: R18CJTPMI) -> dict[str, int]:
        a = self.antenna
        Ls = self.Ls(pmi)
        alphas = self.alphas(pmi)
        S = self._S(Ls)
        v = pmi.rank
        bits: dict[str, int] = {}
        if self.N_L > 1:
            bits["i_L"] = math.ceil(math.log2(self.N_L))
        if not self.restricted_cmr:
            bits["cmr"] = self.n_trp
        i12 = sum(
            math.ceil(math.log2(comb(a.P // 2, L)))
            for L, alpha in zip(Ls, alphas)
            if alpha < 1
        )
        if i12:
            bits["i12"] = i12
        if self.M == 2 and min(self.N_window, self.N3) > 2:
            bits["i16"] = math.ceil(math.log2(self.N_window - 1))
        if self.mode == "mode1" and len(pmi.resources) > 1:
            bits["i19"] = (len(pmi.resources) - 1) * math.ceil(
                math.log2(self.N3 * self.O3)
            )
        K_nz = int(pmi.i17.sum())
        # 5.2.2.2.9: i_{1,7,l} not reported when v <= 2 and all coefficients
        # are nonzero (only reachable at beta = 1... not configurable here,
        # but kept for consistency with the R17 rule)
        if not (v <= 2 and K_nz == S * self.M * v):
            bits["i17"] = v * S * self.M
        bits["i18"] = v * math.ceil(math.log2(S * self.M))
        bits["i23"] = 4 * v
        bits["i24"] = 3 * (K_nz - v)
        bits["i25"] = 4 * (K_nz - v)
        return bits
