"""R15 Type II codebook, regular and port-selection variants, ranks 1-2.

Regular reconstruction (paper Table tabtypeii, TS 38.214 Table 5.2.2.2.4-1):

    w^l = 1/sqrt(N1*N2 * sum_i (p1_i p2_i)^2)
          [ sum_{i<L} v_i p1_i p2_i phi_i ; sum_{i<L} v_i p1_{i+L} p2_{i+L} phi_{i+L} ]

with L beams selected inside one orthogonal group (q1, q2) via the
combinatorial index i_{1,2} (Algorithm 1).  The port-selection variant
(paper Appendix r15ps) replaces v_i by the standard basis vector of port
i_{1,1}*d + i and drops the N1*N2 factor.

Reporting-reduction rules implemented exactly as in the paper:
* strongest coefficient i_{1,3,l}: k1=7, k2=1, c=0, none reported;
* subbandAmplitude=false: all p2=1, i_{2,2,l} absent;
* subbandAmplitude=true: subband amplitudes only for the min(Ml, K2)-1
  strongest other coefficients (K2 = 4 for L in {2,3}, 6 for L=4); the
  Ml - min(Ml, K2) weakest non-zero coefficients fall back to QPSK phases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import comb

import numpy as np

from ..config import AntennaConfig
from ..utils import combinatorics as cb
from ..utils import dft
from ..utils import quantization as qt
from . import _spatial
from .base import CodebookScheme


@dataclass
class TypeIIRestriction:
    """Type II codebook subset restriction B = B1 B2 (paper eqs. a58/a59).

    beta1: combinatorial index (Algorithm 2, 11 bits) selecting the 4
        restricted vector groups out of O1*O2.
    b2: (4, N1*N2) entries in {0,1,2,3}, the 2-bit max-amplitude codepoints
        of Table tabmaxap for each beam (x1, x2) of restricted group k,
        indexed x = N1*x2 + x1.  0 prohibits the beam entirely.
    """

    beta1: int
    b2: np.ndarray

    #: Table tabmaxap codepoint -> max allowed p^(1), as the largest
    #: admissible R15 wideband-amplitude index k1 (R15_WB_AMP[k1] <= max).
    MAX_K1 = {0: -1, 1: 5, 2: 6, 3: 7}  # 0, sqrt(1/4), sqrt(1/2), 1

    def restricted_groups(self, O1: int, O2: int) -> list[int]:
        g, _, _ = cb.decode_restriction_groups(self.beta1, O1, O2)
        return g

    def k1_caps(self, antenna: AntennaConfig, q1: int, q2: int) -> np.ndarray:
        """Max allowed k1 per beam n = N1*n2 + n1 of group (q1, q2);
        7 (no cap) for unrestricted groups, -1 prohibits the beam."""
        n_beams = antenna.N1 * antenna.N2
        g = self.restricted_groups(antenna.O1, antenna.O2)
        group = antenna.O1 * q2 + q1
        if group not in g:
            return np.full(n_beams, 7)
        row = self.b2[g.index(group)]
        return np.array([self.MAX_K1[int(v)] for v in row])


@dataclass
class R15Type2PMI:
    rank: int
    # regular variant: orthogonal group + beam combination
    q1: int | None = None
    q2: int | None = None
    i12: int | None = None
    # port-selection variant: initial port index
    i11_ps: int | None = None
    # per-layer fields
    i13: list[int] | None = None  # strongest coefficient index, in 0..2L-1
    k1: np.ndarray | None = None  # (v, 2L) wideband amplitude indices
    k2: np.ndarray | None = None  # (v, N3, 2L) subband amplitude indices
    c: np.ndarray | None = None  # (v, N3, 2L) phase indices


def k2_cap(L: int) -> int:
    """K^(2): max number of subband amplitudes reported per layer."""
    return 4 if L in (2, 3) else 6


class R15Type2Codebook(CodebookScheme):
    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int = 1,
        L: int = 4,
        n_psk: int = 8,
        subband_amplitude: bool = False,
        port_selection: bool = False,
        d: int = 1,
        restriction: TypeIIRestriction | None = None,
        ri_restriction: np.ndarray | None = None,
    ) -> None:
        if L not in (2, 3, 4):
            raise ValueError("numberOfBeams L must be in {2,3,4}")
        if L > antenna.n_ports_per_pol:
            raise ValueError(
                f"L={L} beams cannot be drawn from an orthogonal group of "
                f"N1*N2={antenna.n_ports_per_pol} beams"
            )
        if n_psk not in (4, 8):
            raise ValueError("phaseAlphabetSize must be 4 or 8")
        self.antenna = antenna
        self.N3 = N3
        self.L = L
        self.n_psk = n_psk
        self.sa = subband_amplitude
        self.port_selection = port_selection
        if port_selection:
            if not (1 <= d <= 4 and d <= min(antenna.P // 2, L)):
                raise ValueError(
                    "portSelectionSamplingSize d must satisfy d<=min(P/2,L), d in 1..4"
                )
            if restriction is not None:
                raise ValueError("subset restriction applies to the regular codebook only")
        if restriction is not None:
            if not 0 <= restriction.beta1 < comb(antenna.O1 * antenna.O2, 4):
                raise ValueError("beta1 out of range")
            if restriction.b2.shape != (4, antenna.N1 * antenna.N2):
                raise ValueError(f"B2 must be (4, {antenna.N1 * antenna.N2}) codepoints")
        self.restriction = restriction
        self.d = d
        # typeII-RI-Restriction / typeII-PortSelectionRI-Restriction: 2-bit
        # bitmap [r0, r1]; r_i = 0 prohibits rank i+1 (TS 38.214 5.2.2.2.3/.4).
        if ri_restriction is None:
            ri_restriction = np.ones(2, dtype=bool)
        self.ri_restriction = np.asarray(ri_restriction, dtype=bool)
        if self.ri_restriction.shape != (2,):
            raise ValueError(
                "typeII-RI-Restriction r = [r0, r1] must have 2 bits"
            )
        self.name = "R15 Type II PS" if port_selection else "R15 Type II"

    def _ri_restriction_name(self) -> str:
        return (
            "typeII-PortSelectionRI-Restriction"
            if self.port_selection
            else "typeII-RI-Restriction"
        )

    # ------------------------------------------------------------------
    # spatial bases
    # ------------------------------------------------------------------

    def _basis(self, pmi: R15Type2PMI) -> np.ndarray:
        """Selected per-polarization spatial bases, shape (L, P/2)."""
        half = self.antenna.P // 2
        if self.port_selection:
            B = np.zeros((self.L, half))
            for i in range(self.L):
                B[i, (pmi.i11_ps * self.d + i) % half] = 1.0
            return B
        n1, n2 = cb.decode_beam_combination(pmi.i12, self.antenna.N1, self.antenna.N2, self.L)
        m1 = [self.antenna.O1 * a + pmi.q1 for a in n1]
        m2 = [self.antenna.O2 * b + pmi.q2 for b in n2]
        return np.stack([dft.spatial_beam(self.antenna, a, b) for a, b in zip(m1, m2)])

    def _beta_factor(self) -> float:
        """N1*N2 for the regular variant (paper Table tabtypeii), 1 for PS."""
        return 1.0 if self.port_selection else float(self.antenna.n_ports_per_pol)

    # ------------------------------------------------------------------
    # coefficient partition (shared by reconstruction, selection, overhead)
    # ------------------------------------------------------------------

    def _partition(self, k1_row: np.ndarray, i_star: int):
        """Split coefficient indices into (strong w/o strongest, weak, zero).

        Ranking is by wideband amplitude (descending), strongest coefficient
        always first; ties broken by coefficient index.
        """
        order = sorted(range(2 * self.L), key=lambda i: (i != i_star, -int(k1_row[i]), i))
        nonzero = [i for i in order if k1_row[i] > 0]
        Ml = len(nonzero)
        n_strong = min(Ml, k2_cap(self.L))
        strong = [i for i in nonzero[:n_strong] if i != i_star]
        weak = nonzero[n_strong:]
        zero = [i for i in range(2 * self.L) if k1_row[i] == 0]
        return strong, weak, zero

    def _phase_alphabets(self, k1_row: np.ndarray, i_star: int) -> np.ndarray:
        """Per-coefficient PSK size used for i_{2,1,l} (paper SA=true rules)."""
        sizes = np.full(2 * self.L, self.n_psk)
        if self.sa:
            _, weak, zero = self._partition(k1_row, i_star)
            sizes[weak] = 4
            sizes[zero] = 4  # value irrelevant (c fixed to 0, amplitude 0)
        return sizes

    # ------------------------------------------------------------------
    # gNB side
    # ------------------------------------------------------------------

    def precoder(self, pmi: R15Type2PMI) -> np.ndarray:
        from .validate import validate_r15

        validate_r15(self, pmi)
        a = self.antenna
        B = self._basis(pmi)  # (L, P/2)
        W = np.zeros((1, self.N3, a.P, pmi.rank), dtype=complex)
        for li in range(pmi.rank):
            p1 = qt.R15_WB_AMP[pmi.k1[li]]  # (2L,)
            sizes = self._phase_alphabets(pmi.k1[li], pmi.i13[li])
            for t in range(self.N3):
                p2 = qt.R15_SB_AMP[pmi.k2[li, t]] if self.sa else np.ones(2 * self.L)
                phi = np.exp(2j * np.pi * pmi.c[li, t] / sizes)
                amp = p1 * p2
                coeff = amp * phi
                beta = self._beta_factor() * np.sum(amp**2)
                w = np.concatenate([B.T @ coeff[: self.L], B.T @ coeff[self.L :]])
                W[0, t, :, li] = w / np.sqrt(beta)
        return W / np.sqrt(pmi.rank)

    # ------------------------------------------------------------------
    # UE side
    # ------------------------------------------------------------------

    def select(self, H: np.ndarray, rank: int = 1) -> R15Type2PMI:
        if rank not in (1, 2):
            raise ValueError("R15 Type II supports ranks 1-2")
        if not self.ri_restriction[rank - 1]:
            raise ValueError(
                f"rank {rank} prohibited by {self._ri_restriction_name()} (r{rank - 1}=0)"
            )
        from ..baselines.ideal import eigen_precoder

        H = np.asarray(H)[-1]  # (N3, Nr, P)
        if H.shape[0] != self.N3:
            raise ValueError(f"channel has {H.shape[0]} frequency units, expected {self.N3}")
        targets = eigen_precoder(H, rank=rank) * np.sqrt(rank)  # (N3, P, v) unit columns

        pmi = R15Type2PMI(rank=rank)
        if self.port_selection:
            pmi.i11_ps = _spatial.select_ps_initial(self.antenna, targets, self.L, self.d)
        else:
            allowed = None
            if self.restriction is not None:
                allowed = lambda q1, q2: self.restriction.k1_caps(self.antenna, q1, q2) >= 0
            pmi.q1, pmi.q2, pmi.i12 = _spatial.select_group_and_beams(
                self.antenna, targets, self.L, allowed=allowed
            )
        B = self._basis(pmi)  # (L, P/2)
        coeff = _spatial.ls_coefficients(B, targets, self._beta_factor())
        self._quantize_into(pmi, coeff, self._selected_caps(pmi))
        return pmi

    def _selected_caps(self, pmi: R15Type2PMI) -> np.ndarray | None:
        """Per-coefficient max k1 (2L,) for the selected beams, or None."""
        if self.restriction is None or self.port_selection:
            return None
        n1, n2 = cb.decode_beam_combination(pmi.i12, self.antenna.N1, self.antenna.N2, self.L)
        caps_group = self.restriction.k1_caps(self.antenna, pmi.q1, pmi.q2)
        caps = np.array([caps_group[self.antenna.N1 * b + a] for a, b in zip(n1, n2)])
        return np.concatenate([caps, caps])  # both polarizations

    def _quantize_into(
        self, pmi: R15Type2PMI, coeff: np.ndarray, caps: np.ndarray | None = None
    ) -> None:
        """Fill i13/k1/k2/c from LS coefficients (v, N3, 2L); ``caps``
        optionally bounds each k1 (subset restriction, Table tabmaxap)."""
        rank = pmi.rank
        L2 = 2 * self.L
        pmi.i13 = []
        pmi.k1 = np.zeros((rank, L2), dtype=int)
        pmi.k2 = np.ones((rank, self.N3, L2), dtype=int)
        pmi.c = np.zeros((rank, self.N3, L2), dtype=int)
        for li in range(rank):
            cl = coeff[li]  # (N3, 2L)
            wb_amp = np.sqrt(np.mean(np.abs(cl) ** 2, axis=0))  # (2L,)
            if caps is not None:
                # the strongest coefficient is reported as k1 = 7, so it must
                # come from an uncapped beam to keep the restriction honest
                eligible = np.where(caps >= 7, wb_amp, -np.inf)
                i_star = int(np.argmax(eligible))
                if not np.isfinite(eligible[i_star]):
                    i_star = int(np.argmax(wb_amp))
            else:
                i_star = int(np.argmax(wb_amp))
            pmi.i13.append(i_star)
            # rotate each subband so the strongest coefficient has zero phase
            rot = np.exp(-1j * np.angle(cl[:, i_star]))[:, None]
            cl = cl * rot
            ref = wb_amp[i_star]
            k1_row = qt.quantize_amplitude(np.minimum(wb_amp / ref, 1.0), qt.R15_WB_AMP)
            if caps is not None:
                k1_row = np.minimum(k1_row, np.maximum(caps, 0))
            k1_row[i_star] = 7
            pmi.k1[li] = k1_row
            strong, weak, zero = self._partition(k1_row, i_star)
            sizes = self._phase_alphabets(k1_row, i_star)
            for t in range(self.N3):
                for i in range(L2):
                    if i == i_star or i in zero:
                        continue  # c=0 / k2=1 already set
                    pmi.c[li, t, i] = int(qt.quantize_phase(np.angle(cl[t, i]), int(sizes[i])))
                if self.sa:
                    p1_vals = qt.R15_WB_AMP[k1_row]
                    for i in strong:
                        rel = np.abs(cl[t, i]) / (ref * p1_vals[i]) if p1_vals[i] > 0 else 1.0
                        pmi.k2[li, t, i] = int(
                            qt.quantize_amplitude(min(rel, 1.0), qt.R15_SB_AMP)
                        )

    # ------------------------------------------------------------------
    # overhead
    # ------------------------------------------------------------------

    def overhead_bits(self, pmi: R15Type2PMI) -> dict[str, int]:
        a = self.antenna
        L = self.L
        bits: dict[str, int] = {}
        if self.port_selection:
            bits["i11"] = math.ceil(math.log2(math.ceil(a.P / (2 * self.d))))
        else:
            bits["i11"] = math.ceil(math.log2(a.O1 * a.O2))
            bits["i12"] = math.ceil(math.log2(comb(a.N1 * a.N2, L)))
        bits["i13"] = pmi.rank * math.ceil(math.log2(2 * L))
        bits["i14"] = pmi.rank * 3 * (2 * L - 1)
        i21 = i22 = 0
        for li in range(pmi.rank):
            strong, weak, _ = self._partition(pmi.k1[li], pmi.i13[li])
            if self.sa:
                per_sb = len(strong) * round(math.log2(self.n_psk)) + 2 * len(weak)
                i22 += self.N3 * len(strong)
            else:
                Ml = int(np.count_nonzero(pmi.k1[li]))
                per_sb = (Ml - 1) * round(math.log2(self.n_psk))
            i21 += self.N3 * per_sb
        bits["i21"] = i21
        if self.sa:
            bits["i22"] = i22
        return bits
