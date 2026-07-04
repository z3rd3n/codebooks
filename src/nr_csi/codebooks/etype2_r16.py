"""R16 Enhanced Type II codebook (regular + port-selection), ranks 1-4.

Reconstruction (paper Table tabesII, TS 38.214 Table 5.2.2.2.5-5):

    w^l_t = 1/sqrt(N1*N2*gamma_{t,l}) *
            [ sum_i v_i p1_{l,0} sum_f y_t^(f) p2_{l,i,f} phi_{l,i,f} ;
              sum_i v_i p1_{l,1} sum_f y_t^(f) p2_{l,i+L,f} phi_{l,i+L,f} ]

    gamma_{t,l} = sum_{i=0}^{2L-1} p1_{l,floor(i/L)}^2 |sum_f y_t^(f) p2 phi|^2

with y_t^(f) = exp(j 2 pi t n3_l^(f) / N3) the M_v selected delay taps
(Algorithm 3 codec, two-level indication when N3 > 19) and a bitmap
i_{1,7,l} marking which of the 2L*M_v coefficients are reported, subject to
K_l^NZ <= K0 = ceil(beta*2L*M_1) and sum_l K_l^NZ <= 2*K0.

The port-selection variant replaces the DFT beams by consecutive ports
i_{1,1}*d + i (paper Table tabesps) and drops the N1*N2 factor.  The paper
does not transcribe the PS-specific (L, p_v, beta) table (TS 38.214 Table
5.2.2.2.6-1), so the regular paramCombination-r16 table is reused here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from math import comb

import numpy as np

from ..config import (
    R16_PARAM_COMBOS,
    R16_PS_PARAM_COMBOS,
    AntennaConfig,
    R16ParamCombo,
    m_v,
)
from ..utils import combinatorics as cb
from ..utils import dft
from ..utils import quantization as qt
from . import _spatial
from .base import CodebookScheme


@dataclass
class R16AmplitudeRestriction:
    """R16 codebook subset restriction B = B1 B2 (TS 38.214 5.2.2.2.5,
    ``n1-n2-codebookSubsetRestriction-r16``).

    Same wire format as the R15 ``TypeIIRestriction`` (B1: combinatorial index
    selecting the 4 restricted vector groups out of O1*O2; B2: one 2-bit
    codepoint per beam of each restricted group), but the codepoint bounds the
    *average* coefficient amplitude of Table 5.2.2.2.5-6 instead of the
    per-coefficient wideband amplitude:

        sqrt( sum_f k3 * (p1 * p2_f)^2 / sum_f k3 ) <= gamma

    per layer l, polarization p and beam i, with k3 the reported-coefficient
    bitmap i_{1,7}.  Codepoint 0 prohibits the beam entirely.
    """

    beta1: int
    b2: np.ndarray

    #: Table 5.2.2.2.5-6 codepoint -> max allowed average amplitude gamma.
    GAMMA = np.array([0.0, math.sqrt(1 / 4), math.sqrt(1 / 2), 1.0])

    def restricted_groups(self, O1: int, O2: int) -> list[int]:
        g, _, _ = cb.decode_restriction_groups(self.beta1, O1, O2)
        return g

    def gamma_caps(self, antenna: AntennaConfig, q1: int, q2: int) -> np.ndarray:
        """Max allowed average amplitude per beam n = N1*n2 + n1 of group
        (q1, q2); 1.0 (no cap) for unrestricted groups, 0.0 prohibits."""
        n_beams = antenna.N1 * antenna.N2
        g = self.restricted_groups(antenna.O1, antenna.O2)
        group = antenna.O1 * q2 + q1
        if group not in g:
            return np.ones(n_beams)
        return self.GAMMA[np.asarray(self.b2[g.index(group)], dtype=int)]


@dataclass
class R16Type2PMI:
    rank: int
    # spatial part: regular (q1,q2,i12) or port-selection (i11_ps)
    q1: int | None = None
    q2: int | None = None
    i12: int | None = None
    i11_ps: int | None = None
    # frequency part
    i15: int | None = None  # M_initial indicator (N3 > 19 only)
    i16: list[int] = field(default_factory=list)  # tap combination per layer
    # coefficient part, all shaped per layer
    i17: np.ndarray | None = None  # bitmap (v, Mv, 2L), bool
    i18: list[int] = field(default_factory=list)  # strongest-coefficient indicator
    k1: np.ndarray | None = None  # (v, 2) reference amplitude indices, 1..15
    k2: np.ndarray | None = None  # (v, Mv, 2L) differential amplitude indices, 0..7
    c: np.ndarray | None = None  # (v, Mv, 2L) phase indices, 0..15


def select_taps(
    tap_energy: np.ndarray, Mv: int, N3: int, m_initial: int | None = None
) -> list[int]:
    """Top-Mv taps (remapped energies, index 0 strongest), honoring the
    2*Mv window constraint when N3 > 19.  ``m_initial`` (in {-2Mv+1..0})
    restricts the search to one window so all layers share i_{1,5}."""
    if Mv == 1:
        return [0]
    if N3 <= 19:
        top = np.argsort(tap_energy)[::-1][:Mv]
        return sorted(int(x) for x in top)
    candidates = range(0, -2 * Mv, -1) if m_initial is None else [m_initial]
    best = (-1.0, None)
    for m_init in candidates:
        window = [(m_init + j) % N3 for j in range(2 * Mv)]
        others = [w for w in window if w != 0]
        others.sort(key=lambda n: -tap_energy[n])
        sel = [0] + others[: Mv - 1]
        e = float(sum(tap_energy[n] for n in sel))
        if e > best[0]:
            best = (e, sorted(sel))
    return best[1]


def encode_i18(bitmap_l: np.ndarray, i_star: int, rank: int) -> int:
    """Strongest-coefficient indicator (paper's dual-mode definition).

    rank 1: position of (i*, f=0) among the nonzero bitmap bits of tap 0;
    rank > 1: the beam index i* directly.
    """
    if rank == 1:
        return int(np.cumsum(bitmap_l[0])[i_star]) - 1
    return i_star


def decode_i18(i18: int, bitmap_l: np.ndarray, rank: int) -> int:
    if rank == 1:
        cs = np.cumsum(bitmap_l[0])
        return int(np.argmax(cs == i18 + 1))
    return i18


class R16Type2Codebook(CodebookScheme):
    N_PSK = 16

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        param_combination: int = 4,
        R: int = 1,
        port_selection: bool = False,
        d: int = 1,
        combo: R16ParamCombo | None = None,
        ri_restriction: np.ndarray | None = None,
        restriction: R16AmplitudeRestriction | None = None,
    ) -> None:
        self.antenna = antenna
        self.N3 = N3
        if R not in (1, 2):
            raise ValueError("R must be 1 or 2")
        self.R = R
        # typeII-RI-Restriction-r16 / typeII-PortSelectionRI-Restriction-r16:
        # 4-bit bitmap [r0..r3]; r_i = 0 prohibits rank i+1.
        if ri_restriction is None:
            ri_restriction = np.ones(4, dtype=bool)
        self.ri_restriction = np.asarray(ri_restriction, dtype=bool)
        if self.ri_restriction.shape != (4,):
            raise ValueError("typeII-RI-Restriction-r16 must have 4 bits [r0..r3]")
        # ``combo`` overrides the standardized paramCombination-r16 table for
        # generalized (L, p_v, beta) sweeps -- e.g. holding (L, M_v) fixed
        # while varying beta past the eight spec rows (used by the Qin Fig. 5
        # reproduction).  Default behaviour is unchanged: look up the spec row,
        # enforcing the configuration bars of 5.2.2.2.5.
        if combo is not None:
            self.combo = combo
        elif port_selection:
            # The port-selection variant uses Table 5.2.2.2.6-1, which omits the
            # L=6 rows (paramCombination 7, 8) of the regular table.
            if param_combination not in R16_PS_PARAM_COMBOS:
                raise ValueError(
                    f"paramCombination-r16={param_combination} is not supported for "
                    f"the eType II port-selection codebook; Table 5.2.2.2.6-1 lists "
                    f"1..6 (the L=6 combinations 7, 8 are not allowed)"
                )
            self.combo = R16_PS_PARAM_COMBOS[param_combination]
        else:
            # Configuration bars of 5.2.2.2.5: "The UE is not expected to be
            # configured with paramCombination-r16 equal to ...".
            if antenna.P == 4 and param_combination >= 3:
                raise ValueError(
                    f"paramCombination-r16={param_combination} is not supported at "
                    f"P_CSI-RS=4 (5.2.2.2.5 bars combinations 3-8)"
                )
            if param_combination in (7, 8):
                if antenna.P < 32:
                    raise ValueError(
                        f"paramCombination-r16={param_combination} requires "
                        f"P_CSI-RS >= 32 (got {antenna.P})"
                    )
                if R == 2:
                    raise ValueError(
                        f"paramCombination-r16={param_combination} is not supported "
                        f"with R=2"
                    )
                if bool(self.ri_restriction[2] or self.ri_restriction[3]):
                    raise ValueError(
                        f"paramCombination-r16={param_combination} requires ranks 3-4 "
                        f"disallowed (typeII-RI-Restriction-r16 with r_i=0 for i>1)"
                    )
            self.combo = R16_PARAM_COMBOS[param_combination]
        self.L = self.combo.L
        self.port_selection = port_selection
        # TS 38.214 5.2.2.2.6: d in {1,2,3,4} and d <= L for port selection.
        if port_selection and not (1 <= d <= 4 and d <= self.L):
            raise ValueError(
                "portSelectionSamplingSize-r16 d must satisfy d in {1,2,3,4} and d <= L"
            )
        if not port_selection and self.L > antenna.n_ports_per_pol:
            raise ValueError(
                f"L={self.L} beams cannot be drawn from an orthogonal group of "
                f"N1*N2={antenna.n_ports_per_pol} beams"
            )
        self.d = d
        if restriction is not None:
            if port_selection:
                raise ValueError("subset restriction applies to the regular codebook only")
            if not 0 <= restriction.beta1 < comb(antenna.O1 * antenna.O2, 4):
                raise ValueError("beta1 out of range")
            if np.asarray(restriction.b2).shape != (4, antenna.N1 * antenna.N2):
                raise ValueError(f"B2 must be (4, {antenna.N1 * antenna.N2}) codepoints")
        self.restriction = restriction
        if N3 < 1:
            raise ValueError("N3 must be positive")
        for v in (1, 2, 3, 4):  # fail at construction, not first use
            if v >= 3 and self.combo.p_v34 is None:
                continue
            self.Mv(v)
        self.name = "R16 eType II PS" if port_selection else "R16 eType II"

    # -- derived parameters ------------------------------------------------

    def Mv(self, rank: int) -> int:
        val = m_v(self.combo.p_v(rank), self.N3, self.R)
        if val > self.N3:
            raise ValueError("M_v exceeds N3")
        return val

    @property
    def K0(self) -> int:
        M1 = m_v(self.combo.p_v(1), self.N3, self.R)
        return math.ceil(self.combo.beta * 2 * self.L * M1)

    def _scale(self) -> float:
        return 1.0 if self.port_selection else float(self.antenna.n_ports_per_pol)

    def _basis(self, pmi: R16Type2PMI) -> np.ndarray:
        if self.port_selection:
            return _spatial.basis_ps(self.antenna, pmi.i11_ps, self.L, self.d)
        return _spatial.basis_regular(self.antenna, pmi.q1, pmi.q2, pmi.i12, self.L)

    # -- gNB side ------------------------------------------------------------

    def _layer_coefficients(self, pmi: R16Type2PMI, li: int) -> np.ndarray:
        """Reported (pruned, quantized) coefficients x_{i,f}, shape (2L, Mv)."""
        p1 = qt.R16_REF_AMP[pmi.k1[li]]  # (2,)
        p2 = qt.R16_DIFF_AMP[pmi.k2[li]]  # (Mv, 2L)
        phi = qt.phase_value(pmi.c[li], self.N_PSK)  # (Mv, 2L)
        x = (p2 * phi * pmi.i17[li]).T  # (2L, Mv)
        pol = np.repeat([0, 1], self.L)
        return x * p1[pol][:, None]

    def precoder(self, pmi: R16Type2PMI) -> np.ndarray:
        from .validate import validate_r16

        validate_r16(self, pmi)
        a = self.antenna
        Mv = self.Mv(pmi.rank)
        B = self._basis(pmi)  # (L, P/2)
        W = np.zeros((1, self.N3, a.P, pmi.rank), dtype=complex)
        for li in range(pmi.rank):
            taps = cb.decode_taps(pmi.i16[li], self.N3, Mv, pmi.i15)
            Y = dft.freq_basis(self.N3, np.array(taps)).T  # (N3, Mv)
            x = self._layer_coefficients(pmi, li)  # (2L, Mv), includes p1
            ct = x @ Y.T  # (2L, N3)
            gamma = np.sum(np.abs(ct) ** 2, axis=0)  # (N3,)
            gamma = np.where(gamma == 0, 1.0, gamma)
            w = np.concatenate([B.T @ ct[: self.L], B.T @ ct[self.L :]], axis=0)  # (P, N3)
            W[0, :, :, li] = (w / np.sqrt(self._scale() * gamma)).T
        return W / np.sqrt(pmi.rank)

    # -- UE side ---------------------------------------------------------------

    def select(self, H: np.ndarray, rank: int = 1) -> R16Type2PMI:
        if not 1 <= rank <= 4:
            raise ValueError("R16 eType II supports ranks 1-4")
        if not self.ri_restriction[rank - 1]:
            name = (
                "typeII-PortSelectionRI-Restriction-r16"
                if self.port_selection
                else "typeII-RI-Restriction-r16"
            )
            raise ValueError(f"rank {rank} prohibited by {name} (r{rank - 1}=0)")
        H = np.asarray(H)[-1]
        if H.shape[0] != self.N3:
            raise ValueError(f"channel has {H.shape[0]} frequency units, expected {self.N3}")
        targets = _spatial.aligned_eigen_targets(H, rank)
        Mv = self.Mv(rank)

        pmi = R16Type2PMI(rank=rank)
        if self.port_selection:
            pmi.i11_ps = _spatial.select_ps_initial(self.antenna, targets, self.L, self.d)
        else:
            allowed = None
            if self.restriction is not None:
                allowed = lambda q1, q2: self.restriction.gamma_caps(
                    self.antenna, q1, q2
                ) > 0
            pmi.q1, pmi.q2, pmi.i12 = _spatial.select_group_and_beams(
                self.antenna, targets, self.L, allowed=allowed
            )
        gamma_caps = self._selected_gamma_caps(pmi)
        B = self._basis(pmi)
        coeff = _spatial.ls_coefficients(B, targets, self._scale())  # (v, N3, 2L)

        L2 = 2 * self.L
        pmi.i17 = np.zeros((rank, Mv, L2), dtype=bool)
        pmi.k1 = np.ones((rank, 2), dtype=int)
        pmi.k2 = np.zeros((rank, Mv, L2), dtype=int)
        pmi.c = np.zeros((rank, Mv, L2), dtype=int)

        kept: list[np.ndarray] = []  # per-layer pruned coefficient matrices (2L, Mv)
        stars: list[int] = []
        m_init_common: int | None = None  # i15 is reported once for all layers
        for li in range(rank):
            C = coeff[li].T  # (2L, N3)
            # Resolve the per-subband phase ambiguity of the eigen targets by
            # rotating each subband so the strongest beam's coefficient is
            # real-positive: sequential alignment would leave a residual phase
            # chirp across frequency that smears the delay taps.
            iref = int(np.argmax(np.sum(np.abs(C) ** 2, axis=1)))
            C = C * np.exp(-1j * np.angle(C[iref]))[None, :]
            # delay-domain coefficients c_hat with c(t) = sum_n c_hat[n] e^{+j2pi t n/N3},
            # i.e. c_hat = fft(c)/N3 (numpy's ifft would flip the tap indices)
            Ctap = np.fft.fft(C, axis=1) / self.N3
            tap_energy = np.sum(np.abs(Ctap) ** 2, axis=0)
            n_star = int(np.argmax(tap_energy))
            Ctap = np.roll(Ctap, -n_star, axis=1)  # strongest tap remapped to 0
            taps = self._select_taps(np.roll(tap_energy, -n_star), Mv, m_init_common)
            i16, i15 = cb.encode_taps(taps, self.N3, Mv, m_initial=m_init_common)
            pmi.i16.append(i16)
            if i15 is not None and m_init_common is None:
                m_init_common = i15 if i15 == 0 else i15 - 2 * Mv
            Ct = Ctap[:, taps]  # (2L, Mv)
            # the strongest coefficient has fixed p1 = p2 = 1, so it must sit
            # on an amplitude-unrestricted beam to keep the restriction honest
            if gamma_caps is not None:
                eligible = np.where(gamma_caps >= 1.0, np.abs(Ct[:, 0]), -np.inf)
                i_star = int(np.argmax(eligible))
                if not np.isfinite(eligible[i_star]):
                    i_star = int(np.argmax(np.abs(Ct[:, 0])))
            else:
                i_star = int(np.argmax(np.abs(Ct[:, 0])))
            stars.append(i_star)
            Ct = Ct / Ct[i_star, 0]
            # per-layer bitmap: K0 strongest coefficients (order-independent
            # flat indexing -- Ct may be F-ordered, where ravel() would copy)
            mags = np.abs(Ct)
            order = np.argsort(mags.reshape(-1))[::-1]
            keep = order[: min(self.K0, mags.size)]
            bm = np.zeros(mags.shape, dtype=bool)
            bm[np.unravel_index(keep, mags.shape)] = True
            # the 3-bit differential amplitude table has no zero entry (min
            # 1/(8*sqrt(2))), so coefficients below half that floor are better
            # represented as "not reported" than quantized upward
            bm &= mags > qt.R16_DIFF_AMP[0] / 2
            bm[i_star, 0] = True
            pmi.i17[li] = bm.T
            kept.append(Ct * bm)
        if m_init_common is not None:
            pmi.i15 = m_init_common if m_init_common == 0 else m_init_common + 2 * Mv

        self._enforce_total_budget(pmi, kept, stars)

        for li in range(rank):
            Ct = kept[li]
            bm = pmi.i17[li].T  # (2L, Mv)
            i_star = stars[li]
            p_star = i_star // self.L
            mags = np.abs(Ct)
            for pol in (0, 1):
                rows = slice(0, self.L) if pol == 0 else slice(self.L, L2)
                if pol == p_star:
                    pmi.k1[li, pol] = 15
                    continue
                ref = mags[rows][bm[rows]].max() if bm[rows].any() else 0.0
                pmi.k1[li, pol] = max(int(qt.quantize_amplitude(ref, qt.R16_REF_AMP)), 1)
            p1 = qt.R16_REF_AMP[pmi.k1[li]]
            for i in range(L2):
                for f in range(Mv):
                    if not bm[i, f]:
                        continue
                    rel = mags[i, f] / p1[i // self.L]
                    pmi.k2[li, f, i] = int(qt.quantize_amplitude(min(rel, 1.0), qt.R16_DIFF_AMP))
                    pmi.c[li, f, i] = int(qt.quantize_phase(np.angle(Ct[i, f]), self.N_PSK))
            # strongest coefficient conventions (not reported, fixed values)
            pmi.k2[li, 0, i_star] = 7
            pmi.c[li, 0, i_star] = 0
        if gamma_caps is not None:
            self._enforce_amplitude_restriction(pmi, gamma_caps, stars)
        # i18 is encoded last: the rank-1 form indexes the *nonzero* bitmap
        # bits, so it must reflect any entries dropped by the restriction.
        for li in range(rank):
            pmi.i18.append(encode_i18(pmi.i17[li], stars[li], rank))
        return pmi

    def _selected_gamma_caps(self, pmi: R16Type2PMI) -> np.ndarray | None:
        """Per-coefficient max average amplitude (2L,) for the selected beams,
        or None when unrestricted (Table 5.2.2.2.5-6)."""
        if self.restriction is None or self.port_selection:
            return None
        n1, n2 = cb.decode_beam_combination(pmi.i12, self.antenna.N1, self.antenna.N2, self.L)
        caps_group = self.restriction.gamma_caps(self.antenna, pmi.q1, pmi.q2)
        caps = np.array([caps_group[self.antenna.N1 * b + a] for a, b in zip(n1, n2)])
        return np.concatenate([caps, caps])  # both polarizations

    def _enforce_amplitude_restriction(
        self, pmi: R16Type2PMI, caps: np.ndarray, stars: list[int]
    ) -> None:
        """Reduce differential amplitudes until each restricted beam meets
        sqrt(sum_f k3 (p1 p2)^2 / sum_f k3) <= gamma (5.2.2.2.5).

        The greedy k2 decrement always terminates: allowed beams have
        gamma >= 1/2 while p1 * min(p2) <= 1/(8 sqrt(2)).  The strongest
        coefficient (fixed k2 = 7, not reported) is never touched; ``select``
        places it on an unrestricted beam whenever one exists.
        """
        for li in range(pmi.rank):
            p1 = qt.R16_REF_AMP[pmi.k1[li]]  # (2,)
            for i in range(2 * self.L):
                gamma = caps[i]
                if gamma >= 1.0:
                    continue
                star_f = 0 if i == stars[li] else None
                while True:
                    k3 = pmi.i17[li][:, i]  # (Mv,)
                    if not k3.any():
                        break
                    amps = p1[i // self.L] * qt.R16_DIFF_AMP[pmi.k2[li][:, i]]
                    avg = math.sqrt(float(np.mean(amps[k3] ** 2)))
                    if avg <= gamma + 1e-12:
                        break
                    k2_col = np.where(k3, pmi.k2[li][:, i], -1)
                    if star_f is not None:
                        k2_col[star_f] = -1  # never reduce the strongest coefficient
                    f = int(np.argmax(k2_col))
                    if k2_col[f] > 0:
                        pmi.k2[li, f, i] -= 1
                    elif k2_col[f] == 0:  # already at the floor: drop the entry
                        pmi.i17[li][f, i] = False
                        pmi.k2[li, f, i] = 0
                        pmi.c[li, f, i] = 0
                    else:  # only the strongest coefficient remains
                        break

    def _select_taps(
        self, tap_energy: np.ndarray, Mv: int, m_initial: int | None = None
    ) -> list[int]:
        return select_taps(tap_energy, Mv, self.N3, m_initial)

    def _enforce_total_budget(
        self, pmi: R16Type2PMI, kept: list[np.ndarray], stars: list[int]
    ) -> None:
        """Drop globally weakest coefficients until sum_l K_l^NZ <= 2*K0."""
        budget = 2 * self.K0
        total = int(sum(b.sum() for b in pmi.i17))
        if total <= budget:
            return
        entries = []  # (magnitude, layer, i, f)
        for li in range(pmi.rank):
            bm = pmi.i17[li].T
            for i in range(bm.shape[0]):
                for f in range(bm.shape[1]):
                    if bm[i, f] and not (f == 0 and i == stars[li]):
                        entries.append((abs(kept[li][i, f]), li, i, f))
        entries.sort()
        for mag, li, i, f in entries:
            if total <= budget:
                break
            pmi.i17[li][f, i] = False
            kept[li][i, f] = 0.0
            total -= 1

    # -- overhead ------------------------------------------------------------

    def overhead_bits(self, pmi: R16Type2PMI) -> dict[str, int]:
        a = self.antenna
        L, Mv = self.L, self.Mv(pmi.rank)
        v = pmi.rank
        bits: dict[str, int] = {}
        if self.port_selection:
            bits["i11"] = math.ceil(math.log2(math.ceil(a.P / (2 * self.d))))
        else:
            bits["i11"] = math.ceil(math.log2(a.O1 * a.O2))
            bits["i12"] = math.ceil(math.log2(comb(a.N1 * a.N2, L)))
        if self.N3 > 19:
            bits["i15"] = math.ceil(math.log2(2 * Mv))
            bits["i16"] = v * math.ceil(math.log2(comb(2 * Mv - 1, Mv - 1)))
        else:
            bits["i16"] = v * math.ceil(math.log2(comb(self.N3 - 1, Mv - 1)))
        bits["i17"] = v * 2 * L * Mv
        K_nz = int(pmi.i17.sum())
        # TS 38.212 Table 6.3.2.1.2-1A: rank 1 spends ceil(log2(K^NZ)) bits on
        # the strongest-coefficient indicator; ranks 2-4 spend ceil(log2(2L))
        # per layer.
        if v == 1:
            bits["i18"] = math.ceil(math.log2(K_nz)) if K_nz > 1 else 0
        else:
            bits["i18"] = v * math.ceil(math.log2(2 * L))
        bits["i23"] = 4 * v
        bits["i24"] = 3 * (K_nz - v)
        bits["i25"] = 4 * (K_nz - v)
        return bits
