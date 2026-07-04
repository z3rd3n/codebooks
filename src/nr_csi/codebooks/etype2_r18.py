"""R18 Enhanced Type II codebook for predicted PMI (Doppler domain), ranks 1-4.

Reconstruction (paper Table tab1, TS 38.214 Table 5.2.2.2.10-2):

    w^l_{t,iota} = 1/sqrt(N1*N2*gamma_{t,iota,l}) *
        [ sum_i v_i p1_{l,0} sum_f y_t^(f) sum_tau z_iota^(tau) p2 phi ; ... ]

The codebook adds a temporal DFT basis z_tau over N4 consecutive slot
intervals on top of the R16 spatial/frequency compression: Q = 2 Doppler
shifts are selected per layer, with n4^(0) = 0 fixed and the second shift
indicated by the offset i_{1,10,l} (n4^(1) = i_{1,10,l} + 1).

When N4 = 1 the codebook degenerates *exactly* to the R16 Enhanced Type II
codebook (the paper states this explicitly); the test suite asserts PMI-level
equality against ``R16Type2Codebook``.

UE-side prediction: ``select`` receives the channel for the N4 future slot
intervals (genie-aided prediction, the standard upper-bound assumption in
3GPP Doppler-codebook evaluations) and fits the Q temporal bases to them, so
``precoder`` yields per-interval predicted precoding matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from math import comb

import numpy as np

from ..config import R18_PARAM_COMBOS, AntennaConfig, m_v
from ..utils import combinatorics as cb
from ..utils import dft
from ..utils import quantization as qt
from . import _spatial
from .base import CodebookScheme
from .etype2_r16 import select_taps


@dataclass
class R18Type2PMI:
    rank: int
    q1: int | None = None
    q2: int | None = None
    i12: int | None = None
    i15: int | None = None
    i16: list[int] = field(default_factory=list)
    i110: list[int] = field(default_factory=list)  # second-shift offset per layer
    i17: np.ndarray | None = None  # bitmap (v, Q, Mv, 2L)
    i18: list[int] = field(default_factory=list)
    k1: np.ndarray | None = None  # (v, 2)
    k2: np.ndarray | None = None  # (v, Q, Mv, 2L)
    c: np.ndarray | None = None  # (v, Q, Mv, 2L)


def encode_i18(bitmap_l: np.ndarray, i_star: int, tau_star: int, rank: int, L: int) -> int:
    """Strongest coefficient among all shifts of the strongest tap (f = 0).

    rank 1: position among the nonzero f=0 bitmap bits, flattened I = 2L*tau+i;
    rank > 1: I = 2L*tau* + i* directly.
    """
    flat_index = 2 * L * tau_star + i_star
    if rank == 1:
        bits_f0 = bitmap_l[:, 0, :].reshape(-1)  # (Q*2L,) in I order
        return int(np.cumsum(bits_f0)[flat_index]) - 1
    return flat_index


def decode_i18(i18: int, bitmap_l: np.ndarray, rank: int, L: int) -> tuple[int, int]:
    """-> (i*, tau*)."""
    if rank == 1:
        bits_f0 = bitmap_l[:, 0, :].reshape(-1)
        flat_index = int(np.argmax(np.cumsum(bits_f0) == i18 + 1))
    else:
        flat_index = i18
    return flat_index % (2 * L), flat_index // (2 * L)


class R18Type2Codebook(CodebookScheme):
    name = "R18 eType II Doppler"
    N_PSK = 16

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        N4: int = 4,
        param_combination: int = 3,
        R: int = 1,
        ri_restriction: np.ndarray | None = None,
    ) -> None:
        if N4 not in (1, 2, 4, 8):
            raise ValueError("N4 must be in {1, 2, 4, 8}")
        self.antenna = antenna
        self.N3 = N3
        self.N4 = N4
        if R not in (1, 2):
            raise ValueError("R must be 1 or 2")
        self.R = R
        if ri_restriction is None:
            ri_restriction = np.ones(4, dtype=bool)
        self.ri_restriction = np.asarray(ri_restriction, dtype=bool)
        if self.ri_restriction.shape != (4,):
            raise ValueError("typeII-Doppler-RI-Restriction-r18 must have 4 bits [r0..r3]")
        # Configuration bars of 5.2.2.2.10: "The UE is not expected to be
        # configured with paramCombination-Doppler-r18 equal to ...".
        if antenna.P == 4 and param_combination >= 4:
            raise ValueError(
                f"paramCombination-Doppler-r18={param_combination} is not supported "
                f"at P_CSI-RS=4 (5.2.2.2.10 bars combinations 4-9)"
            )
        if param_combination in (8, 9):
            if antenna.P < 32:
                raise ValueError(
                    f"paramCombination-Doppler-r18={param_combination} requires "
                    f"P_CSI-RS >= 32 (got {antenna.P})"
                )
            if R == 2:
                raise ValueError(
                    f"paramCombination-Doppler-r18={param_combination} is not "
                    f"supported with R=2"
                )
            if bool(self.ri_restriction[2] or self.ri_restriction[3]):
                raise ValueError(
                    f"paramCombination-Doppler-r18={param_combination} requires ranks "
                    f"3-4 disallowed (typeII-Doppler-RI-Restriction-r18 with r_i=0 "
                    f"for i>1)"
                )
        self.combo = R18_PARAM_COMBOS[param_combination]
        self.L = self.combo.L
        self.Q = 1 if N4 == 1 else 2  # protocol freezes Q = 2 (1 when N4 = 1)
        if self.L > antenna.n_ports_per_pol:
            raise ValueError(
                f"L={self.L} beams cannot be drawn from an orthogonal group of "
                f"N1*N2={antenna.n_ports_per_pol} beams"
            )
        if N3 < 1:
            raise ValueError("N3 must be positive")
        for v in (1, 2, 3, 4):  # fail at construction, not first use
            if v >= 3 and self.combo.p_v34 is None:
                continue
            self.Mv(v)

    def Mv(self, rank: int) -> int:
        val = m_v(self.combo.p_v(rank), self.N3, self.R)
        if val > self.N3:
            raise ValueError("M_v exceeds N3")
        return val

    @property
    def K0(self) -> int:
        M1 = m_v(self.combo.p_v(1), self.N3, self.R)
        return math.ceil(2 * self.combo.beta * self.L * M1 * self.Q)

    def shifts(self, pmi: R18Type2PMI, li: int) -> list[int]:
        if self.Q == 1:
            return [0]
        return [0, pmi.i110[li] + 1]

    def _basis(self, pmi: R18Type2PMI) -> np.ndarray:
        return _spatial.basis_regular(self.antenna, pmi.q1, pmi.q2, pmi.i12, self.L)

    # -- gNB side ----------------------------------------------------------

    def _layer_coefficients(self, pmi: R18Type2PMI, li: int) -> np.ndarray:
        """Quantized coefficients x (2L, Mv, Q), p1 included."""
        p1 = qt.R16_REF_AMP[pmi.k1[li]]
        p2 = qt.R16_DIFF_AMP[pmi.k2[li]]  # (Q, Mv, 2L)
        phi = qt.phase_value(pmi.c[li], self.N_PSK)
        x = (p2 * phi * pmi.i17[li]).transpose(2, 1, 0)  # (2L, Mv, Q)
        pol = np.repeat([0, 1], self.L)
        return x * p1[pol][:, None, None]

    def precoder(self, pmi: R18Type2PMI) -> np.ndarray:
        from .validate import validate_r18

        validate_r18(self, pmi)
        a = self.antenna
        Mv = self.Mv(pmi.rank)
        B = self._basis(pmi)
        W = np.zeros((self.N4, self.N3, a.P, pmi.rank), dtype=complex)
        for li in range(pmi.rank):
            taps = cb.decode_taps(pmi.i16[li], self.N3, Mv, pmi.i15)
            Y = dft.freq_basis(self.N3, np.array(taps)).T  # (N3, Mv)
            Z = dft.time_basis(self.N4, np.array(self.shifts(pmi, li))).T  # (N4, Q)
            x = self._layer_coefficients(pmi, li)  # (2L, Mv, Q)
            ct = np.einsum("afq,tf,iq->ati", x, Y, Z)  # (2L, N3, N4)
            gamma = np.sum(np.abs(ct) ** 2, axis=0)  # (N3, N4)
            gamma = np.where(gamma == 0, 1.0, gamma)
            w = np.concatenate([B.T @ ct[: self.L].reshape(self.L, -1),
                                B.T @ ct[self.L :].reshape(self.L, -1)], axis=0)
            w = w.reshape(a.P, self.N3, self.N4)
            W[:, :, :, li] = (w / np.sqrt(a.n_ports_per_pol * gamma)).transpose(2, 1, 0)
        return W / np.sqrt(pmi.rank)

    # -- UE side -----------------------------------------------------------

    def select(self, H: np.ndarray, rank: int = 1) -> R18Type2PMI:
        if not 1 <= rank <= 4:
            raise ValueError("R18 eType II supports ranks 1-4")
        if not self.ri_restriction[rank - 1]:
            raise ValueError(
                f"rank {rank} prohibited by typeII-Doppler-RI-Restriction-r18 (r{rank - 1}=0)"
            )
        H = np.asarray(H)
        if H.shape[0] != self.N4:
            raise ValueError(f"channel must cover the N4={self.N4} slot intervals")
        if H.shape[1] != self.N3:
            raise ValueError(f"channel has {H.shape[1]} frequency units, expected {self.N3}")
        Mv = self.Mv(rank)
        L2 = 2 * self.L

        targets = np.stack(
            [_spatial.aligned_eigen_targets(H[s], rank) for s in range(self.N4)]
        )  # (N4, N3, P, v)
        flat = targets.reshape(self.N4 * self.N3, self.antenna.P, rank)

        pmi = R18Type2PMI(rank=rank)
        pmi.q1, pmi.q2, pmi.i12 = _spatial.select_group_and_beams(self.antenna, flat, self.L)
        B = self._basis(pmi)
        coeff = _spatial.ls_coefficients(B, flat, float(self.antenna.n_ports_per_pol))
        coeff = coeff.reshape(rank, self.N4, self.N3, L2)

        pmi.i17 = np.zeros((rank, self.Q, Mv, L2), dtype=bool)
        pmi.k1 = np.ones((rank, 2), dtype=int)
        pmi.k2 = np.zeros((rank, self.Q, Mv, L2), dtype=int)
        pmi.c = np.zeros((rank, self.Q, Mv, L2), dtype=int)

        kept, stars = [], []
        m_init_common: int | None = None
        for li in range(rank):
            C = coeff[li].transpose(2, 0, 1)  # (2L, N4, N3)
            iref = int(np.argmax(np.sum(np.abs(C) ** 2, axis=(1, 2))))
            C = C * np.exp(-1j * np.angle(C[iref]))[None, :, :]
            Ctap = np.fft.fft(C, axis=2) / self.N3  # (2L, N4, N3)
            tap_energy = np.sum(np.abs(Ctap) ** 2, axis=(0, 1))
            n_star = int(np.argmax(tap_energy))
            Ctap = np.roll(Ctap, -n_star, axis=2)
            taps = select_taps(np.roll(tap_energy, -n_star), Mv, self.N3, m_init_common)
            i16, i15 = cb.encode_taps(taps, self.N3, Mv, m_initial=m_init_common)
            pmi.i16.append(i16)
            if i15 is not None and m_init_common is None:
                m_init_common = i15 if i15 == 0 else i15 - 2 * Mv
            Csel = Ctap[:, :, taps]  # (2L, N4, Mv)
            Cdop = np.fft.fft(Csel, axis=1) / self.N4  # (2L, N4 shifts, Mv)
            if self.Q == 2:
                shift_energy = np.sum(np.abs(Cdop) ** 2, axis=(0, 2))
                n4_1 = 1 + int(np.argmax(shift_energy[1:]))
                pmi.i110.append(n4_1 - 1)
                x = Cdop[:, [0, n4_1], :].transpose(0, 2, 1)  # (2L, Mv, Q)
            else:
                x = Cdop[:, [0], :].transpose(0, 2, 1)
            # strongest coefficient among all shifts of the strongest tap f=0
            f0 = np.abs(x[:, 0, :])  # (2L, Q)
            i_star, tau_star = np.unravel_index(int(np.argmax(f0)), f0.shape)
            stars.append((int(i_star), int(tau_star)))
            x = x / x[i_star, 0, tau_star]
            mags = np.abs(x)
            order = np.argsort(mags.reshape(-1))[::-1]
            keep = order[: min(self.K0, mags.size)]
            bm = np.zeros(mags.shape, dtype=bool)  # (2L, Mv, Q)
            bm[np.unravel_index(keep, mags.shape)] = True
            bm &= mags > qt.R16_DIFF_AMP[0] / 2
            bm[i_star, 0, tau_star] = True
            pmi.i17[li] = bm.transpose(2, 1, 0)
            kept.append(x * bm)
        if m_init_common is not None:
            pmi.i15 = m_init_common if m_init_common == 0 else m_init_common + 2 * Mv

        self._enforce_total_budget(pmi, kept, stars)

        for li in range(rank):
            x, bm = kept[li], pmi.i17[li].transpose(2, 1, 0)  # (2L, Mv, Q)
            i_star, tau_star = stars[li]
            pmi.i18.append(encode_i18(pmi.i17[li], i_star, tau_star, rank, self.L))
            p_star = i_star // self.L
            mags = np.abs(x)
            for pol in (0, 1):
                rows = slice(0, self.L) if pol == 0 else slice(self.L, L2)
                if pol == p_star:
                    pmi.k1[li, pol] = 15
                    continue
                ref = mags[rows][bm[rows]].max() if bm[rows].any() else 0.0
                pmi.k1[li, pol] = max(int(qt.quantize_amplitude(ref, qt.R16_REF_AMP)), 1)
            p1 = qt.R16_REF_AMP[pmi.k1[li]]
            for i in range(L2):
                for f in range(self.Mv(rank)):
                    for tau in range(self.Q):
                        if not bm[i, f, tau]:
                            continue
                        rel = mags[i, f, tau] / p1[i // self.L]
                        pmi.k2[li, tau, f, i] = int(
                            qt.quantize_amplitude(min(rel, 1.0), qt.R16_DIFF_AMP)
                        )
                        pmi.c[li, tau, f, i] = int(
                            qt.quantize_phase(np.angle(x[i, f, tau]), self.N_PSK)
                        )
            pmi.k2[li, tau_star, 0, i_star] = 7
            pmi.c[li, tau_star, 0, i_star] = 0
        return pmi

    def _enforce_total_budget(self, pmi, kept, stars) -> None:
        budget = 2 * self.K0
        total = int(pmi.i17.sum())
        if total <= budget:
            return
        entries = []
        for li in range(pmi.rank):
            bm = pmi.i17[li].transpose(2, 1, 0)
            i_star, tau_star = stars[li]
            for idx in np.argwhere(bm):
                i, f, tau = (int(v) for v in idx)
                if not (i == i_star and f == 0 and tau == tau_star):
                    entries.append((abs(kept[li][i, f, tau]), li, i, f, tau))
        entries.sort()
        for mag, li, i, f, tau in entries:
            if total <= budget:
                break
            pmi.i17[li][tau, f, i] = False
            kept[li][i, f, tau] = 0.0
            total -= 1

    # -- overhead ----------------------------------------------------------

    def overhead_bits(self, pmi: R18Type2PMI) -> dict[str, int]:
        a = self.antenna
        L, Mv, v = self.L, self.Mv(pmi.rank), pmi.rank
        bits: dict[str, int] = {
            "i11": math.ceil(math.log2(a.O1 * a.O2)),
            "i12": math.ceil(math.log2(comb(a.N1 * a.N2, L))),
        }
        if self.N3 > 19:
            bits["i15"] = math.ceil(math.log2(2 * Mv))
            bits["i16"] = v * math.ceil(math.log2(comb(2 * Mv - 1, Mv - 1)))
        else:
            bits["i16"] = v * math.ceil(math.log2(comb(self.N3 - 1, Mv - 1)))
        bits["i17"] = v * 2 * L * Mv * self.Q
        K_nz = int(pmi.i17.sum())
        # TS 38.212 Table 6.3.2.1.2-1C: rank 1 spends ceil(log2(K^NZ)) bits on
        # the strongest-coefficient indicator; ranks 2-4 spend ceil(log2(2LQ))
        # per layer.
        if v == 1:
            bits["i18"] = math.ceil(math.log2(K_nz)) if K_nz > 1 else 0
        else:
            bits["i18"] = v * math.ceil(math.log2(2 * L * self.Q))
        if self.N4 > 1:
            bits["i110"] = v * math.ceil(math.log2(self.N4 - 1))  # 0 bits when N4 = 2
        bits["i23"] = 4 * v
        bits["i24"] = 3 * (K_nz - v)
        bits["i25"] = 4 * (K_nz - v)
        return bits
