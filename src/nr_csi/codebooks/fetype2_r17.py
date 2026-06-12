"""R17 Further Enhanced Type II port-selection codebook, ranks 1-4.

Reconstruction (paper Table tabfesp, TS 38.214 Table 5.2.2.2.7-3):

    w^l_t = 1/sqrt(gamma_{t,l}) *
            [ sum_i v_{m^(i)} p1_{l,0} sum_f y_t^(f) p2_{l,i,f} phi_{l,i,f} ;
              sum_i v_{m^(i)} p1_{l,1} sum_f y_t^(f) p2_{l,i+L,f} phi_{l,i+L,f} ]

Key R17 changes vs. the R16 PS codebook:
* the UE freely selects any L = K1/2 = alpha*P/2 ports via the Algorithm 4
  codec (errata-corrected); if alpha = 1, all ports are used and i_{1,2}
  is not reported;
* M in {1, 2} delay taps, common to all layers; taps are confined to the
  window {0..N-1} (uplink/downlink delay reciprocity lets the gNB
  pre-compensate delays, so the dominant tap sits at 0);
  - M = 1, or M = 2 with N = 2: i_{1,6} not reported,
  - M = 2 with N = 4: i_{1,6} reports the nonzero offset of the second tap;
* the strongest coefficient is indicated directly: i_{1,8,l} = K1*f* + i*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from math import comb

import numpy as np

from ..config import AntennaConfig, R17_PARAM_COMBOS
from ..utils import combinatorics as cb
from ..utils import dft
from ..utils import quantization as qt
from . import _spatial
from .base import CodebookScheme


@dataclass
class R17Type2PMI:
    rank: int
    i12: int | None = None  # port combination (None when alpha = 1)
    i16: int | None = None  # second-tap offset indicator (None unless M=2, N=4)
    i17: np.ndarray | None = None  # bitmap (v, M, K1), bool
    i18: list[int] = field(default_factory=list)  # K1*f* + i* per layer
    k1: np.ndarray | None = None  # (v, 2) reference amplitude indices
    k2: np.ndarray | None = None  # (v, M, K1) differential amplitude indices
    c: np.ndarray | None = None  # (v, M, K1) phase indices


class R17Type2Codebook(CodebookScheme):
    name = "R17 FeType II PS"
    N_PSK = 16

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        param_combination: int = 7,
        N_window: int = 4,
    ) -> None:
        self.antenna = antenna
        self.N3 = N3
        self.combo = R17_PARAM_COMBOS[param_combination]
        self.M = self.combo.M
        self.alpha = self.combo.alpha
        K1 = self.alpha * antenna.P
        if K1 % 2 != 0 or K1 <= 0:
            raise ValueError(f"alpha*P_CSI-RS = {K1} must be a positive even integer")
        if K1 > antenna.P:
            raise ValueError(f"K1 = alpha*P = {K1} exceeds the {antenna.P} CSI-RS ports")
        if N3 < 1:
            raise ValueError("N3 must be positive")
        self.K1 = int(K1)
        self.L = self.K1 // 2
        if N_window not in (2, 4):
            raise ValueError("tap window N must be 2 or 4")
        self.N_window = N_window

    @property
    def K0(self) -> int:
        return math.ceil(self.combo.beta * self.K1 * self.M)

    def taps(self, pmi: R17Type2PMI) -> list[int]:
        if self.M == 1:
            return [0]
        if min(self.N_window, self.N3) == 2:
            return [0, 1]
        return [0, pmi.i16 + 1]

    def _basis(self, pmi: R17Type2PMI) -> np.ndarray:
        half = self.antenna.P // 2
        ports = list(range(self.L)) if pmi.i12 is None else cb.decode_ports(
            pmi.i12, self.antenna.P, self.L
        )
        B = np.zeros((self.L, half))
        for i, p in enumerate(ports):
            B[i, p] = 1.0
        return B

    # -- gNB side ----------------------------------------------------------

    def _layer_coefficients(self, pmi: R17Type2PMI, li: int) -> np.ndarray:
        p1 = qt.R16_REF_AMP[pmi.k1[li]]
        p2 = qt.R16_DIFF_AMP[pmi.k2[li]]  # (M, K1)
        phi = qt.phase_value(pmi.c[li], self.N_PSK)
        x = (p2 * phi * pmi.i17[li]).T  # (K1, M)
        pol = np.repeat([0, 1], self.L)
        return x * p1[pol][:, None]

    def precoder(self, pmi: R17Type2PMI) -> np.ndarray:
        from .validate import validate_r17

        validate_r17(self, pmi)
        a = self.antenna
        B = self._basis(pmi)
        Y = dft.freq_basis(self.N3, np.array(self.taps(pmi))).T  # (N3, M)
        W = np.zeros((1, self.N3, a.P, pmi.rank), dtype=complex)
        for li in range(pmi.rank):
            x = self._layer_coefficients(pmi, li)  # (K1, M)
            ct = x @ Y.T  # (K1, N3)
            gamma = np.sum(np.abs(ct) ** 2, axis=0)
            gamma = np.where(gamma == 0, 1.0, gamma)
            w = np.concatenate([B.T @ ct[: self.L], B.T @ ct[self.L :]], axis=0)
            W[0, :, :, li] = (w / np.sqrt(gamma)).T
        return W / np.sqrt(pmi.rank)

    # -- UE side -----------------------------------------------------------

    def select(self, H: np.ndarray, rank: int = 1) -> R17Type2PMI:
        if not 1 <= rank <= 4:
            raise ValueError("R17 FeType II supports ranks 1-4")
        H = np.asarray(H)[-1]
        if H.shape[0] != self.N3:
            raise ValueError(f"channel has {H.shape[0]} frequency units, expected {self.N3}")
        targets = _spatial.aligned_eigen_targets(H, rank)
        a = self.antenna
        half = a.P // 2

        pmi = R17Type2PMI(rank=rank)
        if self.alpha < 1:
            energy = np.sum(np.abs(targets) ** 2, axis=(0, 2))
            port_energy = energy[:half] + energy[half:]
            ports = sorted(np.argsort(port_energy)[::-1][: self.L].tolist())
            pmi.i12 = cb.encode_ports(ports, a.P)
        B = self._basis(pmi)
        coeff = _spatial.ls_coefficients(B, targets, 1.0)  # (v, N3, K1)

        # common tap pair for all layers: 0 plus the strongest in-window offset
        if self.M == 2 and min(self.N_window, self.N3) > 2:
            e_tap = np.zeros(self.N3)
            for li in range(rank):
                C = coeff[li].T
                iref = int(np.argmax(np.sum(np.abs(C) ** 2, axis=1)))
                C = C * np.exp(-1j * np.angle(C[iref]))[None, :]
                e_tap += np.sum(np.abs(np.fft.fft(C, axis=1) / self.N3) ** 2, axis=0)
            window = e_tap[1 : min(self.N_window, self.N3)]
            pmi.i16 = int(np.argmax(window))
        taps = self.taps(pmi)
        Y = dft.freq_basis(self.N3, np.array(taps))  # (M, N3)

        pmi.i17 = np.zeros((rank, self.M, self.K1), dtype=bool)
        pmi.k1 = np.ones((rank, 2), dtype=int)
        pmi.k2 = np.zeros((rank, self.M, self.K1), dtype=int)
        pmi.c = np.zeros((rank, self.M, self.K1), dtype=int)

        kept, stars = [], []
        for li in range(rank):
            C = coeff[li].T  # (K1, N3)
            iref = int(np.argmax(np.sum(np.abs(C) ** 2, axis=1)))
            C = C * np.exp(-1j * np.angle(C[iref]))[None, :]
            # LS projection onto the selected taps (orthogonal DFT columns)
            Ct = (C @ Y.conj().T) / self.N3  # (K1, M)
            f_star = int(np.argmax(np.sum(np.abs(Ct) ** 2, axis=0)))
            i_star = int(np.argmax(np.abs(Ct[:, f_star])))
            stars.append((i_star, f_star))
            Ct = Ct / Ct[i_star, f_star]
            mags = np.abs(Ct)
            order = np.argsort(mags.reshape(-1))[::-1]
            keep = order[: min(self.K0, mags.size)]
            bm = np.zeros(mags.shape, dtype=bool)
            bm[np.unravel_index(keep, mags.shape)] = True
            bm &= mags > qt.R16_DIFF_AMP[0] / 2
            bm[i_star, f_star] = True
            pmi.i17[li] = bm.T
            kept.append(Ct * bm)

        self._enforce_total_budget(pmi, kept, stars)

        for li in range(rank):
            Ct, bm = kept[li], pmi.i17[li].T
            i_star, f_star = stars[li]
            pmi.i18.append(self.K1 * f_star + i_star)
            p_star = i_star // self.L
            mags = np.abs(Ct)
            for pol in (0, 1):
                rows = slice(0, self.L) if pol == 0 else slice(self.L, self.K1)
                if pol == p_star:
                    pmi.k1[li, pol] = 15
                    continue
                ref = mags[rows][bm[rows]].max() if bm[rows].any() else 0.0
                pmi.k1[li, pol] = max(int(qt.quantize_amplitude(ref, qt.R16_REF_AMP)), 1)
            p1 = qt.R16_REF_AMP[pmi.k1[li]]
            for i in range(self.K1):
                for f in range(self.M):
                    if not bm[i, f]:
                        continue
                    rel = mags[i, f] / p1[i // self.L]
                    pmi.k2[li, f, i] = int(qt.quantize_amplitude(min(rel, 1.0), qt.R16_DIFF_AMP))
                    pmi.c[li, f, i] = int(qt.quantize_phase(np.angle(Ct[i, f]), self.N_PSK))
            pmi.k2[li, f_star, i_star] = 7
            pmi.c[li, f_star, i_star] = 0
        return pmi

    def _enforce_total_budget(self, pmi, kept, stars) -> None:
        budget = 2 * self.K0
        total = int(pmi.i17.sum())
        if total <= budget:
            return
        entries = []
        for li in range(pmi.rank):
            bm = pmi.i17[li].T
            i_star, f_star = stars[li]
            for i in range(self.K1):
                for f in range(self.M):
                    if bm[i, f] and not (i == i_star and f == f_star):
                        entries.append((abs(kept[li][i, f]), li, i, f))
        entries.sort()
        for mag, li, i, f in entries:
            if total <= budget:
                break
            pmi.i17[li][f, i] = False
            kept[li][i, f] = 0.0
            total -= 1

    # -- overhead ----------------------------------------------------------

    def overhead_bits(self, pmi: R17Type2PMI) -> dict[str, int]:
        v = pmi.rank
        bits: dict[str, int] = {}
        if self.alpha < 1:
            bits["i12"] = math.ceil(math.log2(comb(self.antenna.P // 2, self.L)))
        if self.M == 2 and min(self.N_window, self.N3) > 2:
            bits["i16"] = math.ceil(math.log2(self.N_window - 1))
        bits["i17"] = v * self.K1 * self.M
        bits["i18"] = v * math.ceil(math.log2(self.K1 * self.M))
        K_nz = int(pmi.i17.sum())
        bits["i23"] = 4 * v
        bits["i24"] = 3 * (K_nz - v)
        bits["i25"] = 4 * (K_nz - v)
        return bits
