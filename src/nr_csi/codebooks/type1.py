"""R15 Type I single-panel codebook, TS 38.214 section 5.2.2.2.1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import AntennaConfig
from ..utils import dft
from .base import CodebookScheme

_MODE2_OFFSETS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def i13_offsets(N1: int, N2: int, O1: int, O2: int) -> list[tuple[int, int]]:
    """Rank-2 beam offsets from TS 38.214 Table 5.2.2.2.1-3."""
    if N1 > N2 > 1:
        return [(0, 0), (O1, 0), (0, O2), (2 * O1, 0)]
    if N1 == N2:
        return [(0, 0), (O1, 0), (0, O2), (O1, O2)]
    if N1 > 2 and N2 == 1:
        return [(0, 0), (O1, 0), (2 * O1, 0), (3 * O1, 0)]
    if N1 == 2 and N2 == 1:
        return [(0, 0), (O1, 0)]
    raise ValueError(f"unsupported (N1,N2)=({N1},{N2}) for rank-2 Type I")


def i13_offsets_rank34(N1: int, N2: int, O1: int, O2: int) -> list[tuple[int, int]]:
    """Rank-3/4 offsets for P < 16, TS 38.214 Table 5.2.2.2.1-4."""
    table = {
        (2, 1): [(O1, 0)],
        (4, 1): [(O1, 0), (2 * O1, 0), (3 * O1, 0)],
        (6, 1): [(O1, 0), (2 * O1, 0), (3 * O1, 0), (4 * O1, 0)],
        (2, 2): [(O1, 0), (0, O2), (O1, O2)],
        (3, 2): [(O1, 0), (0, O2), (O1, O2), (2 * O1, 0)],
    }
    try:
        return table[(N1, N2)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported (N1,N2)=({N1},{N2}) for rank-3/4 Type I with P < 16"
        ) from exc


@dataclass
class Type1PMI:
    rank: int
    mode: int
    i11: int
    i12: int
    i2: np.ndarray
    i13: int | None = None  # ranks 2-4


class Type1Codebook(CodebookScheme):
    name = "R15 Type I"

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int = 1,
        mode: int = 1,
        beam_restriction: np.ndarray | None = None,
        rank_restriction: np.ndarray | None = None,
        selection_snr_db: float = 10.0,
    ) -> None:
        if antenna.Ng != 1:
            raise ValueError("Type I single-panel requires Ng=1")
        if N3 < 1:
            raise ValueError("N3 must be positive")
        if mode not in (1, 2):
            raise ValueError("codebookMode must be 1 or 2")
        self.antenna = antenna
        self.N3 = N3
        self.mode = mode
        n_bits = math.prod(antenna.n_beams)
        if beam_restriction is None:
            beam_restriction = np.ones(n_bits, dtype=bool)
        self.beam_restriction = np.asarray(beam_restriction, dtype=bool)
        if self.beam_restriction.shape != (n_bits,):
            raise ValueError(f"beam restriction bitmap must have {n_bits} bits")
        if rank_restriction is None:
            rank_restriction = np.ones(8, dtype=bool)
        self.rank_restriction = np.asarray(rank_restriction, dtype=bool)
        if self.rank_restriction.shape != (8,):
            raise ValueError("rank restriction r = [r0..r7] must have 8 bits")
        self.selection_rho = 10 ** (selection_snr_db / 10)

    def _beam_allowed(self, l: int, m: int) -> bool:
        G1, G2 = self.antenna.n_beams
        return bool(self.beam_restriction[(m % G2) + G2 * (l % G1)])

    def _n_i11(self, rank: int) -> int:
        G1, _ = self.antenna.n_beams
        if rank in (1, 2) and self.mode == 2:
            return G1 // 2
        if rank in (3, 4) and self.antenna.P >= 16:
            return G1 // 2
        if rank in (7, 8) and (self.antenna.N1, self.antenna.N2) == (4, 1):
            return G1 // 2
        return G1

    def _n_i12(self, rank: int) -> int:
        _, G2 = self.antenna.n_beams
        if rank in (1, 2) and self.mode == 2:
            return max(G2 // 2, 1)
        if rank in (7, 8) and self.antenna.N2 == 2 and self.antenna.N1 > 2:
            return G2 // 2
        return G2

    def _n_i13(self, rank: int) -> int:
        a = self.antenna
        if rank == 2:
            return len(i13_offsets(a.N1, a.N2, a.O1, a.O2))
        if rank in (3, 4):
            if a.P >= 16:
                return 4
            return len(i13_offsets_rank34(a.N1, a.N2, a.O1, a.O2))
        return 1

    def _n_i2(self, rank: int) -> int:
        if rank >= 3:
            return 2
        if self.mode == 1:
            return 4 if rank == 1 else 2
        return 16 if rank == 1 else 8

    def _beam_and_phase(self, pmi: Type1PMI, t: int) -> tuple[int, int, int]:
        i2 = int(pmi.i2[t])
        if pmi.mode == 1 or pmi.rank >= 3:
            return pmi.i11, pmi.i12, i2
        n_phases = 4 if pmi.rank == 1 else 2
        offsets = (
            _MODE2_OFFSETS
            if self.antenna.N2 > 1
            else [(0, 0), (1, 0), (2, 0), (3, 0)]
        )
        k1p, k2p = offsets[i2 // n_phases]
        return 2 * pmi.i11 + k1p, 2 * pmi.i12 + k2p, i2 % n_phases

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
        columns = [
            np.concatenate([v1, phi * v1]),
            np.concatenate([v2, -phi * v2]),
        ]
        return np.stack(columns, axis=1) / math.sqrt(2 * a.P)

    def _w_rank34_small(self, rank: int, l: int, m: int, i13: int, n: int) -> np.ndarray:
        a = self.antenna
        k1, k2 = i13_offsets_rank34(a.N1, a.N2, a.O1, a.O2)[i13]
        v1 = dft.spatial_beam(a, l, m)
        v2 = dft.spatial_beam(a, l + k1, m + k2)
        phi = np.exp(1j * np.pi * n / 2)
        columns = [
            np.concatenate([v1, phi * v1]),
            np.concatenate([v2, phi * v2]),
            np.concatenate([v1, -phi * v1]),
        ]
        if rank == 4:
            columns.append(np.concatenate([v2, -phi * v2]))
        return np.stack(columns, axis=1) / math.sqrt(rank * a.P)

    def _w_rank34_large(self, rank: int, l: int, m: int, p: int, n: int) -> np.ndarray:
        a = self.antenna
        half = np.kron(dft.steering(a.N1 // 2, a.O1, l), dft.steering(a.N2, a.O2, m))
        phi = np.exp(1j * np.pi * n / 2)
        theta = np.exp(1j * np.pi * p / 4)
        if rank == 3:
            coefficients = np.array(
                [
                    [1, 1, 1],
                    [theta, -theta, theta],
                    [phi, phi, -phi],
                    [phi * theta, -phi * theta, -phi * theta],
                ],
                dtype=complex,
            )
        else:
            coefficients = np.array(
                [
                    [1, 1, 1, 1],
                    [theta, -theta, theta, -theta],
                    [phi, phi, -phi, -phi],
                    [phi * theta, -phi * theta, -phi * theta, phi * theta],
                ],
                dtype=complex,
            )
        return np.concatenate([c * half[:, None] for c in coefficients], axis=0) / math.sqrt(
            rank * a.P
        )

    def _fixed_beams(self, rank: int, l: int, m: int) -> list[np.ndarray]:
        a = self.antenna
        if rank in (5, 6):
            indices = (
                [(l, m), (l + a.O1, m), (l + a.O1, m + a.O2)]
                if a.N2 > 1
                else [(l + q * a.O1, 0) for q in range(3)]
            )
        elif a.N2 == 1:
            indices = [(l + q * a.O1, 0) for q in range(4)]
        else:
            indices = [
                (l, m),
                (l + a.O1, m),
                (l, m + a.O2),
                (l + a.O1, m + a.O2),
            ]
        return [dft.spatial_beam(a, lp, mp) for lp, mp in indices]

    def _w_rank58(self, rank: int, l: int, m: int, n: int) -> np.ndarray:
        a = self.antenna
        beams = self._fixed_beams(rank, l, m)
        phi = np.exp(1j * np.pi * n / 2)
        if rank == 5:
            top = [beams[0], beams[0], beams[1], beams[1], beams[2]]
            bottom = [phi * beams[0], -phi * beams[0], beams[1], -beams[1], beams[2]]
        elif rank == 6:
            top = [beams[0], beams[0], beams[1], beams[1], beams[2], beams[2]]
            bottom = [
                phi * beams[0],
                -phi * beams[0],
                phi * beams[1],
                -phi * beams[1],
                beams[2],
                -beams[2],
            ]
        elif rank == 7:
            top = [beams[0], beams[0], beams[1], beams[2], beams[2], beams[3], beams[3]]
            bottom = [
                phi * beams[0],
                -phi * beams[0],
                phi * beams[1],
                beams[2],
                -beams[2],
                beams[3],
                -beams[3],
            ]
        else:
            top = [
                beams[0],
                beams[0],
                beams[1],
                beams[1],
                beams[2],
                beams[2],
                beams[3],
                beams[3],
            ]
            bottom = [
                phi * beams[0],
                -phi * beams[0],
                phi * beams[1],
                -phi * beams[1],
                beams[2],
                -beams[2],
                beams[3],
                -beams[3],
            ]
        columns = [np.concatenate(pair) for pair in zip(top, bottom)]
        return np.stack(columns, axis=1) / math.sqrt(rank * a.P)

    def _w_at(self, pmi: Type1PMI, t: int) -> np.ndarray:
        a = self.antenna
        l, m, n = self._beam_and_phase(pmi, t)
        if pmi.rank == 1:
            return self._w_rank1(l, m, n)
        if pmi.rank == 2:
            k1, k2 = i13_offsets(a.N1, a.N2, a.O1, a.O2)[pmi.i13]
            return self._w_rank2(l, l + k1, m, m + k2, n)
        if pmi.rank in (3, 4):
            if a.P < 16:
                return self._w_rank34_small(pmi.rank, l, m, pmi.i13, n)
            return self._w_rank34_large(pmi.rank, l, m, pmi.i13, n)
        return self._w_rank58(pmi.rank, l, m, n)

    def precoder(self, pmi: Type1PMI) -> np.ndarray:
        from .validate import validate_type1

        validate_type1(self, pmi)
        W = np.empty((1, self.N3, self.antenna.P, pmi.rank), dtype=complex)
        for t in range(self.N3):
            W[0, t] = self._w_at(pmi, t)
        return W

    def _candidate_allowed(self, pmi: Type1PMI) -> bool:
        a = self.antenna
        l, m, _ = self._beam_and_phase(pmi, 0)
        if pmi.rank in (3, 4) and a.P >= 16:
            return all(self._beam_allowed(2 * l + delta, m) for delta in (-1, 0, 1))
        if pmi.rank == 1:
            indices = [(l, m)]
        elif pmi.rank == 2:
            k1, k2 = i13_offsets(a.N1, a.N2, a.O1, a.O2)[pmi.i13]
            indices = [(l, m), (l + k1, m + k2)]
        elif pmi.rank in (3, 4):
            k1, k2 = i13_offsets_rank34(a.N1, a.N2, a.O1, a.O2)[pmi.i13]
            indices = [(l, m), (l + k1, m + k2)]
        else:
            count = 3 if pmi.rank in (5, 6) else 4
            if a.N2 == 1:
                indices = [(l + q * a.O1, 0) for q in range(count)]
            elif count == 3:
                indices = [(l, m), (l + a.O1, m), (l + a.O1, m + a.O2)]
            else:
                indices = [
                    (l, m),
                    (l + a.O1, m),
                    (l, m + a.O2),
                    (l + a.O1, m + a.O2),
                ]
        return all(self._beam_allowed(lp, mp) for lp, mp in indices)

    def select(self, H: np.ndarray, rank: int = 1) -> Type1PMI:
        if not 1 <= rank <= min(8, self.antenna.P):
            raise ValueError(f"Type I rank {rank} unsupported for P={self.antenna.P}")
        if not self.rank_restriction[rank - 1]:
            raise ValueError(
                f"rank {rank} prohibited by typeI-SinglePanel-ri-Restriction (r{rank - 1}=0)"
            )
        H = np.asarray(H)
        if H.ndim != 4:
            raise ValueError("H must be [slot, t, rx, port]")
        Ht = H[-1]
        if Ht.shape[0] != self.N3:
            raise ValueError(f"channel has {Ht.shape[0]} frequency units, expected {self.N3}")
        if Ht.shape[-1] != self.antenna.P:
            raise ValueError(f"channel has {Ht.shape[-1]} ports, expected {self.antenna.P}")

        cands = [
            (i11, i12, i13)
            for i11 in range(self._n_i11(rank))
            for i12 in range(self._n_i12(rank))
            for i13 in range(self._n_i13(rank))
        ]
        n_i2 = self._n_i2(rank)
        Wc = np.empty((len(cands), n_i2, self.antenna.P, rank), dtype=complex)
        allowed = np.zeros((len(cands), n_i2), dtype=bool)
        for ci, (i11, i12, i13) in enumerate(cands):
            for i2 in range(n_i2):
                pmi = Type1PMI(
                    rank,
                    self.mode,
                    i11,
                    i12,
                    np.array([i2]),
                    i13 if rank in (2, 3, 4) else None,
                )
                Wc[ci, i2] = self._w_at(pmi, 0)
                allowed[ci, i2] = self._candidate_allowed(pmi)

        HW = np.einsum("tnp,ckpv->cktnv", Ht, Wc)
        gram = np.einsum("...nv,...nw->...vw", HW.conj(), HW)
        eye = np.eye(rank, dtype=complex)
        _, logdet = np.linalg.slogdet(eye + self.selection_rho * gram)
        rates = np.real(logdet) / math.log(2)
        rates = np.where(allowed[:, :, None], rates, -np.inf)
        metric = rates.max(axis=1).sum(axis=-1)
        feasible = allowed.any(axis=1)
        if not feasible.any():
            raise RuntimeError("no feasible Type I PMI under the configured restriction")
        ci = int(np.argmax(np.where(feasible, metric, -np.inf)))
        i11, i12, i13 = cands[ci]
        return Type1PMI(
            rank,
            self.mode,
            i11,
            i12,
            rates[ci].argmax(axis=0).astype(int),
            i13 if rank in (2, 3, 4) else None,
        )

    def overhead_bits(self, pmi: Type1PMI) -> dict[str, int]:
        bits = {
            "i11": math.ceil(math.log2(self._n_i11(pmi.rank))),
            "i2": self.N3 * math.ceil(math.log2(self._n_i2(pmi.rank))),
        }
        if self.antenna.N2 > 1:
            bits["i12"] = math.ceil(math.log2(self._n_i12(pmi.rank)))
        if pmi.rank in (2, 3, 4):
            bits["i13"] = math.ceil(math.log2(self._n_i13(pmi.rank)))
        return bits
