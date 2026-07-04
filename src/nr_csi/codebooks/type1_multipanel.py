"""R15 Type I multi-panel codebook, TS 38.214 section 5.2.2.2.2."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..config import AntennaConfig
from ..utils import dft
from .base import CodebookScheme
from .type1 import i13_offsets


def i13_offsets_multipanel_rank34(
    N1: int, N2: int, O1: int, O2: int
) -> list[tuple[int, int]]:
    """Rank-3/4 offsets from TS 38.214 Table 5.2.2.2.2-2."""
    table = {
        (2, 1): [(O1, 0)],
        (4, 1): [(O1, 0), (2 * O1, 0), (3 * O1, 0)],
        (8, 1): [(O1, 0), (2 * O1, 0), (3 * O1, 0), (4 * O1, 0)],
        (2, 2): [(O1, 0), (0, O2), (O1, O2)],
        (4, 2): [(O1, 0), (0, O2), (O1, O2), (2 * O1, 0)],
    }
    try:
        return table[(N1, N2)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported (N1,N2)=({N1},{N2}) for rank-3/4 Type I multi-panel"
        ) from exc


@dataclass
class Type1MPPMI:
    rank: int
    mode: int
    i11: int
    i12: int
    i14: tuple[int, ...]
    i2: np.ndarray
    i13: int | None = None


class Type1MultiPanelCodebook(CodebookScheme):
    name = "R15 Type I multi-panel"

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int = 1,
        mode: int = 1,
        beam_restriction: np.ndarray | None = None,
        rank_restriction: np.ndarray | None = None,
        selection_snr_db: float = 10.0,
    ) -> None:
        if antenna.Ng not in (2, 4):
            raise ValueError("Type I multi-panel requires Ng in {2,4}")
        if N3 < 1:
            raise ValueError("N3 must be positive")
        if mode not in (1, 2):
            raise ValueError("codebookMode must be 1 or 2")
        if mode == 2 and antenna.Ng != 2:
            raise ValueError("multi-panel codebook Mode 2 requires Ng=2")
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
            rank_restriction = np.ones(4, dtype=bool)
        self.rank_restriction = np.asarray(rank_restriction, dtype=bool)
        if self.rank_restriction.shape != (4,):
            raise ValueError("multi-panel rank restriction must have 4 bits")
        self.selection_rho = 10 ** (selection_snr_db / 10)
        self._beam_grid: np.ndarray | None = None
        self._search_cache: dict[
            int, tuple[list[tuple[int, int, int, tuple[int, ...]]], list[tuple[int, ...]], np.ndarray]
        ] = {}

    def _beam_allowed(self, l: int, m: int) -> bool:
        G1, G2 = self.antenna.n_beams
        return bool(self.beam_restriction[(m % G2) + G2 * (l % G1)])

    def _n_i13(self, rank: int) -> int:
        a = self.antenna
        if rank == 1:
            return 1
        if rank == 2:
            return len(i13_offsets(a.N1, a.N2, a.O1, a.O2))
        return len(i13_offsets_multipanel_rank34(a.N1, a.N2, a.O1, a.O2))

    def _i14_shape(self) -> tuple[int, ...]:
        if self.mode == 1:
            return (self.antenna.Ng - 1,)
        return (2,)

    def _i2_shape(self) -> tuple[int, ...]:
        return (self.N3,) if self.mode == 1 else (self.N3, 3)

    def _i2_states(self, rank: int) -> list[tuple[int, ...]]:
        if self.mode == 1:
            return [(n,) for n in range(4 if rank == 1 else 2)]
        n0 = range(4 if rank == 1 else 2)
        return list(itertools.product(n0, range(2), range(2)))

    def _offset(self, rank: int, i13: int | None) -> tuple[int, int]:
        a = self.antenna
        if rank == 1:
            return 0, 0
        if rank == 2:
            return i13_offsets(a.N1, a.N2, a.O1, a.O2)[i13]
        return i13_offsets_multipanel_rank34(a.N1, a.N2, a.O1, a.O2)[i13]

    def _beam_vector(self, l: int, m: int) -> np.ndarray:
        """Cached ``dft.spatial_beam(l, m)`` lookup.

        The candidate search calls this with the same handful of distinct
        (l, m) pairs tens of thousands of times (once per i14/i2 state per
        candidate); ``spatial_beam`` is periodic in (l, m) with period
        (G1, G2), so the whole oversampled grid can be precomputed once per
        instance and reused instead of rebuilding each vector from scratch.
        """
        if self._beam_grid is None:
            self._beam_grid = dft.spatial_grid(self.antenna)
        G1, G2 = self.antenna.n_beams
        return self._beam_grid[l % G1, m % G2]

    def _base_vector(
        self,
        l: int,
        m: int,
        i14: tuple[int, ...],
        i2: tuple[int, ...],
        family: int,
    ) -> np.ndarray:
        a = self.antenna
        v = self._beam_vector(l, m)
        if self.mode == 1:
            phi_n = np.exp(1j * np.pi * i2[0] / 2)
            panel_phases = [1, *(np.exp(1j * np.pi * p / 2) for p in i14)]
            blocks = []
            for phase in panel_phases:
                blocks.extend([phase * v, (1 if family == 1 else -1) * phi_n * phase * v])
        else:
            n0, n1, n2 = i2
            phi_n0 = np.exp(1j * np.pi * n0 / 2)
            ap1 = np.exp(1j * np.pi / 4) * np.exp(1j * np.pi * i14[0] / 2)
            ap2 = np.exp(1j * np.pi / 4) * np.exp(1j * np.pi * i14[1] / 2)
            bn1 = np.exp(-1j * np.pi / 4) * np.exp(1j * np.pi * n1 / 2)
            bn2 = np.exp(-1j * np.pi / 4) * np.exp(1j * np.pi * n2 / 2)
            sign = 1 if family == 1 else -1
            blocks = [v, sign * phi_n0 * v, ap1 * bn1 * v, sign * ap2 * bn2 * v]
        return np.concatenate(blocks) / math.sqrt(a.P)

    def _w_at(self, pmi: Type1MPPMI, t: int) -> np.ndarray:
        state = (int(pmi.i2[t]),) if self.mode == 1 else tuple(int(x) for x in pmi.i2[t])
        k1, k2 = self._offset(pmi.rank, pmi.i13)
        beam1 = (pmi.i11, pmi.i12)
        beam2 = (pmi.i11 + k1, pmi.i12 + k2)
        b11 = self._base_vector(*beam1, pmi.i14, state, family=1)
        if pmi.rank == 1:
            columns = [b11]
        else:
            b12 = self._base_vector(*beam2, pmi.i14, state, family=1)
            b21 = self._base_vector(*beam1, pmi.i14, state, family=2)
            if pmi.rank == 2:
                columns = [b11, self._base_vector(*beam2, pmi.i14, state, family=2)]
            elif pmi.rank == 3:
                columns = [b11, b12, b21]
            else:
                columns = [
                    b11,
                    b12,
                    b21,
                    self._base_vector(*beam2, pmi.i14, state, family=2),
                ]
        return np.stack(columns, axis=1) / math.sqrt(pmi.rank)

    def precoder(self, pmi: Type1MPPMI) -> np.ndarray:
        from .validate import validate_type1_multipanel

        validate_type1_multipanel(self, pmi)
        W = np.empty((1, self.N3, self.antenna.P, pmi.rank), dtype=complex)
        for t in range(self.N3):
            W[0, t] = self._w_at(pmi, t)
        return W

    def _candidate_allowed(self, rank: int, i11: int, i12: int, i13: int) -> bool:
        k1, k2 = self._offset(rank, i13 if rank > 1 else None)
        return self._beam_allowed(i11, i12) and (
            rank == 1 or self._beam_allowed(i11 + k1, i12 + k2)
        )

    def _base_vector_batch(
        self,
        v: np.ndarray,
        i14_arr: np.ndarray,
        state_arr: np.ndarray,
        family: int,
    ) -> np.ndarray:
        """Vectorized ``_base_vector`` over every (candidate, state) pair at once.

        ``v`` is (C, N1*N2) -- one cached beam per candidate; ``i14_arr`` is
        (C, len(i14)); ``state_arr`` is (S,) for mode 1 or (S, 3) for mode 2.
        Returns (C, S, P). Mirrors ``_base_vector`` exactly (same per-block
        phase formulas and the same post-concatenation ``/sqrt(P)``), just
        broadcast over the whole candidate/state grid instead of called once
        per pair -- this is what used to make ``select()`` recompute the same
        DFT steering vectors hundreds of thousands of times.
        """
        a = self.antenna
        sign = 1 if family == 1 else -1
        C, N = v.shape
        S = state_arr.shape[0]
        if self.mode == 1:
            phi_n = np.exp(1j * np.pi * state_arr / 2)  # (S,)
            phases = np.exp(1j * np.pi * i14_arr / 2)  # (C, Ng-1)
            panel_phases = np.concatenate(
                [np.ones((C, 1), dtype=complex), phases], axis=1
            )  # (C, Ng)
            blocks = []
            for p in range(a.Ng):
                base_p = panel_phases[:, p][:, None] * v  # (C, N)
                blocks.append(np.broadcast_to(base_p[:, None, :], (C, S, N)))
                blocks.append(sign * phi_n[None, :, None] * base_p[:, None, :])
        else:
            n0, n1, n2 = state_arr[:, 0], state_arr[:, 1], state_arr[:, 2]
            phi_n0 = np.exp(1j * np.pi * n0 / 2)  # (S,)
            bn1 = np.exp(-1j * np.pi / 4) * np.exp(1j * np.pi * n1 / 2)  # (S,)
            bn2 = np.exp(-1j * np.pi / 4) * np.exp(1j * np.pi * n2 / 2)  # (S,)
            ap1 = np.exp(1j * np.pi / 4) * np.exp(1j * np.pi * i14_arr[:, 0] / 2)  # (C,)
            ap2 = np.exp(1j * np.pi / 4) * np.exp(1j * np.pi * i14_arr[:, 1] / 2)  # (C,)
            blocks = [
                np.broadcast_to(v[:, None, :], (C, S, N)),
                sign * phi_n0[None, :, None] * v[:, None, :],
                ap1[:, None, None] * bn1[None, :, None] * v[:, None, :],
                sign * ap2[:, None, None] * bn2[None, :, None] * v[:, None, :],
            ]
        return np.concatenate(blocks, axis=-1) / math.sqrt(a.P)

    def _build_wc(
        self,
        rank: int,
        candidates: list[tuple[int, int, int, tuple[int, ...]]],
        states: list[tuple[int, ...]],
    ) -> np.ndarray:
        """Every candidate's precoder, for every reportable state: (C, S, P, rank).

        Depends only on ``rank`` (and the fixed antenna/mode/restrictions),
        never on the channel -- see ``_search_tensors``, which caches this
        per rank so it is built once rather than once per Monte-Carlo drop.
        """
        i11_arr = np.array([c[0] for c in candidates], dtype=int)
        i12_arr = np.array([c[1] for c in candidates], dtype=int)
        i14_arr = np.array([c[3] for c in candidates], dtype=int)
        state_arr = np.array(
            [s[0] for s in states] if self.mode == 1 else states, dtype=int
        )

        v1 = self._beam_vector(i11_arr, i12_arr)
        b11 = self._base_vector_batch(v1, i14_arr, state_arr, family=1)
        if rank == 1:
            Wc = b11[..., None]
        else:
            i13_arr = np.array([c[2] for c in candidates], dtype=int)
            n_i13 = self._n_i13(rank)
            k1_table = np.empty(n_i13, dtype=int)
            k2_table = np.empty(n_i13, dtype=int)
            for i13 in range(n_i13):
                k1_table[i13], k2_table[i13] = self._offset(rank, i13)
            v2 = self._beam_vector(
                i11_arr + k1_table[i13_arr], i12_arr + k2_table[i13_arr]
            )
            b22 = self._base_vector_batch(v2, i14_arr, state_arr, family=2)
            if rank == 2:
                Wc = np.stack([b11, b22], axis=-1)
            else:
                b12 = self._base_vector_batch(v2, i14_arr, state_arr, family=1)
                b21 = self._base_vector_batch(v1, i14_arr, state_arr, family=2)
                columns = (
                    [b11, b12, b21] if rank == 3 else [b11, b12, b21, b22]
                )
                Wc = np.stack(columns, axis=-1)
        return Wc / math.sqrt(rank)

    def _search_tensors(
        self, rank: int
    ) -> tuple[list[tuple[int, int, int, tuple[int, ...]]], list[tuple[int, ...]], np.ndarray]:
        """(candidates, states, Wc) for this rank, cached on the instance.

        None of these depend on the channel, so a fixed codebook instance
        evaluated over many Monte-Carlo drops (the common ``evaluate()``
        usage) reuses the same tensors instead of rebuilding the whole
        candidate/state grid and precoder tensor on every ``select()`` call.
        """
        cached = self._search_cache.get(rank)
        if cached is not None:
            return cached
        G1, G2 = self.antenna.n_beams
        i14_values = list(itertools.product(range(4), repeat=self._i14_shape()[0]))
        candidates = [
            (i11, i12, i13, i14)
            for i11 in range(G1)
            for i12 in range(G2)
            for i13 in range(self._n_i13(rank))
            for i14 in i14_values
            if self._candidate_allowed(rank, i11, i12, i13)
        ]
        if not candidates:
            raise RuntimeError("no feasible multi-panel PMI under the configured restriction")
        states = self._i2_states(rank)
        Wc = self._build_wc(rank, candidates, states)
        cached = (candidates, states, Wc)
        self._search_cache[rank] = cached
        return cached

    def _candidate_rate(self, Ht: np.ndarray, rank: int, Wc: np.ndarray) -> np.ndarray:
        HW = np.einsum("tnp,ckpv->cktnv", Ht, Wc, optimize=True)
        gram = np.einsum("...nv,...nw->...vw", HW.conj(), HW, optimize=True)
        _, logdet = np.linalg.slogdet(np.eye(rank) + self.selection_rho * gram)
        return np.real(logdet) / math.log(2)

    def select(self, H: np.ndarray, rank: int = 1) -> Type1MPPMI:
        if not 1 <= rank <= 4:
            raise ValueError("Type I multi-panel supports ranks 1-4")
        if not self.rank_restriction[rank - 1]:
            raise ValueError(f"rank {rank} prohibited by multi-panel RI restriction")
        H = np.asarray(H)
        if H.ndim != 4:
            raise ValueError("H must be [slot, t, rx, port]")
        Ht = H[-1]
        if Ht.shape[0] != self.N3:
            raise ValueError(f"channel has {Ht.shape[0]} frequency units, expected {self.N3}")
        if Ht.shape[-1] != self.antenna.P:
            raise ValueError(f"channel has {Ht.shape[-1]} ports, expected {self.antenna.P}")

        candidates, states, Wc = self._search_tensors(rank)
        rates = self._candidate_rate(Ht, rank, Wc)
        metrics = rates.max(axis=1).sum(axis=-1)
        ci = int(np.argmax(metrics))
        i11, i12, i13, i14 = candidates[ci]
        best_states = rates[ci].argmax(axis=0)
        selected = [states[int(si)] for si in best_states]
        i2 = (
            np.array([state[0] for state in selected], dtype=int)
            if self.mode == 1
            else np.asarray(selected, dtype=int)
        )
        return Type1MPPMI(
            rank,
            self.mode,
            i11,
            i12,
            i14,
            i2,
            i13 if rank > 1 else None,
        )

    def overhead_bits(self, pmi: Type1MPPMI) -> dict[str, int]:
        G1, G2 = self.antenna.n_beams
        bits = {
            "i11": math.ceil(math.log2(G1)),
            "i14": 2 * len(pmi.i14),
        }
        if self.antenna.N2 > 1:
            bits["i12"] = math.ceil(math.log2(G2))
        if pmi.rank > 1:
            bits["i13"] = math.ceil(math.log2(self._n_i13(pmi.rank)))
        if self.mode == 1:
            bits["i2"] = self.N3 * (2 if pmi.rank == 1 else 1)
        else:
            bits["i2"] = self.N3 * (4 if pmi.rank == 1 else 3)
        return bits
