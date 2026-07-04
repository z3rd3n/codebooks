"""2-port Type I codebook, TS 38.214 Table 5.2.2.2.1-1.

Covers the exact table entries, the twoTX-CodebookSubsetRestriction bit
mapping (bits 0-3 -> rank-1 indices, bits 4-5 -> rank-2 indices), the rank
restriction, selection behaviour on matched channels, and the serializer
round trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from nr_csi.codebooks import TwoPortType1Codebook, pack, unpack
from nr_csi.codebooks.type1 import TwoPortType1PMI


def _chan_for(w: np.ndarray, N3: int = 1) -> np.ndarray:
    """Rank-1 channel [1, N3, 1, 2] whose matched precoder is ``w``."""
    h = np.conj(w[:, 0])
    return np.tile(h, (1, N3, 1, 1))


RANK1_TABLE = [
    np.array([[1], [1]]) / np.sqrt(2),
    np.array([[1], [1j]]) / np.sqrt(2),
    np.array([[1], [-1]]) / np.sqrt(2),
    np.array([[1], [-1j]]) / np.sqrt(2),
]
RANK2_TABLE = [
    np.array([[1, 1], [1, -1]]) / 2,
    np.array([[1, 1], [1j, -1j]]) / 2,
]


class TestTable522211:
    @pytest.mark.parametrize("idx", range(4))
    def test_rank1_entries(self, idx):
        cbk = TwoPortType1Codebook(N3=1)
        W = cbk.precoder(TwoPortType1PMI(1, np.array([idx])))
        assert np.allclose(W[0, 0], RANK1_TABLE[idx])

    @pytest.mark.parametrize("idx", range(2))
    def test_rank2_entries(self, idx):
        cbk = TwoPortType1Codebook(N3=1)
        W = cbk.precoder(TwoPortType1PMI(2, np.array([idx])))
        assert np.allclose(W[0, 0], RANK2_TABLE[idx])

    def test_column_norms(self):
        cbk = TwoPortType1Codebook(N3=1)
        for rank, n in ((1, 4), (2, 2)):
            for idx in range(n):
                W = cbk.precoder(TwoPortType1PMI(rank, np.array([idx])))
                assert np.allclose(np.linalg.norm(W, axis=-2), 1 / np.sqrt(rank))

    def test_out_of_range_index_rejected(self):
        cbk = TwoPortType1Codebook(N3=1)
        with pytest.raises(ValueError):
            cbk.precoder(TwoPortType1PMI(1, np.array([4])))
        with pytest.raises(ValueError):
            cbk.precoder(TwoPortType1PMI(2, np.array([2])))
        with pytest.raises(ValueError):
            cbk.precoder(TwoPortType1PMI(3, np.array([0])))


class TestSelection:
    @pytest.mark.parametrize("idx", range(4))
    def test_matched_channel_recovers_index(self, idx):
        cbk = TwoPortType1Codebook(N3=1)
        pmi = cbk.select(_chan_for(RANK1_TABLE[idx]), rank=1)
        assert pmi.i2[0] == idx

    def test_per_subband_indices(self):
        cbk = TwoPortType1Codebook(N3=2)
        H = np.concatenate(
            [_chan_for(RANK1_TABLE[0]), _chan_for(RANK1_TABLE[2])], axis=1
        )
        pmi = cbk.select(H, rank=1)
        assert list(pmi.i2) == [0, 2]

    def test_rank2_identity_channel(self):
        rng = np.random.default_rng(0)
        cbk = TwoPortType1Codebook(N3=4)
        H = rng.standard_normal((1, 4, 2, 2)) + 1j * rng.standard_normal((1, 4, 2, 2))
        pmi = cbk.select(H, rank=2)
        W = cbk.precoder(pmi)
        assert W.shape == (1, 4, 2, 2)


class TestRestrictions:
    def test_twotx_bitmap_masks_rank1(self):
        # prohibit index 2; a channel matched to [1; -1] must pick elsewhere
        restriction = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
        cbk = TwoPortType1Codebook(N3=1, restriction=restriction)
        pmi = cbk.select(_chan_for(RANK1_TABLE[2]), rank=1)
        assert pmi.i2[0] != 2

    def test_twotx_bitmap_masks_rank2(self):
        restriction = np.array([1, 1, 1, 1, 0, 1], dtype=bool)
        cbk = TwoPortType1Codebook(N3=1, restriction=restriction)
        rng = np.random.default_rng(1)
        H = rng.standard_normal((1, 1, 2, 2)) + 1j * rng.standard_normal((1, 1, 2, 2))
        pmi = cbk.select(H, rank=2)
        assert pmi.i2[0] == 1  # only rank-2 index 1 remains

    def test_all_prohibited_raises(self):
        restriction = np.array([0, 0, 0, 0, 1, 1], dtype=bool)
        cbk = TwoPortType1Codebook(N3=1, restriction=restriction)
        with pytest.raises(RuntimeError, match="twoTX"):
            cbk.select(_chan_for(RANK1_TABLE[0]), rank=1)

    def test_rank_restriction(self):
        cbk = TwoPortType1Codebook(N3=1, rank_restriction=[1, 0])
        with pytest.raises(ValueError, match="ri-Restriction"):
            cbk.select(_chan_for(RANK1_TABLE[0]), rank=2)

    def test_bitmap_length_validated(self):
        with pytest.raises(ValueError, match="6 bits"):
            TwoPortType1Codebook(N3=1, restriction=np.ones(4, dtype=bool))
        with pytest.raises(ValueError, match="2 bits"):
            TwoPortType1Codebook(N3=1, rank_restriction=np.ones(8, dtype=bool))


class TestOverheadAndSerialization:
    def test_overhead_bits(self):
        cbk = TwoPortType1Codebook(N3=5)
        assert cbk.overhead_bits(TwoPortType1PMI(1, np.zeros(5, dtype=int))) == {
            "i2": 10
        }
        assert cbk.overhead_bits(TwoPortType1PMI(2, np.zeros(5, dtype=int))) == {
            "i2": 5
        }

    @pytest.mark.parametrize("rank", [1, 2])
    def test_pack_unpack_round_trip(self, rank):
        rng = np.random.default_rng(2)
        cbk = TwoPortType1Codebook(N3=6)
        H = rng.standard_normal((1, 6, 2, 2)) + 1j * rng.standard_normal((1, 6, 2, 2))
        pmi = cbk.select(H, rank=rank)
        bits = pack(cbk, pmi)
        assert len(bits) == cbk.total_overhead_bits(pmi)
        pmi2 = unpack(cbk, bits, rank)
        assert pmi2.rank == rank and np.array_equal(pmi.i2, pmi2.i2)
