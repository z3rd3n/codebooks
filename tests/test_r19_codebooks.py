"""Tests for the Release-19 codebooks and the R16 port-selection table fix.

Covers, against TS 38.214 v19.3.0 (j30):

* the R16 eType II Port-Selection param table fix (Table 5.2.2.2.6-1);
* the large-array geometries Table 5.2.2.2.1a-1 (48/64/128 ports);
* the three refined Type II codebooks 5.2.2.2.5a / .9a / .11a (config guards +
  reconstruction reuse of R16/R17/R18);
* the Refined Type I single-panel codebook 5.2.2.2.1a, codebookMode 'modeA'
  (Tables 5.2.2.2.1a-2/3/4): orthonormality, the i_{1,3} offset table, the
  readable W^(v) reconstruction, and the companion-beam orthogonality.
"""

from __future__ import annotations

import numpy as np
import pytest

from nr_csi.codebooks.etype2_r16 import R16Type2Codebook
from nr_csi.codebooks.refined_r19 import (
    RefinedEType2Codebook,
    RefinedFeType2PortSelectionCodebook,
    RefinedPredictedEType2Codebook,
)
from nr_csi.codebooks.refined_type1_r19 import (
    RefinedType1PMI,
    RefinedType1SinglePanelCodebook,
    i13_highrank,
    i13_lowrank,
)
from nr_csi.config import (
    R16_PARAM_COMBOS,
    R16_PS_PARAM_COMBOS,
    SUPPORTED_N1N2_R19,
    AntennaConfig,
)
from nr_csi.utils import dft


def _chan(rng, N4, N3, nr, P):
    z = rng.standard_normal((N4, N3, nr, P)) + 1j * rng.standard_normal((N4, N3, nr, P))
    return z / np.sqrt(2)


# ---------------------------------------------------------------------------
# R16 eType II Port-Selection param table -- Table 5.2.2.2.6-1
# ---------------------------------------------------------------------------


def test_table_5_2_2_2_6_1_ps_param_table():
    """The PS table has exactly rows 1..6 of the regular paramCombination-r16
    table; the L=6 rows (7, 8) are absent."""
    assert sorted(R16_PS_PARAM_COMBOS) == [1, 2, 3, 4, 5, 6]
    for i in range(1, 7):
        assert R16_PS_PARAM_COMBOS[i] is R16_PARAM_COMBOS[i]
    assert all(c.L <= 4 for c in R16_PS_PARAM_COMBOS.values())


class TestR16PortSelectionGuards:
    ant = AntennaConfig.standard(4, 2)

    @pytest.mark.parametrize("idx", [1, 2, 3, 4, 5, 6])
    def test_ps_accepts_1_to_6(self, idx):
        cb = R16Type2Codebook(self.ant, N3=12, param_combination=idx, port_selection=True)
        assert cb.L == R16_PS_PARAM_COMBOS[idx].L

    @pytest.mark.parametrize("idx", [7, 8])
    def test_ps_rejects_L6_combos(self, idx):
        with pytest.raises(ValueError, match="port-selection"):
            R16Type2Codebook(self.ant, N3=12, param_combination=idx, port_selection=True)

    @pytest.mark.parametrize("idx", [7, 8])
    def test_regular_still_accepts_L6_combos(self, idx):
        cb = R16Type2Codebook(self.ant, N3=12, param_combination=idx, port_selection=False)
        assert cb.L == 6

    def test_ps_enforces_d_le_L(self):
        # combo 1 -> L = 2, so d = 3 (<= 4 but > L) must be rejected.
        with pytest.raises(ValueError, match="d <= L"):
            R16Type2Codebook(self.ant, N3=12, param_combination=1, port_selection=True, d=3)
        # d = 2 <= L = 2 is fine.
        R16Type2Codebook(self.ant, N3=12, param_combination=1, port_selection=True, d=2)


# ---------------------------------------------------------------------------
# Release-19 large arrays -- Table 5.2.2.2.1a-1
# ---------------------------------------------------------------------------


def test_table_5_2_2_2_1a_1_large_arrays():
    expected = {
        (8, 3): (4, 4),  # 48
        (6, 4): (4, 4),  # 48
        (16, 2): (4, 4),  # 64
        (8, 4): (4, 4),  # 64
        (16, 4): (4, 4),  # 128
        (8, 8): (4, 4),  # 128
    }
    assert SUPPORTED_N1N2_R19 == expected
    for (n1, n2), (o1, o2) in expected.items():
        assert (o1, o2) == (4, 4)
        ant = AntennaConfig.standard(n1, n2)
        assert ant.P == 2 * n1 * n2
        assert ant.P in (48, 64, 128)


# ---------------------------------------------------------------------------
# 5.2.2.2.5a Refined eType II
# ---------------------------------------------------------------------------


class TestRefinedEType2:
    @pytest.mark.parametrize("n1,n2", sorted(SUPPORTED_N1N2_R19))
    @pytest.mark.parametrize("rank", [1, 2, 3, 4])
    def test_roundtrip(self, n1, n2, rank):
        ant = AntennaConfig.standard(n1, n2)
        cb = RefinedEType2Codebook(ant, N3=12, param_combination=3)
        H = _chan(np.random.default_rng(n1 * 10 + n2 + rank), 1, 12, 4, ant.P)
        W = cb.precoder(cb.select(H, rank=rank))
        assert W.shape == (1, 12, ant.P, rank)
        for t in range(12):
            assert np.isclose(np.real(np.trace(W[0, t].conj().T @ W[0, t])), 1.0)

    def test_rejects_non_r19_array(self):
        with pytest.raises(ValueError, match="Release-19"):
            RefinedEType2Codebook(AntennaConfig.standard(4, 2), N3=12)

    def test_combo78_barred_when_rank_gt2_allowed(self):
        for idx in (7, 8):
            with pytest.raises(ValueError, match="7/8"):
                RefinedEType2Codebook(AntennaConfig.standard(8, 3), N3=12, param_combination=idx)

    def test_combo78_barred_when_R2(self):
        for idx in (7, 8):
            with pytest.raises(ValueError):
                RefinedEType2Codebook(
                    AntennaConfig.standard(8, 3), N3=24, param_combination=idx, R=2,
                    ri_restriction=[1, 1, 0, 0],
                )

    def test_combo78_allowed_when_ranks34_disabled(self):
        cb = RefinedEType2Codebook(
            AntennaConfig.standard(8, 3), N3=12, param_combination=7,
            ri_restriction=[1, 1, 0, 0],
        )
        assert cb.L == 6

    def test_ri_restriction_enforced_in_select(self):
        cb = RefinedEType2Codebook(
            AntennaConfig.standard(8, 3), N3=12, param_combination=3,
            ri_restriction=[1, 0, 1, 1],
        )
        H = _chan(np.random.default_rng(0), 1, 12, 4, 48)
        with pytest.raises(ValueError, match="prohibited"):
            cb.select(H, rank=2)


# ---------------------------------------------------------------------------
# 5.2.2.2.9a Refined feType II Port Selection
# ---------------------------------------------------------------------------


class TestRefinedFeType2PS:
    @pytest.mark.parametrize("n1,n2", [(8, 3), (6, 4), (16, 2), (8, 4)])  # 48, 64 ports
    @pytest.mark.parametrize("rank", [1, 2, 3, 4])
    def test_roundtrip(self, n1, n2, rank):
        ant = AntennaConfig.standard(n1, n2)
        cb = RefinedFeType2PortSelectionCodebook(ant, N3=12, param_combination=7)
        H = _chan(np.random.default_rng(n1 + n2 + rank), 1, 12, 4, ant.P)
        W = cb.precoder(cb.select(H, rank=rank))
        assert W.shape == (1, 12, ant.P, rank)

    def test_rejects_128_ports(self):
        with pytest.raises(ValueError, match="ports"):
            RefinedFeType2PortSelectionCodebook(AntennaConfig.standard(8, 8), N3=12)

    def test_rejects_combo8(self):
        with pytest.raises(ValueError, match="paramCombination-r19=8"):
            RefinedFeType2PortSelectionCodebook(
                AntennaConfig.standard(8, 4), N3=12, param_combination=8
            )


# ---------------------------------------------------------------------------
# 5.2.2.2.11a Refined eType II predicted (Doppler)
# ---------------------------------------------------------------------------


class TestRefinedPredictedEType2:
    @pytest.mark.parametrize("n1,n2", sorted(SUPPORTED_N1N2_R19))
    @pytest.mark.parametrize("rank", [1, 2])
    def test_roundtrip(self, n1, n2, rank):
        ant = AntennaConfig.standard(n1, n2)
        cb = RefinedPredictedEType2Codebook(ant, N3=12, N4=4, param_combination=3)
        H = _chan(np.random.default_rng(n1 + n2 + rank), 4, 12, 4, ant.P)
        W = cb.precoder(cb.select(H, rank=rank))
        assert W.shape == (4, 12, ant.P, rank)
        assert cb.Q == 2

    def test_N4_1_degenerates_to_Q1(self):
        cb = RefinedPredictedEType2Codebook(AntennaConfig.standard(8, 4), N3=12, N4=1)
        assert cb.Q == 1

    def test_combo89_barred(self):
        for idx in (8, 9):
            with pytest.raises(ValueError, match="8/9"):
                RefinedPredictedEType2Codebook(
                    AntennaConfig.standard(8, 8), N3=12, param_combination=idx
                )

    def test_combo89_allowed_when_ranks34_disabled(self):
        cb = RefinedPredictedEType2Codebook(
            AntennaConfig.standard(8, 8), N3=12, param_combination=8,
            ri_restriction=[1, 1, 0, 0],
        )
        assert cb.L == 6


# ---------------------------------------------------------------------------
# 5.2.2.2.1a Refined Type I Single-Panel, codebookMode 'modeA'
# ---------------------------------------------------------------------------


def test_table_5_2_2_2_1a_3_i13_offsets():
    """TS 38.214 Table 5.2.2.2.1a-3 (j30 lines 6495-6545)."""
    O1, O2 = 4, 4
    assert [i13_lowrank(2, i, O1, O2) for i in range(4)] == [
        (0, 0), (O1, 0), (0, O2), (2 * O1, 0),
    ]
    for rank in (3, 4):
        assert [i13_lowrank(rank, i, O1, O2) for i in range(4)] == [
            (O1, 0), (0, O2), (O1, O2), (2 * O1, 0),
        ]
    # high-rank (o1,k1),(o2,k2): k1 = i11 mod O1, k2 = i12 mod O2.
    assert i13_highrank(0, 5, 3, O1, O2) == ((O1, 5 % O1), (1, 0))
    assert i13_highrank(1, 5, 3, O1, O2) == ((1, 0), (O2, 3 % O2))


class TestRefinedType1ModeA:
    @pytest.mark.parametrize("n1,n2", sorted(SUPPORTED_N1N2_R19))
    @pytest.mark.parametrize("rank", range(1, 9))
    def test_precoder_orthonormal(self, n1, n2, rank):
        """W^(v) columns are orthonormal up to the 1/sqrt(v) scaling for every
        rank 1..8 on every Release-19 array (companion beams stay orthogonal)."""
        ant = AntennaConfig.standard(n1, n2)
        cb = RefinedType1SinglePanelCodebook(ant, N3=2)
        H = _chan(np.random.default_rng(n1 * 100 + n2 * 10 + rank), 1, 2, 8, ant.P)
        pmi = cb.select(H, rank=rank)
        W = cb.precoder(pmi)
        assert W.shape == (1, 2, ant.P, rank)
        for t in range(2):
            gram = W[0, t].conj().T @ W[0, t]
            assert np.allclose(gram, np.eye(rank) / rank, atol=1e-9)

    def test_reconstruction_rank2_matches_formula(self):
        """W^(2) = (1/sqrt(2P)) [[v_lm, v_l'm'],[phi v_lm, -phi v_l'm']]."""
        ant = AntennaConfig.standard(8, 4)
        cb = RefinedType1SinglePanelCodebook(ant, N3=1)
        pmi = RefinedType1PMI(2, 3, 2, np.array([1]), i13=1)  # n=1 -> phi=j
        W = cb.precoder(pmi)[0, 0]
        k1, k2 = i13_lowrank(2, 1, ant.O1, ant.O2)
        v0 = dft.spatial_beam(ant, 3, 2)
        v1 = dft.spatial_beam(ant, 3 + k1, 2 + k2)
        phi = np.exp(1j * np.pi / 2)
        scale = np.sqrt(2 * ant.P)
        assert np.allclose(W[:, 0], np.concatenate([v0, phi * v0]) / scale)
        assert np.allclose(W[:, 1], np.concatenate([v1, -phi * v1]) / scale)

    def test_reconstruction_rank5_matches_formula(self):
        """W^(5) columns: (b0;phi b0),(b0;-phi b0),(b1;b1),(b1;-b1),(b2;b2)."""
        ant = AntennaConfig.standard(8, 4)
        cb = RefinedType1SinglePanelCodebook(ant, N3=1)
        pmi = RefinedType1PMI(5, 3, 2, np.array([0]), i13=0, i112=(1, 5), i122=(2, 0))
        beams = cb._beams(pmi)
        W = cb.precoder(pmi)[0, 0]
        vb = [dft.spatial_beam(ant, l, m) for l, m in beams]
        scale = np.sqrt(5 * ant.P)
        phi = 1.0  # n = 0
        assert np.allclose(W[:, 0], np.concatenate([vb[0], phi * vb[0]]) / scale)
        assert np.allclose(W[:, 1], np.concatenate([vb[0], -phi * vb[0]]) / scale)
        assert np.allclose(W[:, 2], np.concatenate([vb[1], vb[1]]) / scale)
        assert np.allclose(W[:, 3], np.concatenate([vb[1], -vb[1]]) / scale)
        assert np.allclose(W[:, 4], np.concatenate([vb[2], vb[2]]) / scale)

    @pytest.mark.parametrize("rank", [5, 6, 7, 8])
    def test_high_rank_companion_beams_orthogonal(self, rank):
        """The selected beams (base + companions) are mutually orthogonal, which
        is what makes the W^(v) columns orthonormal."""
        ant = AntennaConfig.standard(8, 4)
        cb = RefinedType1SinglePanelCodebook(ant, N3=1)
        H = _chan(np.random.default_rng(rank), 1, 1, 8, ant.P)
        pmi = cb.select(H, rank=rank)
        beams = [dft.spatial_beam(ant, l, m) for l, m in cb._beams(pmi)]
        n_beams = {5: 3, 6: 3, 7: 4, 8: 4}[rank]
        assert len(beams) == n_beams
        gram = np.array([[b1.conj() @ b2 for b2 in beams] for b1 in beams])
        assert np.allclose(gram, np.diag(np.diag(gram)), atol=1e-9)  # off-diagonals ~ 0

    def test_rejects_non_r19_array(self):
        with pytest.raises(ValueError, match="Release-19"):
            RefinedType1SinglePanelCodebook(AntennaConfig.standard(4, 2), N3=1)

    def test_validate_rejects_malformed_pmi(self):
        ant = AntennaConfig.standard(8, 4)
        cb = RefinedType1SinglePanelCodebook(ant, N3=1)
        # rank 5 needs 2 companion indices.
        with pytest.raises(ValueError, match="companion"):
            cb.precoder(RefinedType1PMI(5, 0, 0, np.array([0]), i13=0, i112=(1,), i122=(2,)))
        # i13 out of range for rank 2-4.
        with pytest.raises(ValueError, match="i13"):
            cb.precoder(RefinedType1PMI(3, 0, 0, np.array([0]), i13=7))
        # i2 out of range (rank>=2 -> only {0,1}).
        with pytest.raises(ValueError, match="i2"):
            cb.precoder(RefinedType1PMI(2, 0, 0, np.array([2]), i13=0))

    @pytest.mark.parametrize("rank,extra_bits", [(1, 0), (2, 0), (5, 1)])
    def test_overhead_has_expected_fields(self, rank, extra_bits):
        ant = AntennaConfig.standard(8, 4)  # G1=32, G2=16
        cb = RefinedType1SinglePanelCodebook(ant, N3=4)
        H = _chan(np.random.default_rng(rank), 1, 4, 8, ant.P)
        pmi = cb.select(H, rank=rank)
        bits = cb.overhead_bits(pmi)
        assert bits["i11"] == 5  # ceil(log2(32))
        assert bits["i12"] == 4  # ceil(log2(16))
        assert bits["i2"] == 4 * (2 if rank == 1 else 1)
        if rank >= 5:
            assert "i112" in bits  # companion-beam indices reported
