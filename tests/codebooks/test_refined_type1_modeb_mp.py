"""Release-19 Refined Type I: codebookMode 'modeB' (5.2.2.2.1a) and the
multi-panel codebook (5.2.2.2.2a).

Covers, against TS 38.214 v19.4.0:

* modeB reverse beam indexing n^(l) = N1*N2 - 1 - i_{1,2,l} and the
  single-beam column pattern of Table 5.2.2.2.1a-7;
* the Table 5.2.2.2.1a-6 (c, c+2) co-phasing pairs for ranks 5-8 and the
  resulting orthonormality;
* modeB index-range validation and overhead accounting;
* the multi-panel geometries of Table 5.2.2.2.2a-1, the per-panel beams and
  i_{1,3,j} offsets (Table 5.2.2.2.2a-2), the W^{1,Ng}/W^{2,Ng} block
  structure, and the cbsr/RI restrictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from nr_csi.codebooks.refined_type1_r19 import (
    RefinedType1ModeBPMI,
    RefinedType1SinglePanelCodebook,
    i13_lowrank,
    modeb_layer_fields,
)
from nr_csi.codebooks.refined_type1mp_r19 import (
    RefinedType1MPPMI,
    RefinedType1MultiPanelCodebook,
)
from nr_csi.config import SUPPORTED_NG_N1N2_R19, AntennaConfig
from nr_csi.utils import dft

ANT48 = AntennaConfig.standard(8, 3)  # 48 ports


def _chan(rng, N3, nr, P):
    z = rng.standard_normal((1, N3, nr, P)) + 1j * rng.standard_normal((1, N3, nr, P))
    return z / np.sqrt(2)


# ---------------------------------------------------------------------------
# modeB
# ---------------------------------------------------------------------------


class TestModeBReconstruction:
    def test_rank1_column_matches_spec(self):
        cbk = RefinedType1SinglePanelCodebook(ANT48, N3=1, mode="modeB")
        a = ANT48
        i12l, q1, q2, c = 5, 2, 1, 3
        pmi = RefinedType1ModeBPMI(
            1, q1, q2, i12l=(i12l,), i2=np.array([[c]])
        )
        W = cbk.precoder(pmi)
        n = a.N1 * a.N2 - 1 - i12l  # reverse indexing of 5.2.2.2.1a
        n1, n2 = n % a.N1, n // a.N1
        v = dft.spatial_beam(a, a.O1 * n1 + q1, a.O2 * n2 + q2)
        phi = np.exp(1j * np.pi * c / 2)
        expected = np.concatenate([v, phi * v]) / np.sqrt(a.P)
        assert np.allclose(W[0, 0, :, 0], expected)

    def test_rank3_per_layer_beams(self):
        cbk = RefinedType1SinglePanelCodebook(ANT48, N3=1, mode="modeB")
        a = ANT48
        pmi = RefinedType1ModeBPMI(
            3, 0, 0, i12l=(0, 7, 11), i2=np.array([[0, 1, 2]])
        )
        W = cbk.precoder(pmi)
        for li, (i12l, c) in enumerate(zip(pmi.i12l, (0, 1, 2))):
            n = a.N1 * a.N2 - 1 - i12l
            v = dft.spatial_beam(a, a.O1 * (n % a.N1), a.O2 * (n // a.N1))
            phi = np.exp(1j * np.pi * c / 2)
            expected = np.concatenate([v, phi * v]) / np.sqrt(3 * a.P)
            assert np.allclose(W[0, 0, :, li], expected)

    @pytest.mark.parametrize("rank", [5, 6, 7, 8])
    def test_highrank_pairs_are_orthogonal(self, rank):
        rng = np.random.default_rng(rank)
        cbk = RefinedType1SinglePanelCodebook(ANT48, N3=2, mode="modeB")
        H = _chan(rng, 2, 8, ANT48.P)
        pmi = cbk.select(H, rank=rank)
        W = cbk.precoder(pmi)
        for t in range(2):
            gram = W[0, t].conj().T @ W[0, t]
            assert np.allclose(gram, np.eye(rank) / rank, atol=1e-12)

    @pytest.mark.parametrize("rank", [5, 6, 7, 8])
    def test_layer_fields_cover_all_layers(self, rank):
        fields = modeb_layer_fields(rank)
        assert len(fields) == rank
        # paired layers carry (c, c+2): offsets 0 and 2 on the same field
        by_field: dict[int, list] = {}
        for g, fi, off in fields:
            by_field.setdefault(fi, []).append(off)
        for offs in by_field.values():
            assert offs in ([0, 2], [None])

    def test_pair_cophasing_is_c_and_c_plus_2(self):
        """v=6, i_{2,g}=1 -> the two layers on beam g use phi_1 and phi_3."""
        cbk = RefinedType1SinglePanelCodebook(ANT48, N3=1, mode="modeB")
        pmi = RefinedType1ModeBPMI(6, 0, 0, i12=0, i2=np.array([[1, 0, 0]]))
        W = cbk.precoder(pmi)
        half = ANT48.P // 2
        # layers 0, 1 share the first beam; bottom-half phases differ by pi
        b_top = W[0, 0, :half, 0]
        assert np.allclose(W[0, 0, :half, 1], b_top)
        ratio0 = W[0, 0, half:, 0] / b_top
        ratio1 = W[0, 0, half:, 1] / b_top
        assert np.allclose(ratio0, np.exp(1j * np.pi / 2))  # phi_1
        assert np.allclose(ratio1, np.exp(1j * np.pi * 3 / 2))  # phi_3


class TestModeBValidationAndOverhead:
    def test_index_ranges(self):
        cbk = RefinedType1SinglePanelCodebook(ANT48, N3=1, mode="modeB")
        with pytest.raises(ValueError, match="i_{1,2,l}"):
            cbk.precoder(
                RefinedType1ModeBPMI(1, 0, 0, i12l=(24,), i2=np.array([[0]]))
            )
        with pytest.raises(ValueError, match="i_{2,l}"):
            cbk.precoder(
                RefinedType1ModeBPMI(1, 0, 0, i12l=(0,), i2=np.array([[4]]))
            )
        # rank 6: the pair fields are 1-bit
        with pytest.raises(ValueError, match=r"i_\{2,1\}"):
            cbk.precoder(
                RefinedType1ModeBPMI(6, 0, 0, i12=0, i2=np.array([[2, 0, 0]]))
            )

    def test_mode_validation(self):
        with pytest.raises(ValueError, match="modeA.*modeB|codebookMode"):
            RefinedType1SinglePanelCodebook(ANT48, N3=1, mode="modeC")

    def test_ri_restriction(self):
        cbk = RefinedType1SinglePanelCodebook(
            ANT48, N3=1, mode="modeB", ri_restriction=[1, 0, 1, 1, 1, 1, 1, 1]
        )
        H = _chan(np.random.default_rng(0), 1, 4, ANT48.P)
        with pytest.raises(ValueError, match="typeI-SinglePanelRI-Restriction-r19"):
            cbk.select(H, rank=2)

    def test_overhead_bits(self):
        cbk = RefinedType1SinglePanelCodebook(ANT48, N3=4, mode="modeB")
        # rank 2: i11 = 4 (q1, q2), i12 = 2 * ceil(log2(24)) = 10, i2 = 4*2*2
        pmi = RefinedType1ModeBPMI(
            2, 0, 0, i12l=(0, 1), i2=np.zeros((4, 2), dtype=int)
        )
        assert cbk.overhead_bits(pmi) == {"i11": 4, "i12": 10, "i2": 16}
        # rank 5: i12 combinatorial C(24,3)=2024 -> 11 bits; i2 = 1+1+2 per t
        pmi5 = RefinedType1ModeBPMI(5, 0, 0, i12=0, i2=np.zeros((4, 3), dtype=int))
        assert cbk.overhead_bits(pmi5) == {"i11": 4, "i12": 11, "i2": 16}

    @pytest.mark.parametrize("rank", range(1, 9))
    def test_select_precoder_round_trip(self, rank):
        rng = np.random.default_rng(100 + rank)
        cbk = RefinedType1SinglePanelCodebook(ANT48, N3=3, mode="modeB")
        H = _chan(rng, 3, 8, ANT48.P)
        pmi = cbk.select(H, rank=rank)
        W = cbk.precoder(pmi)
        assert W.shape == (1, 3, ANT48.P, rank)
        assert np.allclose(np.linalg.norm(W, axis=-2), 1 / np.sqrt(rank))


# ---------------------------------------------------------------------------
# multi-panel (5.2.2.2.2a)
# ---------------------------------------------------------------------------


class TestMultiPanelConfig:
    @pytest.mark.parametrize("cfg", sorted(SUPPORTED_NG_N1N2_R19))
    def test_supported_configs_construct(self, cfg):
        ng, n1, n2 = cfg
        ant = AntennaConfig.standard(n1, n2, Ng=ng)
        assert ant.P == 2 * ng * n1 * n2
        RefinedType1MultiPanelCodebook(ant, N3=1)

    def test_legacy_multipanel_config_rejected(self):
        ant = AntennaConfig.standard(4, 1, Ng=2)  # legacy Table 5.2.2.2.2-1 row
        with pytest.raises(ValueError, match="5.2.2.2.2a-1"):
            RefinedType1MultiPanelCodebook(ant, N3=1)


class TestMultiPanelReconstruction:
    ANT = AntennaConfig.standard(4, 3, Ng=2)  # 48 ports

    def test_rank1_block_structure(self):
        a = self.ANT
        cbk = RefinedType1MultiPanelCodebook(a, N3=1)
        i11, i12, i14, n = (3, 7), (1, 5), (2,), (1, 3)
        pmi = RefinedType1MPPMI(1, i11, i12, i14, np.array([n]))
        W = cbk.precoder(pmi)[0, 0, :, 0]
        npp = a.n_ports_per_pol
        expected = []
        panel_phases = [1.0, np.exp(1j * np.pi * i14[0] / 2)]
        for j in range(2):
            v = dft.spatial_beam(a, i11[j], i12[j])
            phi_n = np.exp(1j * np.pi * n[j] / 2)
            expected.extend([panel_phases[j] * v, phi_n * panel_phases[j] * v])
        expected = np.concatenate(expected) / np.sqrt(a.P)
        assert np.allclose(W, expected)

    def test_rank2_uses_per_panel_i13_offsets(self):
        a = self.ANT
        cbk = RefinedType1MultiPanelCodebook(a, N3=1)
        i13 = (1, 2)  # different companion offset per panel
        pmi = RefinedType1MPPMI(
            2, (0, 4), (0, 2), (0,), np.array([[0, 1]]), i13
        )
        W = cbk.precoder(pmi)
        npp = a.n_ports_per_pol
        # second column (W^{2,Ng} of the companion beams): check panel-2 top block
        k1, k2 = i13_lowrank(2, i13[1], a.O1, a.O2)
        v2p = dft.spatial_beam(a, 4 + k1, 2 + k2)
        block = W[0, 0, 2 * npp : 3 * npp, 1] * np.sqrt(2 * a.P)
        assert np.allclose(block, v2p)  # i14 = 0 -> panel phase 1

    @pytest.mark.parametrize("rank", [1, 2, 3, 4])
    def test_orthonormal_columns(self, rank):
        rng = np.random.default_rng(rank)
        cbk = RefinedType1MultiPanelCodebook(self.ANT, N3=2)
        H = _chan(rng, 2, 4, self.ANT.P)
        pmi = cbk.select(H, rank=rank)
        W = cbk.precoder(pmi)
        for t in range(2):
            gram = W[0, t].conj().T @ W[0, t]
            assert np.allclose(gram, np.eye(rank) / rank, atol=1e-12)

    def test_w2_flips_polarization_sign(self):
        """Rank 2 with i13 = (0, 0): both columns share the beams; the second
        column negates every panel's bottom (second-polarization) block."""
        a = self.ANT
        cbk = RefinedType1MultiPanelCodebook(a, N3=1)
        pmi = RefinedType1MPPMI(2, (1, 2), (3, 4), (1,), np.array([[1, 0]]), (0, 0))
        W = cbk.precoder(pmi)
        npp = a.n_ports_per_pol
        c1, c2 = W[0, 0, :, 0], W[0, 0, :, 1]
        for j in range(a.Ng):
            top = slice(2 * j * npp, (2 * j + 1) * npp)
            bot = slice((2 * j + 1) * npp, (2 * j + 2) * npp)
            assert np.allclose(c1[top], c2[top])
            assert np.allclose(c1[bot], -c2[bot])


class TestMultiPanelRestrictions:
    ANT = AntennaConfig.standard(4, 3, Ng=2)

    def test_beam_restriction_masks_selection(self):
        a = self.ANT
        G1, G2 = a.n_beams
        rng = np.random.default_rng(5)
        H = _chan(rng, 1, 4, a.P)
        free = RefinedType1MultiPanelCodebook(a, N3=1).select(H, rank=1)
        bitmap = np.ones(G1 * G2, dtype=bool)
        for j in range(a.Ng):
            bitmap[free.i12[j] + G2 * free.i11[j]] = False
        pmi = RefinedType1MultiPanelCodebook(
            a, N3=1, beam_restriction=bitmap
        ).select(H, rank=1)
        for j in range(a.Ng):
            assert bitmap[pmi.i12[j] + G2 * pmi.i11[j]]

    def test_all_beams_prohibited_raises(self):
        a = self.ANT
        cbk = RefinedType1MultiPanelCodebook(
            a, N3=1, beam_restriction=np.zeros(np.prod(a.n_beams), dtype=bool)
        )
        H = _chan(np.random.default_rng(6), 1, 2, a.P)
        with pytest.raises(RuntimeError, match="cbsr"):
            cbk.select(H, rank=1)

    def test_ri_restriction(self):
        cbk = RefinedType1MultiPanelCodebook(
            self.ANT, N3=1, ri_restriction=[1, 1, 0, 0]
        )
        H = _chan(np.random.default_rng(7), 1, 4, self.ANT.P)
        with pytest.raises(ValueError, match="ri-Restriction-r19"):
            cbk.select(H, rank=3)

    def test_overhead_bits(self):
        a = self.ANT  # G1 = 16, G2 = 12
        cbk = RefinedType1MultiPanelCodebook(a, N3=2)
        pmi = RefinedType1MPPMI(1, (0, 0), (0, 0), (0,), np.zeros((2, 2), dtype=int))
        bits = cbk.overhead_bits(pmi)
        assert bits == {"i11": 2 * 4, "i12": 2 * 4, "i14": 2, "i2": 2 * 2 * 2}
        pmi2 = RefinedType1MPPMI(
            2, (0, 0), (0, 0), (0,), np.zeros((2, 2), dtype=int), (0, 0)
        )
        bits2 = cbk.overhead_bits(pmi2)
        assert bits2["i13"] == 4 and bits2["i2"] == 2 * 2 * 1
