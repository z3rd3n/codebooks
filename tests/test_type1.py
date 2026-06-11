"""R15 Type I codebook anchors (paper Tables tabmode1, tabmode2, tabmap)."""

import numpy as np
import pytest

from nr_csi.baselines import eigen_precoder
from nr_csi.channel import Ray, SyntheticRayChannel
from nr_csi.codebooks.type1 import Type1Codebook, Type1PMI, i13_offsets
from nr_csi.config import AntennaConfig
from nr_csi.metrics import sgcs, su_rate
from nr_csi.utils import dft

CFG = AntennaConfig.standard(4, 2)  # P = 16, (O1,O2) = (4,4)


def dual_beam(cfg, m1, m2, n):
    v = dft.spatial_beam(cfg, m1, m2)
    return np.concatenate([v, np.exp(1j * np.pi * n / 2) * v])


class TestReconstruction:
    def test_rank1_norm_and_structure(self):
        cb = Type1Codebook(CFG, N3=1)
        pmi = Type1PMI(rank=1, mode=1, i11=5, i12=3, i2=np.array([2]))
        W = cb.precoder(pmi)
        assert W.shape == (1, 1, CFG.P, 1)
        assert np.isclose(np.linalg.norm(W[0, 0]), 1.0)  # 1/sqrt(P) scaling
        expected = dual_beam(CFG, 5, 3, 2) / np.sqrt(CFG.P)
        assert np.allclose(W[0, 0, :, 0], expected)

    def test_rank2_norm_and_offset_table(self):
        cb = Type1Codebook(CFG, N3=1)
        for i13, (k1, k2) in enumerate(i13_offsets(CFG.N1, CFG.N2, CFG.O1, CFG.O2)):
            pmi = Type1PMI(rank=2, mode=1, i11=2, i12=1, i2=np.array([1]), i13=i13)
            W = cb.precoder(pmi)[0, 0]
            assert np.isclose(np.linalg.norm(W), 1.0)  # Frobenius: total power 1
            v1 = dft.spatial_beam(CFG, 2, 1)
            v2 = dft.spatial_beam(CFG, 2 + k1, 1 + k2)
            phi = np.exp(1j * np.pi / 2)
            expected = np.stack(
                [np.concatenate([v1, phi * v1]), np.concatenate([v2, -phi * v2])], axis=1
            ) / np.sqrt(2 * CFG.P)
            assert np.allclose(W, expected)

    def test_i13_offset_regimes(self):
        """Table tabmap: the three paper regimes + the spec's N1=2,N2=1 case."""
        assert i13_offsets(4, 2, 4, 4) == [(0, 0), (4, 0), (0, 4), (8, 0)]  # N1>N2>1
        assert i13_offsets(4, 4, 4, 4) == [(0, 0), (4, 0), (0, 4), (4, 4)]  # N1=N2
        assert i13_offsets(8, 1, 4, 1) == [(0, 0), (4, 0), (8, 0), (12, 0)]  # N1>2, N2=1
        assert i13_offsets(2, 1, 4, 1) == [(0, 0), (4, 0)]  # N1=2, N2=1

    def test_mode2_beam_and_phase_mapping_rank2(self):
        """Paper Table tabmode2: i2 even/odd -> phi index, i2//2 -> beam offset."""
        cb = Type1Codebook(CFG, N3=1, mode=2)
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for i2 in range(8):
            pmi = Type1PMI(rank=2, mode=2, i11=1, i12=1, i2=np.array([i2]), i13=0)
            l, m, n = cb._beam_and_phase(pmi, 0)
            assert n == i2 % 2
            assert (l, m) == (2 * 1 + offsets[i2 // 2][0], 2 * 1 + offsets[i2 // 2][1])

    def test_mode2_rank1_sixteen_states(self):
        cb = Type1Codebook(CFG, N3=1, mode=2)
        seen = set()
        for i2 in range(16):
            pmi = Type1PMI(rank=1, mode=2, i11=0, i12=0, i2=np.array([i2]))
            seen.add(cb._beam_and_phase(pmi, 0))
        assert len(seen) == 16  # 4 beams x 4 co-phases, all distinct

    def test_mode2_requires_n2_gt_1(self):
        with pytest.raises(ValueError):
            Type1Codebook(AntennaConfig.standard(8, 1), mode=2)


class TestSelection:
    def test_recovers_on_grid_ray_exactly(self):
        ray = Ray(gain=1.0, m1=9, m2=6, pol_phase=np.pi / 2)  # phi index n = 1
        H = SyntheticRayChannel(CFG, [ray], N3=1).generate()
        cb = Type1Codebook(CFG, N3=1)
        pmi = cb.select(H, rank=1)
        assert (pmi.i11, pmi.i12) == (9, 6)
        assert pmi.i2[0] == 1
        # matched on-grid beam => Type I achieves the eigen upper bound
        W = cb.precoder(pmi)[0]
        W_ub = eigen_precoder(H[0], rank=1)
        assert sgcs(W_ub, W) > 1 - 1e-12
        assert np.isclose(su_rate(H[0], W, 10.0), su_rate(H[0], W_ub, 10.0))

    def test_per_subband_co_phasing(self):
        """Mode 1 keeps one wideband beam but adapts i2 per subband."""
        v = dft.spatial_beam(CFG, 4, 2)
        H = np.zeros((1, 2, 1, CFG.P), dtype=complex)
        for t, n in enumerate([0, 3]):  # different pol co-phase per subband
            vd = np.concatenate([v, np.exp(1j * np.pi * n / 2) * v])
            H[0, t, 0] = vd.conj()
        cb = Type1Codebook(CFG, N3=2)
        pmi = cb.select(H, rank=1)
        assert (pmi.i11, pmi.i12) == (4, 2)
        assert pmi.i2.tolist() == [0, 3]

    def test_beam_restriction_is_honored(self):
        ray = Ray(gain=1.0, m1=9, m2=6)
        H = SyntheticRayChannel(CFG, [ray], N3=1).generate()
        G1, G2 = CFG.n_beams
        bitmap = np.ones(G1 * G2, dtype=bool)
        bitmap[6 + G2 * 9] = False  # forbid the true best beam v_{9,6}
        cb = Type1Codebook(CFG, N3=1, beam_restriction=bitmap)
        pmi = cb.select(H, rank=1)
        assert (pmi.i11, pmi.i12) != (9, 6)

    def test_rank2_two_ray_channel(self):
        """Two rays on i13-compatible beams with orthogonal rx signatures.

        The rank-2 Type I structure applies co-phase +phi to layer 1 and -phi
        to layer 2 (Table tabmode1), so the second ray must arrive with the
        opposite polarization phase to be exactly representable.
        """
        rays = [
            Ray(gain=1.0, m1=4, m2=2, pol_phase=0.0, a_rx=np.array([1.0, 0.0])),
            Ray(gain=1.0, m1=4 + CFG.O1, m2=2, pol_phase=np.pi, a_rx=np.array([0.0, 1.0])),
        ]
        H = SyntheticRayChannel(CFG, rays, N3=1, n_rx=2).generate()
        cb = Type1Codebook(CFG, N3=1)
        pmi = cb.select(H, rank=2)
        assert (pmi.i11, pmi.i12) == (4, 2)
        assert pmi.i13 == 1  # second beam at horizontal offset k1 = O1
        W = cb.precoder(pmi)[0]
        W_ub = eigen_precoder(H[0], rank=2)
        # both layers exactly matched => codebook achieves the eigen bound
        r_cb = su_rate(H[0], W, 10.0)
        r_ub = su_rate(H[0], W_ub, 10.0)
        assert r_cb <= r_ub + 1e-9
        assert np.isclose(r_cb, r_ub, rtol=1e-9)

    def test_mode2_selection_recovers_group_member(self):
        ray = Ray(gain=1.0, m1=2 * 3 + 1, m2=2 * 2, pol_phase=np.pi)  # n=2
        H = SyntheticRayChannel(CFG, [ray], N3=1).generate()
        cb = Type1Codebook(CFG, N3=1, mode=2)
        pmi = cb.select(H, rank=1)
        l, m, n = cb._beam_and_phase(pmi, 0)
        assert (l, m, n) == (7, 4, 2)


class TestOverhead:
    def test_mode1_rank1_bits(self):
        cb = Type1Codebook(CFG, N3=4)
        pmi = Type1PMI(rank=1, mode=1, i11=0, i12=0, i2=np.zeros(4, dtype=int))
        bits = cb.overhead_bits(pmi)
        assert bits["i11"] == 4  # log2(N1*O1) = log2(16)
        assert bits["i12"] == 3  # log2(N2*O2) = log2(8)
        assert bits["i2"] == 4 * 2  # 2 bits per subband
        assert "i13" not in bits

    def test_mode1_rank2_bits(self):
        cb = Type1Codebook(CFG, N3=4)
        pmi = Type1PMI(rank=2, mode=1, i11=0, i12=0, i2=np.zeros(4, dtype=int), i13=0)
        bits = cb.overhead_bits(pmi)
        assert bits["i13"] == 2
        assert bits["i2"] == 4 * 1  # 1 bit per subband
