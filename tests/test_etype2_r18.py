"""R18 Doppler codebook anchors: N4=1 == R16, prediction gain, Tucker model."""

import numpy as np
import pytest

from nr_csi.channel import Ray, SyntheticRayChannel
from nr_csi.codebooks import compact
from nr_csi.codebooks.etype2_r16 import R16Type2Codebook
from nr_csi.codebooks.etype2_r18 import (
    R18Type2Codebook,
    R18Type2PMI,
    decode_i18,
    encode_i18,
)
from nr_csi.config import AntennaConfig
from nr_csi.metrics import sgcs
from nr_csi.utils import combinatorics as cb

CFG = AntennaConfig.standard(4, 2)  # P = 16


def random_valid_pmi(cbk: R18Type2Codebook, rank: int, seed: int = 0) -> R18Type2PMI:
    rng = np.random.default_rng(seed)
    a, L, Q = cbk.antenna, cbk.L, cbk.Q
    Mv = cbk.Mv(rank)
    flat = sorted(rng.choice(a.N1 * a.N2, size=L, replace=False).tolist())
    pmi = R18Type2PMI(
        rank=rank,
        q1=int(rng.integers(a.O1)),
        q2=int(rng.integers(a.O2)),
        i12=cb.combo_to_index(flat, a.N1 * a.N2),
    )
    pmi.i17 = np.zeros((rank, Q, Mv, 2 * L), dtype=bool)
    pmi.k1 = np.ones((rank, 2), dtype=int)
    pmi.k2 = np.zeros((rank, Q, Mv, 2 * L), dtype=int)
    pmi.c = np.zeros((rank, Q, Mv, 2 * L), dtype=int)
    for li in range(rank):
        taps = sorted(
            [0] + rng.choice(np.arange(1, cbk.N3), size=Mv - 1, replace=False).tolist()
        )
        i16, _ = cb.encode_taps(taps, cbk.N3, Mv)
        pmi.i16.append(i16)
        if Q == 2:
            pmi.i110.append(int(rng.integers(cbk.N4 - 1)))
        n_keep = min(cbk.K0, Q * Mv * 2 * L)
        keep = rng.choice(Q * Mv * 2 * L, size=n_keep, replace=False)
        bm = np.zeros(Q * Mv * 2 * L, dtype=bool)
        bm[keep] = True
        bm = bm.reshape(Q, Mv, 2 * L)
        i_star, tau_star = int(rng.integers(2 * L)), int(rng.integers(Q))
        bm[tau_star, 0, i_star] = True
        pmi.i17[li] = bm
        pmi.k1[li] = [15, int(rng.integers(1, 16))]
        if i_star >= L:
            pmi.k1[li] = pmi.k1[li][::-1]
        pmi.k2[li][bm] = rng.integers(0, 8, size=int(bm.sum()))
        pmi.c[li][bm] = rng.integers(0, 16, size=int(bm.sum()))
        pmi.k2[li, tau_star, 0, i_star] = 7
        pmi.c[li, tau_star, 0, i_star] = 0
        pmi.i18.append(encode_i18(bm, i_star, tau_star, rank, L))
    return pmi


class TestN4OneDegeneratesToR16:
    """The paper's strongest structural statement: N4=1 reduces to R16."""

    @pytest.mark.parametrize("combo", [2, 3])
    @pytest.mark.parametrize("rank", [1, 2])
    def test_pmi_and_precoder_match_r16(self, combo, rank):
        rng = np.random.default_rng(combo * 10 + rank)
        N3 = 12
        H = rng.standard_normal((1, N3, 2, CFG.P)) + 1j * rng.standard_normal(
            (1, N3, 2, CFG.P)
        )
        r18 = R18Type2Codebook(CFG, N3=N3, N4=1, param_combination=combo)
        r16 = R16Type2Codebook(CFG, N3=N3, param_combination=combo)
        assert r18.K0 == r16.K0 and r18.Mv(rank) == r16.Mv(rank)
        p18 = r18.select(H, rank=rank)
        p16 = r16.select(H, rank=rank)
        assert (p18.q1, p18.q2, p18.i12) == (p16.q1, p16.q2, p16.i12)
        assert p18.i16 == p16.i16
        assert p18.i18 == p16.i18
        assert np.array_equal(p18.i17[:, 0], p16.i17)
        assert np.array_equal(p18.k1, p16.k1)
        assert np.array_equal(p18.k2[:, 0], p16.k2)
        assert np.array_equal(p18.c[:, 0], p16.c)
        W18 = r18.precoder(p18)
        W16 = r16.precoder(p16)
        assert np.allclose(W18, W16, atol=1e-12)


class TestReconstruction:
    @pytest.mark.parametrize("rank", [1, 2])
    def test_unit_norm_per_interval_and_subband(self, rank):
        cbk = R18Type2Codebook(CFG, N3=8, N4=4, param_combination=3)
        pmi = random_valid_pmi(cbk, rank, seed=rank)
        W = cbk.precoder(pmi)
        assert W.shape == (4, 8, CFG.P, rank)
        for iota in range(4):
            for t in range(8):
                assert np.allclose(np.linalg.norm(W[iota, t], axis=0), 1 / np.sqrt(rank))

    @pytest.mark.parametrize("rank", [1, 2])
    def test_spec_equals_kron_and_tucker_compact(self, rank):
        cbk = R18Type2Codebook(CFG, N3=8, N4=4, param_combination=3)
        for seed in range(4):
            pmi = random_valid_pmi(cbk, rank, seed=seed)
            W_spec = cbk.precoder(pmi)  # (N4, N3, P, v)
            B = cbk._basis(pmi)
            for li in range(rank):
                taps = cb.decode_taps(pmi.i16[li], cbk.N3, cbk.Mv(rank), pmi.i15)
                x = cbk._layer_coefficients(pmi, li)  # (2L, Mv, Q)
                shifts = cbk.shifts(pmi, li)
                W_kron = compact.compact_r18(B, x, taps, shifts, cbk.N3, cbk.N4)
                W_tuck = compact.tucker_r18(B, x, taps, shifts, cbk.N3, cbk.N4)
                assert np.allclose(W_kron, W_tuck, atol=1e-10)
                s = sgcs(W_spec[:, :, :, li][..., None], W_kron[..., None])
                assert s > 1 - 1e-12

    def test_i18_roundtrip_with_shifts(self):
        rng = np.random.default_rng(5)
        L, Q, Mv = 4, 2, 3
        for rank in (1, 2):
            for _ in range(20):
                bm = rng.random((Q, Mv, 2 * L)) < 0.4
                i_star, tau_star = int(rng.integers(2 * L)), int(rng.integers(Q))
                bm[tau_star, 0, i_star] = True
                i18 = encode_i18(bm, i_star, tau_star, rank, L)
                assert decode_i18(i18, bm, rank, L) == (i_star, tau_star)

    def test_k0_includes_doppler_factor(self):
        cbk = R18Type2Codebook(CFG, N3=8, N4=4, param_combination=3)  # L=4, beta=1/4
        # K0 = ceil(2 * beta * L * M1 * Q) = ceil(2 * 0.25 * 4 * 2 * 2)
        assert cbk.K0 == 8
        cbk1 = R18Type2Codebook(CFG, N3=8, N4=1, param_combination=3)
        assert cbk1.K0 == 4  # Q = 1


class TestDopplerPrediction:
    # (8,1) array: an off-group horizontal beam needs all 8 group beams, so
    # the 3-ray on-grid target makes group (0,0) the unique top-4 winner
    # (with N1=4 and L=4, any group ties at 100% for single-m2 targets).
    CFG8 = AntennaConfig.standard(8, 1)

    def _mobile_channel(self, N3, N4):
        rays = [
            Ray(gain=1.0, m1=4, m2=0, doppler=0),
            Ray(gain=0.5, m1=8, m2=0, doppler=1),  # one DFT Doppler shift
            Ray(gain=0.5, m1=12, m2=0, doppler=0),
        ]
        ch = SyntheticRayChannel(self.CFG8, rays, N3=N3, n_rx=1, doppler_period=N4)
        return ch.generate(n_slots=N4)

    def test_recovers_doppler_shift_exactly(self):
        N3, N4 = 8, 4
        H = self._mobile_channel(N3, N4)
        cbk = R18Type2Codebook(self.CFG8, N3=N3, N4=N4, param_combination=3)
        pmi = cbk.select(H, rank=1)
        # the optimal precoder rotates with the *conjugate* of the channel's
        # Doppler phase: channel shift +1 => precoder shift N4-1 = 3
        assert pmi.i110 == [2]
        W = cbk.precoder(pmi)
        from nr_csi.baselines import eigen_precoder

        for iota in range(N4):
            target = eigen_precoder(H[iota], rank=1)
            assert sgcs(target, W[iota]) > 1 - 1e-9

    def test_outperforms_static_r16_under_mobility(self):
        """R16 reports once and holds the precoder; R18 predicts per interval."""
        N3, N4 = 8, 4
        H = self._mobile_channel(N3, N4)
        from nr_csi.baselines import eigen_precoder

        targets = [eigen_precoder(H[i], rank=1) for i in range(N4)]

        r18 = R18Type2Codebook(self.CFG8, N3=N3, N4=N4, param_combination=3)
        W18 = r18.precoder(r18.select(H, rank=1))
        s18 = np.mean([sgcs(targets[i], W18[i]) for i in range(N4)])

        r16 = R16Type2Codebook(self.CFG8, N3=N3, param_combination=3)
        W16 = r16.precoder(r16.select(H[-1:], rank=1))  # latest measurement, held
        s16 = np.mean([sgcs(targets[i], W16[0]) for i in range(N4)])

        assert s18 > 0.999
        assert s16 < 0.9  # held precoder degrades on future intervals
        assert s18 > s16 + 0.1


class TestOverhead:
    def test_bits(self):
        cbk = R18Type2Codebook(CFG, N3=8, N4=4, param_combination=3)
        pmi = random_valid_pmi(cbk, rank=2, seed=9)
        bits = cbk.overhead_bits(pmi)
        Mv = cbk.Mv(2)
        assert bits["i17"] == 2 * 2 * cbk.L * Mv * 2  # v * 2L * Mv * Q
        assert bits["i18"] == 2 * int(np.ceil(np.log2(2 * cbk.L * 2)))
        assert bits["i110"] == 2 * int(np.ceil(np.log2(3)))  # N4 - 1 = 3
        K_nz = int(pmi.i17.sum())
        assert bits["i24"] == 3 * (K_nz - 2)
        assert bits["i25"] == 4 * (K_nz - 2)

    def test_n4_two_zero_offset_bits(self):
        cbk = R18Type2Codebook(CFG, N3=8, N4=2, param_combination=3)
        pmi = random_valid_pmi(cbk, rank=1, seed=10)
        assert cbk.overhead_bits(pmi)["i110"] == 0
