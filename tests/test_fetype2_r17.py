"""R17 FeType II anchors: free port selection, M/alpha combos, R15-PS comparison."""

import numpy as np

from nr_csi.codebooks.fetype2_r17 import R17Type2Codebook
from nr_csi.codebooks.type2_r15 import R15Type2Codebook
from nr_csi.config import AntennaConfig
from nr_csi.metrics import sgcs
from nr_csi.utils import combinatorics as cb

CFG = AntennaConfig.standard(4, 2)  # P = 16, P/2 = 8


def port_channel(ports_amps_taps, N3, P):
    """Beam-domain channel with energy on given (port, amp, tap) triples."""
    w = np.zeros((N3, P), dtype=complex)
    for port, amp, tap in ports_amps_taps:
        w[:, port] = amp * np.exp(2j * np.pi * np.arange(N3) * tap / N3)
    return w.conj()[None, :, None, :], w


class TestConfig:
    def test_L_derived_from_alpha(self):
        # combo 5: alpha = 1/2 -> K1 = 8, L = 4
        cbk = R17Type2Codebook(CFG, N3=8, param_combination=5)
        assert (cbk.K1, cbk.L, cbk.M) == (8, 4, 2)
        # combo 4: alpha = 1 -> all 16 ports, M = 1
        cbk = R17Type2Codebook(CFG, N3=8, param_combination=4)
        assert (cbk.K1, cbk.L, cbk.M) == (16, 8, 1)

    def test_k0_value(self):
        cbk = R17Type2Codebook(CFG, N3=8, param_combination=5)  # beta=1/2
        assert cbk.K0 == int(np.ceil(0.5 * 8 * 2))


class TestSelection:
    def test_alpha_one_selects_all_ports(self):
        rng = np.random.default_rng(0)
        H = rng.standard_normal((1, 8, 1, CFG.P)) + 1j * rng.standard_normal((1, 8, 1, CFG.P))
        cbk = R17Type2Codebook(CFG, N3=8, param_combination=4)
        pmi = cbk.select(H, rank=1)
        assert pmi.i12 is None  # not reported
        B = cbk._basis(pmi)
        assert np.allclose(B, np.eye(CFG.P // 2))

    def test_m1_gives_wideband_flat_precoder(self):
        rng = np.random.default_rng(1)
        H = rng.standard_normal((1, 8, 1, CFG.P)) + 1j * rng.standard_normal((1, 8, 1, CFG.P))
        cbk = R17Type2Codebook(CFG, N3=8, param_combination=2)  # M=1
        W = cbk.precoder(cbk.select(H, rank=1))
        for t in range(1, 8):
            assert np.allclose(W[0, t], W[0, 0])

    def test_offset_indicator_recovered(self):
        # ports 1 and 6 (non-contiguous), second tap at offset 2
        H, w = port_channel([(1, 1.0, 0), (6, 0.5, 2)], N3=8, P=CFG.P)
        cbk = R17Type2Codebook(CFG, N3=8, param_combination=5, N_window=4)  # alpha=1/2, M=2
        pmi = cbk.select(H, rank=1)
        assert cbk.taps(pmi) == [0, 2]
        assert pmi.i16 == 1  # offset 2 -> index 1
        ports = cb.decode_ports(pmi.i12, CFG.P, cbk.L)
        assert 1 in ports and 6 in ports
        W = cbk.precoder(pmi)[0]
        w_ref = w / np.linalg.norm(w, axis=1, keepdims=True)
        assert sgcs(w_ref[:, :, None], W) > 1 - 1e-9

    def test_free_selection_beats_r15_consecutive_ps(self):
        """The R17 motivation: non-contiguous strong ports."""
        H, w = port_channel(
            [(0, 1.0, 0), (3, 0.8, 0), (5, 0.7, 0), (7, 0.6, 0)], N3=4, P=CFG.P
        )
        w_ref = (w / np.linalg.norm(w, axis=1, keepdims=True))[:, :, None]
        r17 = R17Type2Codebook(CFG, N3=4, param_combination=5)  # L=4 free ports
        s17 = sgcs(w_ref, r17.precoder(r17.select(H, rank=1))[0])
        r15 = R15Type2Codebook(CFG, N3=4, L=4, port_selection=True, d=1)
        s15 = sgcs(w_ref, r15.precoder(r15.select(H, rank=1))[0])
        assert s17 > 0.99
        assert s17 > s15 + 0.05  # consecutive windows cannot cover {0,3,5,7}

    def test_budget_and_norm(self):
        rng = np.random.default_rng(2)
        # rank 4 requires at least 4 receive antennas (RI <= Nr)
        H = rng.standard_normal((1, 8, 4, CFG.P)) + 1j * rng.standard_normal((1, 8, 4, CFG.P))
        cbk = R17Type2Codebook(CFG, N3=8, param_combination=8)  # M=2, beta=3/4
        for rank in (1, 2, 3, 4):
            pmi = cbk.select(H, rank=rank)
            per_layer = pmi.i17.reshape(rank, -1).sum(axis=1)
            assert (per_layer <= cbk.K0).all()
            assert pmi.i17.sum() <= 2 * cbk.K0
            W = cbk.precoder(pmi)
            for t in range(8):
                assert np.allclose(np.linalg.norm(W[0, t], axis=0), 1 / np.sqrt(rank))

    def test_strongest_coefficient_indicator(self):
        H, _ = port_channel([(2, 1.0, 0), (4, 0.5, 1)], N3=8, P=CFG.P)
        cbk = R17Type2Codebook(CFG, N3=8, param_combination=5)
        pmi = cbk.select(H, rank=1)
        f_star, i_star = divmod(pmi.i18[0], cbk.K1)
        assert pmi.i17[0, f_star, i_star]
        assert pmi.k2[0, f_star, i_star] == 7
        assert pmi.c[0, f_star, i_star] == 0
        assert pmi.k1[0, i_star // cbk.L] == 15


class TestOverhead:
    def test_bits(self):
        cbk = R17Type2Codebook(CFG, N3=8, param_combination=5)  # alpha=1/2, M=2
        pmi = cbk.select(
            port_channel([(1, 1.0, 0), (6, 0.5, 2)], N3=8, P=CFG.P)[0], rank=1
        )
        bits = cbk.overhead_bits(pmi)
        assert bits["i12"] == int(np.ceil(np.log2(70)))  # C(8,4)
        assert bits["i16"] == int(np.ceil(np.log2(3)))  # N_window-1 = 3
        assert bits["i17"] == cbk.K1 * cbk.M
        assert bits["i18"] == int(np.ceil(np.log2(cbk.K1 * cbk.M)))
        K_nz = int(pmi.i17.sum())
        assert bits["i24"] == 3 * (K_nz - 1)
        assert bits["i25"] == 4 * (K_nz - 1)

    def test_alpha_one_drops_i12(self):
        cbk = R17Type2Codebook(CFG, N3=8, param_combination=4)
        rng = np.random.default_rng(3)
        H = rng.standard_normal((1, 8, 1, CFG.P)) + 1j * rng.standard_normal((1, 8, 1, CFG.P))
        bits = cbk.overhead_bits(cbk.select(H, rank=1))
        assert "i12" not in bits and "i16" not in bits  # M=1 drops i16 too
