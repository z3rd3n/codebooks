"""R16 eType II anchors: normalization, FD compression, budgets, i18, compact model."""

import numpy as np
import pytest

from nr_csi.codebooks import compact
from nr_csi.codebooks.etype2_r16 import (
    R16Type2Codebook,
    R16Type2PMI,
    decode_i18,
    encode_i18,
)
from nr_csi.config import AntennaConfig, R16_PARAM_COMBOS
from nr_csi.metrics import sgcs
from nr_csi.utils import combinatorics as cb
from nr_csi.utils import dft
from nr_csi.utils import quantization as qt

CFG = AntennaConfig.standard(4, 2)  # P = 16, N1N2 = 8


def random_valid_pmi(cbk: R16Type2Codebook, rank: int, seed: int = 0) -> R16Type2PMI:
    rng = np.random.default_rng(seed)
    a, L = cbk.antenna, cbk.L
    Mv = cbk.Mv(rank)
    pmi = R16Type2PMI(rank=rank)
    if cbk.port_selection:
        pmi.i11_ps = int(rng.integers(int(np.ceil(a.P / (2 * cbk.d)))))
    else:
        flat = sorted(rng.choice(a.N1 * a.N2, size=L, replace=False).tolist())
        pmi.q1 = int(rng.integers(a.O1))
        pmi.q2 = int(rng.integers(a.O2))
        pmi.i12 = cb.combo_to_index(flat, a.N1 * a.N2)
    if cbk.N3 > 19:
        m_init = int(rng.integers(-2 * Mv + 1, 1))
        pmi.i15 = m_init if m_init == 0 else m_init + 2 * Mv
    else:
        m_init = None
    pmi.i17 = np.zeros((rank, Mv, 2 * L), dtype=bool)
    pmi.k1 = np.ones((rank, 2), dtype=int)
    pmi.k2 = np.zeros((rank, Mv, 2 * L), dtype=int)
    pmi.c = np.zeros((rank, Mv, 2 * L), dtype=int)
    for li in range(rank):
        if cbk.N3 > 19:
            window = [(m_init + j) % cbk.N3 for j in range(2 * Mv)]
            rest = rng.choice([w for w in window if w != 0], size=Mv - 1, replace=False)
            taps = sorted([0] + [int(t) for t in rest])
        else:
            taps = sorted(
                [0] + rng.choice(np.arange(1, cbk.N3), size=Mv - 1, replace=False).tolist()
            )
        i16, _ = cb.encode_taps(taps, cbk.N3, Mv, m_initial=m_init)
        pmi.i16.append(i16)
        n_keep = min(cbk.K0, Mv * 2 * L)
        keep = rng.choice(Mv * 2 * L, size=n_keep, replace=False)
        bm = np.zeros(Mv * 2 * L, dtype=bool)
        bm[keep] = True
        bm = bm.reshape(Mv, 2 * L)
        i_star = int(rng.integers(2 * L))
        bm[0, i_star] = True
        pmi.i17[li] = bm
        pmi.k1[li] = [15, int(rng.integers(1, 16))]
        if i_star >= L:  # strongest polarization gets the reference 15
            pmi.k1[li] = pmi.k1[li][::-1]
        pmi.k2[li][bm] = rng.integers(0, 8, size=int(bm.sum()))
        pmi.c[li][bm] = rng.integers(0, 16, size=int(bm.sum()))
        pmi.k2[li, 0, i_star] = 7
        pmi.c[li, 0, i_star] = 0
        pmi.i18.append(encode_i18(bm, i_star, rank))
    return pmi


class TestReconstruction:
    @pytest.mark.parametrize("rank", [1, 2, 4])
    def test_unit_norm_per_layer_and_total(self, rank):
        combo = 4 if rank <= 2 else 5
        cbk = R16Type2Codebook(CFG, N3=12, param_combination=combo)
        pmi = random_valid_pmi(cbk, rank, seed=rank)
        W = cbk.precoder(pmi)
        assert W.shape == (1, 12, CFG.P, rank)
        for t in range(12):
            assert np.allclose(np.linalg.norm(W[0, t], axis=0), 1 / np.sqrt(rank))

    @pytest.mark.parametrize("port_selection", [False, True])
    @pytest.mark.parametrize("rank", [1, 2])
    def test_spec_equals_compact_model(self, rank, port_selection):
        """Spec-table reconstruction == compact model W_s W_c W_f^T (directions)."""
        cbk = R16Type2Codebook(CFG, N3=12, param_combination=3, port_selection=port_selection)
        for seed in range(5):
            pmi = random_valid_pmi(cbk, rank, seed=seed)
            W_spec = cbk.precoder(pmi)[0]  # (N3, P, v)
            Mv = cbk.Mv(rank)
            B = cbk._basis(pmi)
            for li in range(rank):
                taps = cb.decode_taps(pmi.i16[li], cbk.N3, Mv, pmi.i15)
                Wc = cbk._layer_coefficients(pmi, li)  # (2L, Mv) incl p1
                W_cm = compact.compact_r16(B, Wc, taps, cbk.N3)  # (P, N3)
                s = sgcs(W_spec[:, :, li][:, :, None], W_cm.T[:, :, None])
                assert s > 1 - 1e-12

    def test_i18_dual_mode_roundtrip(self):
        rng = np.random.default_rng(3)
        for rank in (1, 2):
            for _ in range(20):
                bm = rng.random((3, 8)) < 0.5
                i_star = int(rng.integers(8))
                bm[0, i_star] = True
                i18 = encode_i18(bm, i_star, rank)
                assert decode_i18(i18, bm, rank) == i_star

    def test_k0_value(self):
        # combo 4: L=4, beta=1/2, p_v(1)=1/4; N3=12,R=1 -> M1=3, K0=ceil(0.5*8*3)=12
        cbk = R16Type2Codebook(CFG, N3=12, param_combination=4)
        assert cbk.K0 == 12
        assert cbk.Mv(1) == 3 and cbk.Mv(3) == 2  # p_v34 = 1/8 -> ceil(12/8)


class TestSelection:
    def test_exact_recovery_with_fd_compression(self):
        """Channel whose per-subband precoders are an exact 3-tap codebook word."""
        N3 = 12
        cbk = R16Type2Codebook(CFG, N3=N3, param_combination=5)  # L=4, Mv=3, K0=18
        # all beams share m2=2 so only group (1,2) reaches 100% energy with
        # <= 4 beams (any q2 group spans the N2=2 vertical space, so beams
        # spread over several m2 values would make the group choice ambiguous)
        beams = [(1, 2), (5, 2), (9, 2)]  # group (1,2), flat n = 0, 1, 2
        taps = [0, 3, 9]
        amps = [1.0, 0.5, 0.5]
        half = CFG.P // 2
        w = np.zeros((N3, CFG.P), dtype=complex)
        for (m1, m2), tap, amp in zip(beams, taps, amps):
            v = dft.spatial_beam(CFG, m1, m2)
            w[:, :half] += amp * np.exp(2j * np.pi * np.arange(N3) * tap / N3)[:, None] * v
        H = w.conj()[None, :, None, :]  # (1, N3, 1, P)
        pmi = cbk.select(H, rank=1)
        assert (pmi.q1, pmi.q2) == (1, 2)
        assert cb.decode_taps(pmi.i16[0], N3, cbk.Mv(1), pmi.i15) == [0, 3, 9]
        assert pmi.k1[0, 0] == 15  # strongest polarization is pol 0
        W = cbk.precoder(pmi)[0]  # (N3, P, 1)
        w_ref = w / np.linalg.norm(w, axis=1, keepdims=True)
        assert sgcs(w_ref[:, :, None], W) > 1 - 1e-10

    def test_budget_constraints_respected(self):
        rng = np.random.default_rng(11)
        cbk = R16Type2Codebook(CFG, N3=12, param_combination=5)  # supports rank 4
        H = rng.standard_normal((1, 12, 4, CFG.P)) + 1j * rng.standard_normal((1, 12, 4, CFG.P))
        for rank in (1, 2, 3, 4):
            pmi = cbk.select(H, rank=rank)
            per_layer = pmi.i17.reshape(rank, -1).sum(axis=1)
            assert (per_layer <= cbk.K0).all()
            assert pmi.i17.sum() <= 2 * cbk.K0
            for li in range(rank):
                i_star = decode_i18(pmi.i18[li], pmi.i17[li], rank)
                assert pmi.i17[li][0, i_star]
                assert pmi.k2[li, 0, i_star] == 7
                assert pmi.c[li, 0, i_star] == 0
                assert pmi.k1[li, i_star // cbk.L] == 15

    def test_two_level_indication_large_n3(self):
        """N3 > 19 engages the i15 window; all layers share one M_initial."""
        N3 = 24
        cbk = R16Type2Codebook(CFG, N3=N3, param_combination=4, R=2)  # Mv = ceil(.25*12)=3
        rng = np.random.default_rng(12)
        H = rng.standard_normal((1, N3, 2, CFG.P)) + 1j * rng.standard_normal((1, N3, 2, CFG.P))
        pmi = cbk.select(H, rank=2)
        assert pmi.i15 is not None and 0 <= pmi.i15 < 2 * cbk.Mv(2)
        W = cbk.precoder(pmi)
        assert np.all(np.isfinite(W))
        for li in range(2):
            taps = cb.decode_taps(pmi.i16[li], N3, cbk.Mv(2), pmi.i15)
            assert taps[0] == 0 and len(taps) == 3

    def test_fd_compression_tracks_delay_channel(self):
        """A 2-tap channel is represented far better than its uncompressed SE
        would suggest given only Mv=3 of 12 taps are reported."""
        from nr_csi.channel import Ray, SyntheticRayChannel

        rays = [
            Ray(gain=1.0, m1=4, m2=0, delay=0),
            Ray(gain=0.6, m1=8, m2=4, delay=5),
        ]
        ch = SyntheticRayChannel(CFG, rays, N3=12, n_rx=1)
        H = ch.generate()
        cbk = R16Type2Codebook(CFG, N3=12, param_combination=5)
        pmi = cbk.select(H, rank=1)
        from nr_csi.codebooks._spatial import aligned_eigen_targets

        targets = aligned_eigen_targets(H[0], rank=1)
        W = cbk.precoder(pmi)[0]
        assert sgcs(targets, W) > 0.95


class TestPortSelection:
    def test_ps_basis_and_selection(self):
        cbk = R16Type2Codebook(CFG, N3=4, param_combination=1, port_selection=True, d=2)
        pmi = random_valid_pmi(cbk, rank=1, seed=5)
        B = cbk._basis(pmi)
        half = CFG.P // 2
        for i in range(cbk.L):
            assert B[i, (pmi.i11_ps * 2 + i) % half] == 1.0

    def test_ps_recovers_consecutive_port_channel(self):
        half = CFG.P // 2
        N3 = 4
        w = np.zeros((N3, CFG.P), dtype=complex)
        # energy on consecutive ports 2,3 with one delay tap on port 3;
        # combo 6 gives Mv = ceil(N3/2) = 2 so the tap can be represented
        w[:, 2] = 1.0
        w[:, 3] = 0.5 * np.exp(2j * np.pi * np.arange(N3) * 1 / N3)
        H = w.conj()[None, :, None, :]
        cbk = R16Type2Codebook(CFG, N3=N3, param_combination=6, port_selection=True, d=2)
        pmi = cbk.select(H, rank=1)
        window = [(pmi.i11_ps * 2 + i) % half for i in range(cbk.L)]
        assert 2 in window and 3 in window
        W = cbk.precoder(pmi)[0]
        w_ref = w / np.linalg.norm(w, axis=1, keepdims=True)
        assert sgcs(w_ref[:, :, None], W) > 1 - 1e-9


class TestOverhead:
    def test_bit_formula_small_n3(self):
        cbk = R16Type2Codebook(CFG, N3=12, param_combination=4)  # L=4, Mv=3
        pmi = random_valid_pmi(cbk, rank=2, seed=6)
        bits = cbk.overhead_bits(pmi)
        assert bits["i11"] == 4  # log2(16)
        assert bits["i12"] == 7  # ceil(log2 C(8,4)) = ceil(log2 70)
        assert "i15" not in bits
        assert bits["i16"] == 2 * int(np.ceil(np.log2(55)))  # C(11,2) = 55
        assert bits["i17"] == 2 * 2 * 4 * 3  # v * 2L * Mv
        assert bits["i18"] == 2 * 3  # ceil(log2 8)
        K_nz = int(pmi.i17.sum())
        assert bits["i24"] == 3 * (K_nz - 2)
        assert bits["i25"] == 4 * (K_nz - 2)

    def test_bit_formula_large_n3(self):
        cbk = R16Type2Codebook(CFG, N3=24, param_combination=4, R=2)  # Mv=3
        pmi = random_valid_pmi(cbk, rank=1, seed=7)
        bits = cbk.overhead_bits(pmi)
        assert bits["i15"] == int(np.ceil(np.log2(6)))
        assert bits["i16"] == int(np.ceil(np.log2(10)))  # C(2Mv-1, Mv-1) = C(5,2)
