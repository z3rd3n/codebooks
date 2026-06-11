"""R15 Type II anchors: normalization, reporting rules, exact recovery, PS/PEB equivalence."""

import numpy as np
import pytest

from nr_csi.codebooks.type2_r15 import R15Type2Codebook, R15Type2PMI, k2_cap
from nr_csi.config import AntennaConfig
from nr_csi.metrics import sgcs
from nr_csi.utils import dft
from nr_csi.utils import quantization as qt

CFG = AntennaConfig.standard(4, 2)  # P = 16


def make_target_channel(cfg, beams, amps, phases, pol_amps, pol_phases, N3=1):
    """Rank-1 channel whose optimal precoder is an exact codebook combination.

    w_target = [sum a_i e^{j phi_i} v_i ; sum b_i e^{j psi_i} v_i]  (unnormalized)
    """
    half = cfg.P // 2
    w = np.zeros(cfg.P, dtype=complex)
    for (m1, m2), a, ph, b, ps in zip(beams, amps, phases, pol_amps, pol_phases):
        v = dft.spatial_beam(cfg, m1, m2)
        w[:half] += a * np.exp(1j * ph) * v
        w[half:] += b * np.exp(1j * ps) * v
    H = np.tile(w.conj()[None, None, None, :], (1, N3, 1, 1))  # (1, N3, 1, P)
    return H, w / np.linalg.norm(w)


class TestReconstruction:
    def _random_pmi(self, cbk, rank=1, seed=0):
        rng = np.random.default_rng(seed)
        a = cbk.antenna
        n = sorted(rng.choice(a.N1 * a.N2, size=cbk.L, replace=False).tolist())
        from nr_csi.utils import combinatorics as cb

        pmi = R15Type2PMI(
            rank=rank,
            q1=int(rng.integers(a.O1)),
            q2=int(rng.integers(a.O2)),
            i12=cb.combo_to_index(n, a.N1 * a.N2),
            i13=[int(rng.integers(2 * cbk.L)) for _ in range(rank)],
            k1=rng.integers(1, 8, size=(rank, 2 * cbk.L)),
            k2=rng.integers(0, 2, size=(rank, cbk.N3, 2 * cbk.L)),
            c=rng.integers(0, cbk.n_psk, size=(rank, cbk.N3, 2 * cbk.L)),
        )
        for li in range(rank):
            pmi.k1[li, pmi.i13[li]] = 7
            pmi.k2[li, :, pmi.i13[li]] = 1
            pmi.c[li, :, pmi.i13[li]] = 0
        return pmi

    @pytest.mark.parametrize("rank", [1, 2])
    def test_unit_norm_per_layer_and_total(self, rank):
        cbk = R15Type2Codebook(CFG, N3=3, L=4, subband_amplitude=True)
        pmi = self._random_pmi(cbk, rank=rank)
        W = cbk.precoder(pmi)
        assert W.shape == (1, 3, CFG.P, rank)
        for t in range(3):
            # each layer column norm 1/sqrt(rank) => total power 1 (spec scaling)
            assert np.allclose(np.linalg.norm(W[0, t], axis=0), 1 / np.sqrt(rank))
            assert np.isclose(np.linalg.norm(W[0, t]), 1.0)

    def test_beams_confined_to_selected_combination(self):
        """Within the selected orthogonal group, only the L chosen beams carry
        energy (beams of one group form an orthogonal basis; cross-group beams
        are deliberately non-orthogonal, so we check intra-group support)."""
        from nr_csi.utils import combinatorics as cb

        cbk = R15Type2Codebook(CFG, N3=1, L=2)
        pmi = self._random_pmi(cbk)
        W = cbk.precoder(pmi)[0, 0, :, 0]
        Bg = dft.orthogonal_group(CFG, pmi.q1, pmi.q2)
        half = CFG.P // 2
        n1, n2 = cb.decode_beam_combination(pmi.i12, CFG.N1, CFG.N2, cbk.L)
        selected = {b * CFG.N1 + a for a, b in zip(n1, n2)}
        for pol in (W[:half], W[half:]):
            proj = np.abs(Bg.conj() @ pol)
            for n in range(CFG.N1 * CFG.N2):
                if n not in selected:
                    assert proj[n] < 1e-9


class TestSelection:
    def test_exact_recovery_on_table_grid(self):
        """Channel built from L grid beams with table-exact amplitudes/phases."""
        beams = [(1, 2), (5, 2), (1, 6), (9, 10)]  # all in group (q1,q2)=(1,2)
        amps = [1.0, np.sqrt(1 / 4), np.sqrt(1 / 2), np.sqrt(1 / 16)]
        phases = [0.0, 2 * np.pi * 3 / 8, 2 * np.pi * 5 / 8, 2 * np.pi * 1 / 8]
        pol_amps = [np.sqrt(1 / 8), np.sqrt(1 / 4), np.sqrt(1 / 32), np.sqrt(1 / 4)]
        pol_phases = [2 * np.pi * 7 / 8, 0.0, 2 * np.pi * 2 / 8, 2 * np.pi * 4 / 8]
        H, w_target = make_target_channel(CFG, beams, amps, phases, pol_amps, pol_phases)
        cbk = R15Type2Codebook(CFG, N3=1, L=4, n_psk=8)
        pmi = cbk.select(H, rank=1)
        assert (pmi.q1, pmi.q2) == (1, 2)
        W = cbk.precoder(pmi)
        assert sgcs(w_target[:, None], W[0, 0]) > 1 - 1e-10

    def test_strongest_coefficient_convention(self):
        H, _ = make_target_channel(
            CFG, [(0, 0), (4, 0)], [1.0, 0.5], [0.3, 1.1], [0.7, 0.2], [2.0, 0.5]
        )
        cbk = R15Type2Codebook(CFG, N3=1, L=2, subband_amplitude=True)
        pmi = cbk.select(H, rank=1)
        i_star = pmi.i13[0]
        assert pmi.k1[0, i_star] == 7
        assert pmi.k2[0, 0, i_star] == 1
        assert pmi.c[0, 0, i_star] == 0

    def test_sa_false_means_all_subband_amps_one(self):
        rng = np.random.default_rng(7)
        H = rng.standard_normal((1, 4, 2, CFG.P)) + 1j * rng.standard_normal((1, 4, 2, CFG.P))
        cbk = R15Type2Codebook(CFG, N3=4, L=4, subband_amplitude=False)
        pmi = cbk.select(H, rank=2)
        assert (pmi.k2 == 1).all()

    def test_quantization_improves_with_L(self):
        rng = np.random.default_rng(8)
        H = rng.standard_normal((1, 1, 1, CFG.P)) + 1j * rng.standard_normal((1, 1, 1, CFG.P))
        from nr_csi.baselines import eigen_precoder

        target = eigen_precoder(H[0], rank=1)
        scores = []
        for L in (2, 3, 4):
            cbk = R15Type2Codebook(CFG, N3=1, L=L)
            W = cbk.precoder(cbk.select(H, rank=1))
            scores.append(sgcs(target, W[0, 0][None]))
        assert scores[0] <= scores[1] + 1e-6 <= scores[2] + 2e-6
        # i.i.d. channels have no angular sparsity; top-4-of-8 beams capture
        # roughly 80% of the energy, so ~0.75+ SGCS is the expected regime
        assert scores[2] > 0.75


class TestReportingRules:
    def test_k2_cap_values(self):
        assert k2_cap(2) == 4 and k2_cap(3) == 4 and k2_cap(4) == 6

    def test_partition_caps_subband_amplitudes(self):
        cbk = R15Type2Codebook(CFG, N3=1, L=4, subband_amplitude=True)
        k1 = np.array([7, 6, 6, 5, 4, 3, 2, 1])  # all 8 nonzero
        strong, weak, zero = cbk._partition(k1, i_star=0)
        assert len(strong) == k2_cap(4) - 1  # 5 strongest besides the strongest
        assert len(weak) == 8 - k2_cap(4)  # 2 weakest fall back to QPSK
        assert zero == []
        sizes = cbk._phase_alphabets(k1, i_star=0)
        assert (sizes[strong] == cbk.n_psk).all()
        assert (sizes[weak] == 4).all()

    def test_zero_amplitude_coefficients_not_reported(self):
        cbk = R15Type2Codebook(CFG, N3=1, L=2, subband_amplitude=True)
        k1 = np.array([7, 0, 3, 0])
        strong, weak, zero = cbk._partition(k1, i_star=0)
        assert zero == [1, 3]
        assert strong == [2] and weak == []


class TestPortSelectionEquivalence:
    def test_ps_through_dft_peb_equals_regular(self):
        """Paper eqs. (regy)/(psy): PS codebook + full-connect DFT PEB does the
        same job as the regular codebook on the antenna-domain channel."""
        half = CFG.P // 2
        # physical channel built from group-(0,0) beams whose flat indices
        # {0,1,2,3} are consecutive: R15 PS can only select consecutive ports
        # (the free-selection limitation lifted later by R17)
        beams = [(0, 0), (4, 0), (8, 0), (12, 0)]  # n1 = 0..3, n2 = 0
        amps = [1.0, np.sqrt(1 / 2), np.sqrt(1 / 4), np.sqrt(1 / 8)]
        phases = [0, np.pi / 4, np.pi / 2, -np.pi / 4]
        H_phys, w_target = make_target_channel(CFG, beams, amps, phases, amps, phases)

        # full-connect PEB: columns = the N1N2 orthogonal beams of group (0,0)
        F = dft.orthogonal_group(CFG, 0, 0).T  # (P/2 physical, P/2 logical)
        H1, H2 = H_phys[0, 0, :, :half], H_phys[0, 0, :, half:]
        H_eff = np.concatenate([H1 @ F, H2 @ F], axis=1)[None, None]  # beam-domain

        ps = R15Type2Codebook(CFG, N3=1, L=4, port_selection=True, d=1)
        pmi = ps.select(H_eff, rank=1)
        w_ps = ps.precoder(pmi)[0, 0, :, 0]
        # map the PS precoder through the PEB back to the physical antennas
        w_phys = np.concatenate([F @ w_ps[:half], F @ w_ps[half:]])
        assert sgcs(w_target[:, None], w_phys[:, None]) > 1 - 1e-10

    def test_ps_port_window_and_wraparound(self):
        ps = R15Type2Codebook(CFG, N3=1, L=2, port_selection=True, d=2)
        pmi = R15Type2PMI(
            rank=1, i11_ps=3, i13=[0],
            k1=np.array([[7, 7, 7, 7]]),
            k2=np.ones((1, 1, 4), dtype=int),
            c=np.zeros((1, 1, 4), dtype=int),
        )
        B = ps._basis(pmi)
        assert B[0, 6] == 1.0 and B[1, 7] == 1.0  # ports 3*2, 3*2+1


class TestOverhead:
    def test_sa_false_bit_counts(self):
        cbk = R15Type2Codebook(CFG, N3=3, L=2, n_psk=8, subband_amplitude=False)
        pmi = R15Type2PMI(
            rank=1, q1=0, q2=0, i12=0, i13=[0],
            k1=np.array([[7, 5, 4, 3]]),  # all nonzero -> Ml = 4
            k2=np.ones((1, 3, 4), dtype=int),
            c=np.zeros((1, 3, 4), dtype=int),
        )
        bits = cbk.overhead_bits(pmi)
        assert bits["i11"] == 4  # log2(O1*O2) = log2(16)
        assert bits["i12"] == 5  # ceil(log2(C(8,2))) = ceil(log2 28)
        assert bits["i13"] == 2  # log2(2L) = 2
        assert bits["i14"] == 9  # 3*(2L-1)
        assert bits["i21"] == 3 * 3 * 3  # N3 * (Ml-1) * log2(8)
        assert "i22" not in bits

    def test_sa_true_bit_counts_with_cap(self):
        cbk = R15Type2Codebook(CFG, N3=2, L=4, n_psk=8, subband_amplitude=True)
        pmi = R15Type2PMI(
            rank=1, q1=0, q2=0, i12=0, i13=[0],
            k1=np.array([[7, 6, 6, 5, 4, 3, 2, 1]]),
            k2=np.ones((1, 2, 8), dtype=int),
            c=np.zeros((1, 2, 8), dtype=int),
        )
        bits = cbk.overhead_bits(pmi)
        # 5 strong (8-PSK) + 2 weak (QPSK) phases per subband
        assert bits["i21"] == 2 * (5 * 3 + 2 * 2)
        assert bits["i22"] == 2 * 5
