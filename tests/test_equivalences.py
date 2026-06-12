"""Cross-codebook equivalences (plan B5): the structural relationships the
paper states between codebook generations."""

import numpy as np
import pytest

from nr_csi.baselines.ideal import eigen_precoder
from nr_csi.channel import RandomRayChannel, Ray, SyntheticRayChannel
from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    Type1Codebook,
)
from nr_csi.config import AntennaConfig
from nr_csi.metrics.similarity import sgcs
from nr_csi.utils import dft

ANT = AntennaConfig.standard(4, 2)  # P = 16


def beam_domain(H_phys, F):
    """Apply a full-connect PEB F per polarization: physical -> beam domain."""
    half = F.shape[0]
    return np.concatenate([H_phys[..., :half] @ F, H_phys[..., half:] @ F], axis=-1)


class TestR16PortSelectionPEB:
    def test_ps_through_dft_peb_equals_regular(self):
        """R16 level of the R15 PEB identity: the PS codebook fed with the
        beam-domain channel reconstructs the same precoders as the regular
        codebook on the physical channel, including the delay compression."""
        N3, L = 8, 4
        rng = np.random.default_rng(4)
        half = ANT.P // 2
        # physical channel from the 4 consecutive group-(0,0) beams 0..3
        # (R16 PS selects consecutive ports), each with its own delay tap
        beams = dft.orthogonal_group(ANT, 0, 0)[:L]  # (L, P/2)
        delays = [0, 1, 2, 3]
        t = np.arange(N3)
        w = np.zeros((N3, ANT.P), dtype=complex)
        for i in range(L):
            g1, g2 = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            ramp = np.exp(2j * np.pi * t * delays[i] / N3)
            w[:, :half] += 0.5 * g1 * np.outer(ramp, beams[i])
            w[:, half:] += 0.5 * g2 * np.outer(ramp, beams[i])
        H_phys = w.conj()[None, :, None, :]  # (1, N3, 1, P)

        reg = R16Type2Codebook(ANT, N3=N3, param_combination=4)
        pmi_reg = reg.select(H_phys, rank=1)
        W_reg = reg.precoder(pmi_reg)

        F = dft.orthogonal_group(ANT, 0, 0).T  # (P/2 physical, P/2 logical)
        H_eff = beam_domain(H_phys, F)
        ps = R16Type2Codebook(ANT, N3=N3, param_combination=4, port_selection=True, d=1)
        pmi_ps = ps.select(H_eff, rank=1)
        W_ps = ps.precoder(pmi_ps)
        # the compressed indications agree...
        assert pmi_ps.i16 == pmi_reg.i16
        assert int(pmi_ps.i17.sum()) == int(pmi_reg.i17.sum())
        # ... and the PS precoder mapped through the PEB equals the regular one
        for tt in range(N3):
            w_ps = W_ps[0, tt, :, 0]
            w_phys = np.concatenate([F @ w_ps[:half], F @ w_ps[half:]])
            assert sgcs(W_reg[0, tt], w_phys[:, None]) > 1 - 1e-9


class TestR17VsR16PS:
    def test_alpha1_m1_equals_r16ps_mv1_on_flat_channel(self):
        """R17 with alpha=1, M=1 and R16-PS with d=1, Mv=1 both reduce to
        per-port wideband coefficient quantization with the same quantizer
        and the same K0 budget: identical precoder directions."""
        ant = AntennaConfig.standard(2, 2)  # P = 8: R16 L=4 covers all ports
        N3 = 4
        rng = np.random.default_rng(1)
        h = rng.standard_normal(ant.P) + 1j * rng.standard_normal(ant.P)
        H = np.tile(h.conj()[None, None, None, :], (1, N3, 1, 1))  # flat

        r16ps = R16Type2Codebook(ant, N3=N3, param_combination=4,  # L=4, Mv=1
                                 port_selection=True, d=1)
        r17 = R17Type2Codebook(ant, N3=N3, param_combination=2)  # M=1, alpha=1
        assert r16ps.Mv(1) == 1 and r16ps.K0 == r17.K0
        W16 = r16ps.precoder(r16ps.select(H, rank=1))
        W17 = r17.precoder(r17.select(H, rank=1))
        for t in range(N3):
            assert sgcs(W16[0, t], W17[0, t]) > 1 - 1e-9

    def test_r17_beats_consecutive_selection_on_scattered_ports(self):
        """R17's free port selection vs the consecutive window: strong ports
        {0, 5} (per polarization) cannot be covered by an R16-PS window of
        L=2 consecutive ports, but R17 picks them freely."""
        ant = AntennaConfig.standard(4, 2)
        N3 = 2
        h = np.zeros(ant.P, dtype=complex)
        for p in (0, 5):  # scattered strong ports, both polarizations
            h[p] = 1.0
            h[p + ant.P // 2] = 0.7j
        H = np.tile(h.conj()[None, None, None, :], (1, N3, 1, 1))
        r17 = R17Type2Codebook(ant, N3=N3, param_combination=5)  # alpha=1/2, L=2
        r15ps = R15Type2Codebook(ant, N3=N3, L=2, port_selection=True, d=2)
        target = h[:, None] / np.linalg.norm(h)
        s17 = sgcs(target, r17.precoder(r17.select(H, rank=1))[0, 0])
        s15 = sgcs(target, r15ps.precoder(r15ps.select(H, rank=1))[0, 0])
        assert s17 > 0.99
        assert s17 > s15 + 0.1


class TestR15VsR16FlatChannel:
    def test_r16_at_least_as_good_minus_quantizer_gap(self):
        """On a frequency-flat *beam-sparse* channel only tap 0 is active, so
        R16's FD compression is lossless and the comparison reduces to
        quantizer resolution (16PSK/4+3-bit vs 8PSK/3-bit): R16 >= R15 - eps.

        Sparsity matters: R16's K0 = ceil(beta*2L*M1) = 4 coefficient budget
        cannot cover a dense channel that R15 (which reports all 2L wideband
        amplitudes) handles -- on i.i.d. flat channels R15 *wins*.  The paper's
        equivalence claim lives in Type II's design regime, <= K0 active
        beam coefficients.  (8,1) array: N2 = 2 would let the group heuristic
        pick an oblique vertical basis that needs more than K0 coefficients."""
        ant = AntennaConfig.standard(8, 1)
        N3 = 4
        rng = np.random.default_rng(2)
        r15 = R15Type2Codebook(ant, N3=N3, L=4, n_psk=8)
        r16 = R16Type2Codebook(ant, N3=N3, param_combination=4)  # L=4, K0=4
        beams = dft.orthogonal_group(ant, 0, 0)[[1, 6]]  # 2 beams x 2 pols = K0
        diffs = []
        for _ in range(10):
            g = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
            h = np.concatenate([g[0] @ beams, g[1] @ beams])
            H = np.tile(h.conj()[None, None, None, :], (1, N3, 1, 1))
            target = eigen_precoder(H, rank=1)
            s15 = sgcs(target, r15.precoder(r15.select(H, rank=1)))
            s16 = sgcs(target, r16.precoder(r16.select(H, rank=1)))
            assert s16 > 0.95  # within budget, R16 tracks the sparse channel
            diffs.append(s16 - s15)
        assert np.mean(diffs) > -0.01
        assert min(diffs) > -0.05


class TestTypeIVsTypeII:
    def test_single_beam_channel_makes_them_equal(self):
        """Paper's f2 remark: Type I is 'Type II with L=1'.  On a single
        on-grid beam channel with a QPSK polarization co-phase, both are
        exact and produce the same precoder.  (8,1) array: with N2 = 2 any
        two vertical beams of any group span the whole vertical space, and
        the group heuristic may pick an oblique exact basis instead."""
        ant = AntennaConfig.standard(8, 1)
        ray = Ray(gain=1.0, m1=12, pol_phase=np.pi / 2)
        H = SyntheticRayChannel(ant, [ray], N3=1).generate()
        target = eigen_precoder(H, rank=1)[0]
        t1 = Type1Codebook(ant, N3=1)
        t2 = R15Type2Codebook(ant, N3=1, L=2, n_psk=8)
        W1 = t1.precoder(t1.select(H, rank=1))
        W2 = t2.precoder(t2.select(H, rank=1))
        assert sgcs(target, W1[0]) > 1 - 1e-10
        assert sgcs(target, W2[0]) > 1 - 1e-10
        assert sgcs(W1[0, 0], W2[0, 0]) > 1 - 1e-10
