"""Tests for V6 features: Type II subset restriction (paper eqs. a58/a59 +
Table tabmaxap), rank-restriction bit sequences, tabCSS wiring, harness
realism knobs, and MU-MIMO evaluation."""

import numpy as np
import pytest

from nr_csi.channel import RandomRayChannel
from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
    TypeIIRestriction,
)
from nr_csi.config import AntennaConfig, n3_for_bwp, subband_size_options
from nr_csi.eval import evaluate, evaluate_mu
from nr_csi.utils import combinatorics as cb

ANT = AntennaConfig.standard(4, 2)  # P = 16, O1*O2 = 16


def random_channel(rng, N3, n_rx=2):
    return rng.standard_normal((1, N3, n_rx, ANT.P)) + 1j * rng.standard_normal(
        (1, N3, n_rx, ANT.P)
    )


def make_restriction(groups, b2_rows=None):
    """Restriction over the given 4 groups g = O1*r2 + r1."""
    r1 = [g % ANT.O1 for g in groups]
    r2 = [g // ANT.O1 for g in groups]
    beta1 = cb.encode_restriction_groups(r1, r2, ANT.O1, ANT.O2)
    b2 = np.zeros((4, ANT.N1 * ANT.N2), dtype=int) if b2_rows is None else b2_rows
    return TypeIIRestriction(beta1=beta1, b2=b2)


class TestSubsetRestriction:
    def test_b1_is_11_bits(self):
        restr = make_restriction([0, 3, 7, 12])
        assert restr.beta1 < 2**11
        g, r1, r2 = cb.decode_restriction_groups(restr.beta1, ANT.O1, ANT.O2)
        assert g == [0, 3, 7, 12]

    def test_prohibited_beam_never_selected(self):
        """All beams of 4 groups fully prohibited (codepoint 0): the selected
        group is never one of them, over many random channels."""
        groups = [0, 1, 4, 5]
        restr = make_restriction(groups)  # b2 all zero -> beams prohibited
        cbk = R15Type2Codebook(ANT, N3=2, L=2, restriction=restr)
        rng = np.random.default_rng(0)
        for _ in range(10):
            pmi = cbk.select(random_channel(rng, 2), rank=1)
            assert ANT.O1 * pmi.q2 + pmi.q1 not in groups

    def test_partially_prohibited_group_avoids_those_beams(self):
        """Group 0 has only beams {0, 1} allowed (max codepoint), the rest
        prohibited.  A channel concentrated on group-0 beams {0, 1, 2} must
        select within {0, 1} -- beam 2 is strong but prohibited."""
        from nr_csi.utils import dft

        b2 = np.zeros((4, ANT.N1 * ANT.N2), dtype=int)
        b2[0, [0, 1]] = 3
        restr = make_restriction([0, 5, 10, 15], b2)
        beams = dft.orthogonal_group(ANT, 0, 0)[[0, 1, 2]]
        h = np.concatenate([1.0 * beams[0] + 0.8 * beams[1] + 0.9 * beams[2],
                            0.5 * beams[0] + 0.4 * beams[1] + 0.45 * beams[2]])
        H = np.tile(h.conj()[None, None, None, :], (1, 2, 1, 1))

        # without the restriction, the strong-but-prohibited beam 2 is used
        free = R15Type2Codebook(ANT, N3=2, L=2).select(H, rank=1)
        assert (free.q1, free.q2) == (0, 0)
        n1, n2 = cb.decode_beam_combination(free.i12, ANT.N1, ANT.N2, 2)
        assert 2 in {ANT.N1 * b + a for a, b in zip(n1, n2)}

        # with it, whatever group is selected, no prohibited beam appears
        # (the codebook may legally escape to an unrestricted oblique group)
        cbk = R15Type2Codebook(ANT, N3=2, L=2, restriction=restr)
        pmi = cbk.select(H, rank=1)
        n1, n2 = cb.decode_beam_combination(pmi.i12, ANT.N1, ANT.N2, 2)
        caps = restr.k1_caps(ANT, pmi.q1, pmi.q2)
        assert all(caps[ANT.N1 * b + a] >= 0 for a, b in zip(n1, n2))
        assert (pmi.q1, pmi.q2, pmi.i12) != (free.q1, free.q2, free.i12)

    def test_amplitude_cap_honored(self):
        """Beams capped at sqrt(1/4) (codepoint 1) never report k1 > 5."""
        b2 = np.full((4, ANT.N1 * ANT.N2), 1, dtype=int)  # cap sqrt(1/4)
        b2[:, 0] = 3  # one full-power beam per group keeps a valid strongest
        groups = [0, 1, 2, 3]  # all q2=0 groups restricted
        restr = make_restriction(groups, b2)
        cbk = R15Type2Codebook(ANT, N3=2, L=4, restriction=restr)
        rng = np.random.default_rng(2)
        for _ in range(10):
            pmi = cbk.select(random_channel(rng, 2), rank=1)
            caps = cbk._selected_caps(pmi)
            for i in range(2 * cbk.L):
                if i == pmi.i13[0]:
                    continue
                assert pmi.k1[0, i] <= caps[i]

    def test_restriction_rejected_for_port_selection(self):
        with pytest.raises(ValueError, match="regular codebook"):
            R15Type2Codebook(ANT, N3=2, L=2, port_selection=True,
                             restriction=make_restriction([0, 1, 2, 3]))


class TestRankRestriction:
    def test_type1_prohibited_rank(self):
        r = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)  # rank 1 only
        cbk = Type1Codebook(ANT, N3=2, rank_restriction=r)
        rng = np.random.default_rng(3)
        H = random_channel(rng, 2)
        cbk.select(H, rank=1)  # allowed
        with pytest.raises(ValueError, match="prohibited"):
            cbk.select(H, rank=2)

    def test_r18_prohibited_rank(self):
        r = np.array([1, 0, 1, 1], dtype=bool)  # rank 2 prohibited
        cbk = R18Type2Codebook(ANT, N3=4, N4=2, param_combination=2, ri_restriction=r)
        rng = np.random.default_rng(4)
        H = rng.standard_normal((2, 4, 2, ANT.P)) + 1j * rng.standard_normal((2, 4, 2, ANT.P))
        cbk.select(H, rank=1)
        with pytest.raises(ValueError, match="prohibited"):
            cbk.select(H, rank=2)

    def test_bitmap_shape_validated(self):
        with pytest.raises(ValueError):
            Type1Codebook(ANT, N3=2, rank_restriction=np.ones(4, dtype=bool))
        with pytest.raises(ValueError):
            R18Type2Codebook(ANT, N3=4, N4=2, param_combination=2,
                             ri_restriction=np.ones(8, dtype=bool))


class TestTabCSSWiring:
    def test_paper_worked_example(self):
        assert n3_for_bwp(273, 16) == 18
        assert n3_for_bwp(273, 16, R=2) == 36

    def test_allowed_sizes(self):
        assert subband_size_options(24) == (4, 8)
        assert subband_size_options(100) == (8, 16)
        assert subband_size_options(275) == (16, 32)
        with pytest.raises(ValueError):
            subband_size_options(20)
        with pytest.raises(ValueError):
            n3_for_bwp(100, 4)  # size 4 not allowed for 100 RB

    def test_n3_range_claim(self):
        """Paper: N3 ranges over 3..36 across all valid configurations."""
        values = set()
        for n_rb in range(24, 276):
            for size in subband_size_options(n_rb):
                for R in (1, 2):
                    values.add(n3_for_bwp(n_rb, size, R))
        assert min(values) == 3
        assert max(values) == 36


class TestHarnessKnobs:
    def _mobile_channel(self):
        return RandomRayChannel(ANT, N3=4, n_rx=2, max_doppler=1.0, doppler_period=4)

    def test_feedback_delay_ages_static_reports(self):
        from nr_csi.codebooks import R16Type2Codebook

        cbk = R16Type2Codebook(ANT, N3=4, param_combination=6)
        fresh = evaluate(cbk, self._mobile_channel(), snr_db=[10.0], n_drops=10,
                         rng=np.random.default_rng(5))
        aged = evaluate(cbk, self._mobile_channel(), snr_db=[10.0], n_drops=10,
                        feedback_delay_slots=2, rng=np.random.default_rng(5))
        assert aged.sgcs < fresh.sgcs

    def test_measurement_noise_degrades_gracefully(self):
        cbk = R15Type2Codebook(ANT, N3=4, L=4)
        chan = RandomRayChannel(ANT, N3=4, n_rx=2)
        clean = evaluate(cbk, chan, snr_db=[10.0], n_drops=10,
                         rng=np.random.default_rng(6))
        noisy = evaluate(cbk, chan, snr_db=[10.0], n_drops=10,
                         measurement_snr_db=0.0, rng=np.random.default_rng(6))
        very_noisy = evaluate(cbk, chan, snr_db=[10.0], n_drops=10,
                              measurement_snr_db=-10.0, rng=np.random.default_rng(6))
        assert very_noisy.sgcs < noisy.sgcs < clean.sgcs
        assert noisy.sgcs > 0.5  # graceful at 0 dB measurement SNR

    def test_r18_beats_aged_r16_via_harness(self):
        """The R16-vs-R18 mobility comparison as a one-liner (plan C2.1)."""
        r16 = R16Type2Codebook(ANT, N3=4, param_combination=6)
        r18 = R18Type2Codebook(ANT, N3=4, N4=4, param_combination=7)
        kw = dict(snr_db=[10.0], n_drops=10, feedback_delay_slots=3)
        s16 = evaluate(r16, self._mobile_channel(), rng=np.random.default_rng(7), **kw)
        s18 = evaluate(r18, self._mobile_channel(), n_slots=4,
                       rng=np.random.default_rng(7), **kw)
        assert s18.sgcs > s16.sgcs

    def test_measurement_slots_noiseless_noop(self):
        """S3 knob, static + noiseless: a longer observation window changes
        nothing (the time-average of identical slots is the slot)."""
        cbk = R16Type2Codebook(ANT, N3=4, param_combination=6)
        chan = RandomRayChannel(ANT, N3=4, n_rx=2)  # static drops
        base = evaluate(cbk, chan, snr_db=[10.0], n_drops=6,
                        rng=np.random.default_rng(11))
        windowed = evaluate(cbk, chan, snr_db=[10.0], n_drops=6,
                            measurement_slots=4, rng=np.random.default_rng(11))
        assert windowed.sgcs == base.sgcs
        assert windowed.se == base.se
        assert windowed.overhead_bits == base.overhead_bits

    def test_measurement_slots_averages_noise(self):
        """S3 fairness knob: at low measurement SNR a 4-slot observation
        window lets a single-interval scheme average the estimation noise --
        the advantage R18's N4-slot window silently enjoyed."""
        cbk = R16Type2Codebook(ANT, N3=4, param_combination=6)
        chan = RandomRayChannel(ANT, N3=4, n_rx=2)
        kw = dict(snr_db=[10.0], n_drops=10, measurement_snr_db=-5.0)
        one = evaluate(cbk, chan, measurement_slots=1,
                       rng=np.random.default_rng(12), **kw)
        four = evaluate(cbk, chan, measurement_slots=4,
                        rng=np.random.default_rng(12), **kw)
        assert four.sgcs > one.sgcs + 0.02

    def test_measurement_slots_validation(self):
        r18 = R18Type2Codebook(ANT, N3=4, N4=4, param_combination=7)
        with pytest.raises(ValueError, match="measurement_slots"):
            evaluate(r18, self._mobile_channel(), snr_db=[10.0], n_drops=1,
                     n_slots=4, measurement_slots=2)

    def test_delay_aware_zero_delay_noop(self):
        r18 = R18Type2Codebook(ANT, N3=4, N4=4, param_combination=7)
        kw = dict(snr_db=[10.0], n_drops=4, n_slots=4)
        a = evaluate(r18, self._mobile_channel(), delay_aware=True,
                     rng=np.random.default_rng(13), **kw)
        b = evaluate(r18, self._mobile_channel(), rng=np.random.default_rng(13), **kw)
        assert a.sgcs == b.sgcs
        assert a.se == b.se

    def test_delay_aware_single_interval_noop(self):
        cbk = R16Type2Codebook(ANT, N3=4, param_combination=6)
        kw = dict(snr_db=[10.0], n_drops=6, feedback_delay_slots=2)
        a = evaluate(cbk, self._mobile_channel(), delay_aware=True,
                     rng=np.random.default_rng(14), **kw)
        b = evaluate(cbk, self._mobile_channel(), rng=np.random.default_rng(14), **kw)
        assert a.sgcs == b.sgcs
        assert a.se == b.se

    def test_delay_aware_recovers_r18_prediction(self):
        """S4: a delay-aware gNB indexes the *predicted* interval d + j
        instead of replaying interval j -- the harness now shows the
        prediction gain fig_05's left panel showed by hand.  On-grid
        Doppler, deterministic channel."""
        from nr_csi.channel import Ray, SyntheticRayChannel

        rays = [
            Ray(gain=1.0, m1=4, m2=2, pol_phase=0.7),
            Ray(gain=0.17, m1=8, m2=6, delay=1, doppler=1.0, pol_phase=2.1),
        ]
        chan = SyntheticRayChannel(ANT, rays, N3=4, n_rx=1, doppler_period=4)
        r18 = R18Type2Codebook(ANT, N3=4, N4=4, param_combination=7)
        kw = dict(snr_db=[10.0], n_drops=1, n_slots=4)
        d0 = evaluate(r18, chan, feedback_delay_slots=0, **kw)
        oblivious = evaluate(r18, chan, feedback_delay_slots=2, **kw)
        aware = evaluate(r18, chan, feedback_delay_slots=2, delay_aware=True, **kw)
        assert aware.sgcs > oblivious.sgcs + 0.05
        assert aware.sgcs > d0.sgcs - 0.05


class TestMuEvaluation:
    def test_zf_sum_rate_and_full_csi_gap(self):
        cbk = R15Type2Codebook(ANT, N3=4, L=4)
        chan = RandomRayChannel(ANT, N3=4, n_rx=1)
        res = evaluate_mu(cbk, chan, n_users=2, snr_db=[0.0, 10.0], n_drops=8,
                          rng=np.random.default_rng(8))
        assert res.n_users == 2
        assert res.sum_rate[0] < res.sum_rate[1]  # grows with SNR
        for got, full in zip(res.sum_rate, res.sum_rate_full_csi):
            assert 0 < got <= full + 1e-9  # quantized feedback <= full CSI

    def test_type2_beats_type1_in_mu(self):
        """Type II's raison d'etre (plan C2.2): finer co-scheduling precision
        pays off in MU-MIMO sum rate."""
        chan = RandomRayChannel(ANT, N3=4, n_rx=1)
        t1 = evaluate_mu(Type1Codebook(ANT, N3=4), chan, n_users=2,
                         snr_db=[20.0], n_drops=8, rng=np.random.default_rng(9))
        t2 = evaluate_mu(R15Type2Codebook(ANT, N3=4, L=4), chan, n_users=2,
                         snr_db=[20.0], n_drops=8, rng=np.random.default_rng(9))
        assert t2.sum_rate[0] > t1.sum_rate[0]

    def test_rzf_regularization_runs(self):
        cbk = R15Type2Codebook(ANT, N3=4, L=2)
        chan = RandomRayChannel(ANT, N3=4, n_rx=1)
        res = evaluate_mu(cbk, chan, n_users=2, snr_db=[10.0], n_drops=4,
                          rng=np.random.default_rng(10), regularization=0.1)
        assert res.sum_rate[0] > 0

    def test_rank2_mu_multiplexing_and_quantization_gap(self):
        """rank>1 MU: the gNB zero-forces across all K*rank streams and a
        user's rate sums its layers.  Full-CSI rank 2 must beat full-CSI rank 1
        (the extra stream is real multiplexing gain), and every quantized sum
        rate stays at or below its full-CSI reference."""
        chan = RandomRayChannel(ANT, N3=4, n_rx=2, n_paths=4)
        kw = dict(n_users=2, snr_db=[20.0], n_drops=12)
        r1 = evaluate_mu(R16Type2Codebook(ANT, N3=4, param_combination=6), chan,
                         rank=1, rng=np.random.default_rng(3), **kw)
        r2 = evaluate_mu(R16Type2Codebook(ANT, N3=4, param_combination=6), chan,
                         rank=2, rng=np.random.default_rng(3), **kw)
        assert r2.rank == 2
        assert r2.sum_rate_full_csi[0] > r1.sum_rate_full_csi[0]
        assert 0 < r1.sum_rate[0] <= r1.sum_rate_full_csi[0] + 1e-9
        assert 0 < r2.sum_rate[0] <= r2.sum_rate_full_csi[0] + 1e-9

    def test_rank2_type2_beats_type1(self):
        """Finer feedback unlocks higher-rank MU: at rank 2 Type II's accurate
        2-layer directions keep the inter-stream ZF clean where Type I's coarse
        ones leak (plan C2.2, sharpened at rank 2)."""
        chan = RandomRayChannel(ANT, N3=4, n_rx=2, n_paths=4)
        kw = dict(n_users=2, snr_db=[20.0], n_drops=16, rank=2)
        t1 = evaluate_mu(Type1Codebook(ANT, N3=4), chan,
                         rng=np.random.default_rng(5), **kw)
        t2 = evaluate_mu(R16Type2Codebook(ANT, N3=4, param_combination=6), chan,
                         rng=np.random.default_rng(5), **kw)
        assert t2.sum_rate[0] > t1.sum_rate[0]
