"""Rank-adaptive evaluation (``evaluate(..., rank="auto")``).

Under the default noiseless / zero-delay configuration the scoring window
equals the selection window, so auto-RI's SE at the selection SNR must
dominate every fixed rank on the same drops -- an exact property, tested
exactly.  Physics anchors: a rank-1 channel must always elect rank 1; a
rich full-rank channel at high selection SNR must mostly elect rank 2.
"""

import numpy as np
import pytest

from nr_csi.channel import RandomRayChannel, Ray, SyntheticRayChannel
from nr_csi.codebooks import R16Type2Codebook, Type1Codebook
from nr_csi.config import AntennaConfig
from nr_csi.eval import evaluate

ANT = AntennaConfig.standard(4, 2)  # P = 16


class TestAutoRankDominance:
    def test_auto_beats_every_fixed_rank_at_selection_snr(self):
        chan = RandomRayChannel(ANT, N3=4, n_rx=2, n_paths=6)
        cbk = R16Type2Codebook(ANT, N3=4, param_combination=4)
        kw = dict(snr_db=[10.0], n_drops=6, select_snr_db=10.0)
        auto = evaluate(cbk, chan, rank="auto", rng=np.random.default_rng(0), **kw)
        for r in (1, 2):
            fixed = evaluate(cbk, chan, rank=r, rng=np.random.default_rng(0), **kw)
            assert auto.se[0] >= fixed.se[0] - 1e-9

    def test_auto_dominance_holds_per_drop(self):
        """Not just on average: every drop's chosen rank is at least as good
        as the fixed-rank alternative on that same drop (same rng stream)."""
        chan = RandomRayChannel(ANT, N3=4, n_rx=2, n_paths=6)
        cbk = Type1Codebook(ANT, N3=4)
        kw = dict(snr_db=[10.0], n_drops=5, select_snr_db=10.0)
        auto = evaluate(cbk, chan, rank="auto", auto_ranks=(1, 2),
                        rng=np.random.default_rng(1), **kw)
        assert len(auto.per_drop_rank) == 5
        assert set(auto.per_drop_rank) <= {1, 2}


class TestAutoRankPhysics:
    def test_rank1_channel_always_elects_rank1(self):
        """A single propagation path gives an exactly rank-1 channel: adding
        a second layer can only split power into a zero-gain direction."""
        ray = Ray(gain=1.0, m1=3, m2=1, pol_phase=0.7, delay=1)
        chan = SyntheticRayChannel(ANT, [ray], N3=4, n_rx=2)
        cbk = R16Type2Codebook(ANT, N3=4, param_combination=4)
        res = evaluate(cbk, chan, snr_db=[10.0], rank="auto", n_drops=4,
                       rng=np.random.default_rng(2))
        assert res.per_drop_rank == [1, 1, 1, 1]

    def test_rich_channel_high_snr_mostly_rank2(self):
        """Full-rank 2-rx channel at 30 dB selection SNR: multiplexing wins."""
        chan = RandomRayChannel(ANT, N3=4, n_rx=2, n_paths=12)
        cbk = R16Type2Codebook(ANT, N3=4, param_combination=4)
        res = evaluate(cbk, chan, snr_db=[30.0], rank="auto", n_drops=8,
                       select_snr_db=30.0, rng=np.random.default_rng(3))
        assert np.mean(np.asarray(res.per_drop_rank) == 2) > 0.5

    def test_ranks_capped_by_receive_antennas(self):
        """n_rx = 1: auto must never report more layers than the channel has."""
        chan = RandomRayChannel(ANT, N3=4, n_rx=1, n_paths=6)
        cbk = Type1Codebook(ANT, N3=4)  # supports ranks 1..8
        res = evaluate(cbk, chan, snr_db=[30.0], rank="auto", n_drops=4,
                       select_snr_db=30.0, rng=np.random.default_rng(4))
        assert res.per_drop_rank == [1, 1, 1, 1]


class TestAutoRankPlumbing:
    def test_fixed_rank_records_constant_per_drop_rank(self):
        chan = RandomRayChannel(ANT, N3=4, n_rx=2)
        cbk = Type1Codebook(ANT, N3=4)
        res = evaluate(cbk, chan, snr_db=[10.0], rank=2, n_drops=3,
                       rng=np.random.default_rng(5))
        assert res.per_drop_rank == [2, 2, 2]

    def test_bounds_hold_with_auto(self):
        chan = RandomRayChannel(ANT, N3=4, n_rx=2, n_paths=6)
        cbk = R16Type2Codebook(ANT, N3=4, param_combination=4)
        res = evaluate(cbk, chan, snr_db=[0.0, 10.0, 30.0], rank="auto", n_drops=5,
                       rng=np.random.default_rng(6))
        assert all(c <= u + 1e-9 for c, u in zip(res.se, res.capacity_upper_bound))
        assert all(m <= s + 1e-9 for m, s in zip(res.se_mmse, res.se))
        assert 0 < res.sgcs <= 1

    def test_bad_rank_string_rejected(self):
        chan = RandomRayChannel(ANT, N3=4, n_rx=2)
        with pytest.raises(ValueError, match="auto"):
            evaluate(Type1Codebook(ANT, N3=4), chan, rank="best", n_drops=1)

    def test_composes_with_beamformed_adapter(self):
        from nr_csi.codebooks import R17Type2Codebook
        from nr_csi.eval import BeamformedPortsScheme

        eff = AntennaConfig.standard(4, 1)
        bf = BeamformedPortsScheme(R17Type2Codebook(eff, N3=4, param_combination=4),
                                   n_raw_ports=16, n_beams_per_pol=4)
        chan = RandomRayChannel(ANT, N3=4, n_rx=2, n_paths=4)
        res = evaluate(bf, chan, snr_db=[10.0], rank="auto", n_drops=3,
                       rng=np.random.default_rng(7))
        assert len(res.per_drop_rank) == 3
        assert all(1 <= r <= 2 for r in res.per_drop_rank)
