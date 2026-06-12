"""Statistical tests for the paper's qualitative comparison table (plan A7):
precision ordering, overhead-vs-precision trade, mobility, and the
applicable-scenario boundary between regular and port-selection codebooks."""

import numpy as np
import pytest

from nr_csi.baselines.ideal import eigen_precoder
from nr_csi.channel import RandomRayChannel
from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
)
from nr_csi.config import AntennaConfig
from nr_csi.metrics.similarity import sgcs
from nr_csi.utils import dft

pytestmark = pytest.mark.slow

ANT = AntennaConfig.standard(4, 2)  # P = 16
N3 = 8
N_DROPS = 15


def mean_sgcs_bits(cbk, chan, seed, n_drops=N_DROPS, transform=None):
    rng = np.random.default_rng(seed)
    ss, bb = [], []
    for _ in range(n_drops):
        H = chan.generate(n_slots=1, rng=rng)
        if transform is not None:
            H = transform(H)
        pmi = cbk.select(H, rank=1)
        W = cbk.precoder(pmi)
        ss.append(sgcs(eigen_precoder(H, rank=1), W))
        bb.append(cbk.total_overhead_bits(pmi))
    return float(np.mean(ss)), float(np.mean(bb))


class TestPrecisionRow:
    def test_type2_beats_type1(self):
        """'High' vs 'Medium' precision: beam combination beats single-beam."""
        chan = RandomRayChannel(ANT, N3=N3, n_rx=2)
        s1, _ = mean_sgcs_bits(Type1Codebook(ANT, N3=N3), chan, seed=10)
        s2, _ = mean_sgcs_bits(R15Type2Codebook(ANT, N3=N3, L=4), chan, seed=10)
        assert s2 > s1 + 0.1


class TestOverheadRow:
    def test_r16_cheaper_than_r15_at_comparable_precision(self):
        """'Medium' vs 'High' overhead: the FD compression saves bits without
        giving up SGCS (within +-0.05) on delay-sparse channels."""
        chan = RandomRayChannel(ANT, N3=N3, n_rx=2)
        s15, b15 = mean_sgcs_bits(R15Type2Codebook(ANT, N3=N3, L=4), chan, seed=11)
        s16, b16 = mean_sgcs_bits(
            R16Type2Codebook(ANT, N3=N3, param_combination=6), chan, seed=11
        )
        assert abs(s16 - s15) < 0.05
        # rank-1, N3=8 is the modest end of the saving; the paper's 10x claim
        # (rank 2, N3=18, QPSK R15) is asserted nominally in test_overhead.py
        assert b16 < 0.8 * b15


class TestMobilityRow:
    @pytest.mark.parametrize("seed", [20, 21, 22])
    def test_only_r18_tracks_future_intervals(self, seed):
        """Under Doppler, one R18 predicted report keeps future-interval SGCS
        above the static codebooks' held reports."""
        N4 = 4
        chan = RandomRayChannel(
            ANT, N3=N3, n_rx=1, max_doppler=1.0, doppler_period=N4
        )
        rng = np.random.default_rng(seed)
        r18 = R18Type2Codebook(ANT, N3=N3, N4=N4, param_combination=7)
        r16 = R16Type2Codebook(ANT, N3=N3, param_combination=6)
        r15 = R15Type2Codebook(ANT, N3=N3, L=4)
        s18, s16, s15 = [], [], []
        for _ in range(8):
            H = chan.generate(n_slots=N4, rng=rng)
            targets = eigen_precoder(H, rank=1)
            W18 = r18.precoder(r18.select(H, rank=1))
            W16 = r16.precoder(r16.select(H[:1], rank=1))
            W15 = r15.precoder(r15.select(H[:1], rank=1))
            future = range(1, N4)
            s18.append(np.mean([sgcs(targets[s], W18[s]) for s in future]))
            s16.append(np.mean([sgcs(targets[s], W16[0]) for s in future]))
            s15.append(np.mean([sgcs(targets[s], W15[0]) for s in future]))
        assert np.mean(s18) > np.mean(s16)
        assert np.mean(s18) > np.mean(s15)


class TestApplicableScenarioRow:
    """Regular codebooks live in the antenna domain, port-selection codebooks
    in the (post-PEB) beam domain -- each wins on its home turf."""

    def _beam_domain(self, H):
        half = ANT.P // 2
        F = dft.orthogonal_group(ANT, 0, 0).T  # full-connect DFT PEB
        return np.concatenate([H[..., :half] @ F, H[..., half:] @ F], axis=-1)

    def test_regular_wins_in_antenna_domain(self):
        chan = RandomRayChannel(ANT, N3=N3, n_rx=2)
        r16 = R16Type2Codebook(ANT, N3=N3, param_combination=6)
        r17 = R17Type2Codebook(ANT, N3=N3, param_combination=6)
        s_reg, _ = mean_sgcs_bits(r16, chan, seed=12)
        s_ps, _ = mean_sgcs_bits(r17, chan, seed=12)
        assert s_reg > s_ps + 0.1

    def test_port_selection_wins_in_beam_domain(self):
        # 2 paths: the PEB concentrates the channel on few ports, which is
        # the regime port selection is designed for
        chan = RandomRayChannel(ANT, N3=N3, n_rx=2, n_paths=2)
        r16 = R16Type2Codebook(ANT, N3=N3, param_combination=6)
        r17 = R17Type2Codebook(ANT, N3=N3, param_combination=6)
        s_reg, _ = mean_sgcs_bits(r16, chan, seed=12, transform=self._beam_domain)
        s_ps, _ = mean_sgcs_bits(r17, chan, seed=12, transform=self._beam_domain)
        assert s_ps > s_reg + 0.1
