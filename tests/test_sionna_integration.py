"""End-to-end 38.901 CDL integration (requires the optional sionna extra)."""

import numpy as np
import pytest

pytest.importorskip("sionna")
pytestmark = pytest.mark.sionna

from nr_csi.channel.sionna_adapter import SionnaCDLChannel
from nr_csi.codebooks.etype2_r16 import R16Type2Codebook
from nr_csi.codebooks.etype2_r18 import R18Type2Codebook
from nr_csi.codebooks.fetype2_r17 import R17Type2Codebook
from nr_csi.codebooks.type1 import Type1Codebook
from nr_csi.codebooks.type2_r15 import R15Type2Codebook
from nr_csi.config import AntennaConfig
from nr_csi.eval import evaluate
from nr_csi.metrics import sgcs

ANT = AntennaConfig.standard(4, 2)  # P = 16
N3 = 8


@pytest.fixture(scope="module")
def channel():
    return SionnaCDLChannel(ANT, N3=N3, model="A", n_rx=2, ue_speed_kmh=3.0)


@pytest.fixture(scope="module")
def schemes():
    return [
        Type1Codebook(ANT, N3=N3),
        R15Type2Codebook(ANT, N3=N3, L=4),
        R15Type2Codebook(ANT, N3=N3, L=4, port_selection=True, d=1),
        R16Type2Codebook(ANT, N3=N3, param_combination=4),
        R16Type2Codebook(ANT, N3=N3, param_combination=4, port_selection=True, d=1),
        R17Type2Codebook(ANT, N3=N3, param_combination=5),
        R18Type2Codebook(ANT, N3=N3, N4=1, param_combination=3),
    ]


def test_all_codebooks_run_end_to_end(channel, schemes):
    H = channel.generate(n_slots=1)
    assert H.shape == (1, N3, 2, ANT.P)
    for scheme in schemes:
        pmi = scheme.select(H, rank=1)
        W = scheme.precoder(pmi)
        assert np.all(np.isfinite(W))
        for t in range(N3):
            assert np.isclose(np.linalg.norm(W[0, t]), 1.0, atol=1e-6), scheme.name
        assert scheme.total_overhead_bits(pmi) > 0


def test_se_monotone_in_snr_and_beats_random(channel):
    cbk = R16Type2Codebook(ANT, N3=N3, param_combination=4)
    res = evaluate(cbk, channel, snr_db=[0.0, 10.0, 20.0], rank=1, n_drops=5)
    assert res.se[0] < res.se[1] < res.se[2]
    assert all(c <= u + 1e-9 for c, u in zip(res.se, res.se_upper_bound))
    assert res.sgcs > 0.5  # CDL-A is reasonably sparse; eType II should track it


def test_rank2_runs(channel):
    cbk = R16Type2Codebook(ANT, N3=N3, param_combination=4)
    H = channel.generate(n_slots=1)
    pmi = cbk.select(H, rank=2)
    W = cbk.precoder(pmi)
    assert W.shape == (1, N3, ANT.P, 2)


def test_r18_tracks_mobility_better_than_held_r16():
    """At UE speed, one R18 predicted PMI beats a held R16 report."""
    N4 = 4
    fast = SionnaCDLChannel(
        ANT, N3=N3, model="A", n_rx=2, ue_speed_kmh=30.0, interval_duration=5e-3
    )
    r18 = R18Type2Codebook(ANT, N3=N3, N4=N4, param_combination=5)
    r16 = R16Type2Codebook(ANT, N3=N3, param_combination=5)
    from nr_csi.baselines import eigen_precoder

    s18_all, s16_all = [], []
    for _ in range(5):
        H = fast.generate(n_slots=N4)
        targets = eigen_precoder(H, rank=1)
        W18 = r18.precoder(r18.select(H, rank=1))
        W16 = r16.precoder(r16.select(H[:1], rank=1))  # first interval, held
        s18_all.append(np.mean([sgcs(targets[i], W18[i]) for i in range(N4)]))
        s16_all.append(np.mean([sgcs(targets[i], W16[0]) for i in range(N4)]))
    assert np.mean(s18_all) > np.mean(s16_all)
