"""End-to-end 38.901 CDL integration (requires the optional sionna extra)."""

import numpy as np
import pytest

pytest.importorskip("sionna")
pytestmark = pytest.mark.sionna

from nr_csi.channel.sionna_adapter import SionnaCDLChannel
from nr_csi.codebooks.etype2_r16 import R16Type2Codebook
from nr_csi.codebooks.etype2_r18 import R18Type2Codebook
from nr_csi.codebooks.fetype2_r17 import R17Type2Codebook
from nr_csi.codebooks.refined_type1mp_r19 import RefinedType1MultiPanelCodebook
from nr_csi.codebooks.type1 import Type1Codebook
from nr_csi.codebooks.type1_multipanel import Type1MultiPanelCodebook
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


def test_los_beamforming_gain_approaches_full_array_gain():
    """Near-LoS CDL-D: Type I must approach the full array gain P over a
    single antenna.  This only happens if the DFT beams align with the
    channel's spatial phase ramp, i.e. the whole port pipeline (permutation,
    polarization split, co-phasing) is oriented correctly."""
    chan = SionnaCDLChannel(
        ANT, N3=4, model="D", delay_spread=10e-9, n_rx=1, ue_speed_kmh=0.0
    )
    cbk = Type1Codebook(ANT, N3=4)
    gains = []
    for _ in range(3):
        H = chan.generate(n_slots=1)  # (1, N3, 1, P)
        pmi = cbk.select(H, rank=1)
        W = cbk.precoder(pmi)
        bf = np.mean(np.abs(np.einsum("trp,tpl->trl", H[0], W[0])) ** 2)
        single = np.mean(np.abs(H[0]) ** 2)  # per-port (single-antenna) power
        gains.append(bf / single)
    # full array gain is P = 2*N1*N2; allow ~3 dB for off-grid beam mismatch
    assert np.mean(gains) > 0.5 * ANT.P


def test_port_permutation_matches_panel_geometry():
    """Our port k = pol*N1*N2 + n1*N2 + n2 must land on the antenna at
    (y0 + n1*dy, z0 + n2*dz) of the right polarization.  Verified directly
    against the panel's antenna positions, so an orientation error (n1/n2
    swapped, wrong fastest axis, interleaved polarizations) fails even though
    CDL's near-broadside LoS would mask it in a beamforming-gain check."""
    chan = SionnaCDLChannel(ANT, N3=4, model="D", n_rx=1)
    arr = chan.bs_array
    pos = np.asarray(arr.ant_pos)
    perm = chan._port_perm
    n_half = ANT.N1 * ANT.N2
    pol_sets = [set(np.asarray(arr.ant_ind_pol1).ravel().tolist()),
                set(np.asarray(arr.ant_ind_pol2).ravel().tolist())]
    ys = np.unique(np.round(pos[:, 1], 9))
    zs = np.unique(np.round(pos[:, 2], 9))
    assert len(ys) == ANT.N1 and len(zs) == ANT.N2
    for pol in (0, 1):
        for n1 in range(ANT.N1):
            for n2 in range(ANT.N2):
                k = pol * n_half + n1 * ANT.N2 + n2
                ant = perm[k]
                assert ant in pol_sets[pol], f"port {k} not in polarization {pol}"
                assert np.isclose(pos[ant, 1], ys[n1]), f"port {k}: wrong column"
                assert np.isclose(pos[ant, 2], zs[n2]), f"port {k}: wrong row"


def test_multipanel_port_permutation_is_panel_major():
    """For Ng>1, port k = panel*2*N1*N2 + pol*N1*N2 + n1*N2+n2 (panel-major,
    then polarization-major -- the layout Type1MultiPanelCodebook/
    RefinedType1MultiPanelCodebook build). Each panel's block must land on a
    physically distinct, non-overlapping group of antennas of the right
    polarization and (n1, n2) position within that panel."""
    ant = AntennaConfig.standard(2, 2, Ng=2)
    chan = SionnaCDLChannel(ant, N3=4, model="D", n_rx=1)
    pos = np.asarray(chan.bs_array.ant_pos)
    perm = chan._port_perm
    npp = ant.N1 * ant.N2
    pol_sets = [
        set(np.asarray(chan.bs_array.ant_ind_pol1).ravel().tolist()),
        set(np.asarray(chan.bs_array.ant_ind_pol2).ravel().tolist()),
    ]
    panel_y_ranges = []
    for panel in range(ant.Ng):
        block = perm[panel * 2 * npp : (panel + 1) * 2 * npp]
        panel_y_ranges.append((pos[block, 1].min(), pos[block, 1].max()))
        for pol in (0, 1):
            pol_block = block[pol * npp : (pol + 1) * npp]
            assert all(a in pol_sets[pol] for a in pol_block)
        ys = np.unique(np.round(pos[block[:npp], 1], 9))
        zs = np.unique(np.round(pos[block[:npp], 2], 9))
        assert len(ys) == ant.N1 and len(zs) == ant.N2
    # panels must not spatially overlap
    (lo0, hi0), (lo1, hi1) = panel_y_ranges
    assert hi0 < lo1 or hi1 < lo0


def test_multipanel_codebooks_run_end_to_end():
    """Previously: PanelArray was built with no Ng, so P didn't match the
    channel's port count and generation raised deep inside Sionna/einsum."""
    ant1 = AntennaConfig.standard(2, 2, Ng=2)  # R15 Table 5.2.2.2.2-1
    ant2 = AntennaConfig.standard(4, 3, Ng=2)  # R19 Table 5.2.2.2.2a-1
    for ant, cbk in (
        (ant1, Type1MultiPanelCodebook(ant1, N3=4, mode=1)),
        (ant2, RefinedType1MultiPanelCodebook(ant2, N3=4)),
    ):
        chan = SionnaCDLChannel(ant, N3=4, model="A", n_rx=2, ue_speed_kmh=3.0)
        H = chan.generate(n_slots=1)
        assert H.shape == (1, 4, 2, ant.P)
        pmi = cbk.select(H, rank=2)
        W = cbk.precoder(pmi)
        assert np.all(np.isfinite(W))
        assert W.shape == (1, 4, ant.P, 2)


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


def test_tabcss_wired_subband_mapping():
    """BWP-driven grid mapping: 26 RB with subband size 8 -> N3 = 4 PMI
    units, RB-aligned (the last subband is short: 2 RB)."""
    chan = SionnaCDLChannel(ANT, model="A", n_rx=1, n_rb=26, subband_size=8,
                            fft_size=512)
    assert chan.N3 == 4
    slices = chan._unit_slices
    assert [s.start for s in slices] == [0, 96, 192, 288]
    assert slices[-1].stop == 26 * 12
    H = chan.generate(n_slots=1)
    assert H.shape == (1, 4, 1, ANT.P)
    assert np.all(np.isfinite(H))


def test_tabcss_rejects_invalid_combination():
    with pytest.raises(ValueError):
        SionnaCDLChannel(ANT, model="A", n_rb=100, subband_size=4, fft_size=2048)
    with pytest.raises(ValueError):  # 273 RB do not fit a 256-point grid
        SionnaCDLChannel(ANT, model="A", n_rb=273, subband_size=16)


def test_larger_delay_spread_increases_frequency_selectivity():
    """Correctness of the frequency/delay wiring: a larger configured delay
    spread must shorten the coherence bandwidth on the PMI grid."""
    from nr_csi.channel.diagnostics import coherence_lag, freq_correlation, rms_delay_spread

    narrow = SionnaCDLChannel(ANT, N3=32, model="C", n_rx=2, delay_spread=30e-9)
    wide = SionnaCDLChannel(ANT, N3=32, model="C", n_rx=2, delay_spread=300e-9)
    Hn = np.stack([narrow.generate(n_slots=1) for _ in range(8)])
    Hw = np.stack([wide.generate(n_slots=1) for _ in range(8)])
    assert rms_delay_spread(Hw) > rms_delay_spread(Hn)
    assert coherence_lag(freq_correlation(Hw)) < coherence_lag(freq_correlation(Hn))


def test_ue_speed_increases_temporal_decorrelation():
    """Correctness of the Doppler/time wiring: a faster UE must shorten the
    temporal coherence (slots) at a fixed slot interval."""
    from nr_csi.channel.diagnostics import coherence_lag, time_correlation

    slow = SionnaCDLChannel(ANT, N3=8, model="C", n_rx=2, ue_speed_kmh=3.0,
                            interval_duration=2e-3)
    fast = SionnaCDLChannel(ANT, N3=8, model="C", n_rx=2, ue_speed_kmh=120.0,
                            interval_duration=2e-3)
    Hs = np.stack([slow.generate(n_slots=12) for _ in range(8)])
    Hf = np.stack([fast.generate(n_slots=12) for _ in range(8)])
    assert coherence_lag(time_correlation(Hf)) < coherence_lag(time_correlation(Hs))
