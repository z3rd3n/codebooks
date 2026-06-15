"""Channel diagnostics validated against the synthetic ray channel.

The synthetic channel places rays at known delays / Dopplers / angles, so each
diagnostic has a closed-form expected value -- which checks both the diagnostic
and the channel's frequency/time wiring without needing Sionna.
"""

import numpy as np
import pytest

from nr_csi.channel.diagnostics import (
    coherence_lag,
    freq_correlation,
    power_delay_profile,
    rms_delay_spread,
    singular_value_spectrum,
    spatial_covariance_spectrum,
    taps_to_seconds,
    time_correlation,
)
from nr_csi.channel.synthetic import Ray, SyntheticRayChannel
from nr_csi.config import AntennaConfig

ANT = AntennaConfig.standard(4, 2)
N3 = 16


def _chan(rays, n_rx=1, doppler_period=1):
    return SyntheticRayChannel(ANT, rays, N3=N3, n_rx=n_rx, doppler_period=doppler_period)


def test_pdp_impulse_at_ray_delay():
    # single on-grid ray at delay tap 3 -> clean impulse; the causal (ifft) PDP
    # mirrors the framework's conjugate tap convention, so the peak is at N3-3.
    H = _chan([Ray(gain=1.0, m1=0.0, delay=3.0)]).generate(n_slots=1)
    pdp = power_delay_profile(H)
    assert pdp.shape == (N3,)
    assert np.argmax(pdp) == (N3 - 3) % N3
    assert pdp.max() > 0.999  # all energy in one tap
    assert rms_delay_spread(H) < 1e-6  # a single tap has zero spread


def test_pdp_two_taps_and_rms():
    # equal-gain rays 4 taps apart -> two impulses; circular RMS spread ~ 2 taps
    H = _chan([Ray(1.0, m1=0.0, delay=0.0), Ray(1.0, m1=1.0, delay=4.0)]).generate()
    pdp = power_delay_profile(H)
    peaks = np.argsort(pdp)[-2:]
    assert set(peaks.tolist()) == {(N3 - 0) % N3, (N3 - 4) % N3}  # {0, 12}
    # two equal taps a distance 4 apart: circular std ~ 2.1 taps
    assert rms_delay_spread(H) == pytest.approx(2.1, abs=0.2)


def test_larger_delay_spread_decorrelates_faster_in_frequency():
    narrow = _chan([Ray(1.0, m1=0.0, delay=0.0), Ray(1.0, m1=1.0, delay=1.0)]).generate()
    wide = _chan([Ray(1.0, m1=0.0, delay=0.0), Ray(1.0, m1=1.0, delay=6.0)]).generate()
    rho_n, rho_w = freq_correlation(narrow), freq_correlation(wide)
    assert rho_n[0] == pytest.approx(1.0)
    assert rho_w[0] == pytest.approx(1.0)
    # wider delay spread -> shorter coherence bandwidth
    assert coherence_lag(rho_w) < coherence_lag(rho_n)


def test_freq_correlation_unity_for_single_ray():
    # a single ray is flat-fading across the PMI grid: |rho| == 1 at every lag
    H = _chan([Ray(1.0, m1=0.0, delay=2.0)]).generate()
    assert np.allclose(freq_correlation(H), 1.0, atol=1e-9)


def test_time_correlation_decays_with_doppler_spread():
    # two rays with the same Doppler -> coherent in time; spread Dopplers -> decay
    period = 8
    one = _chan([Ray(1.0, m1=0.0, doppler=1.0)], doppler_period=period).generate(n_slots=period)
    spread = _chan(
        [Ray(1.0, m1=0.0, doppler=0.0), Ray(1.0, m1=1.0, doppler=3.0)],
        doppler_period=period,
    ).generate(n_slots=period)
    rho_one, rho_spread = time_correlation(one), time_correlation(spread)
    assert rho_one[0] == pytest.approx(1.0)
    assert np.allclose(rho_one, 1.0, atol=1e-9)  # single Doppler = constant magnitude
    assert coherence_lag(rho_spread) < coherence_lag(rho_one)


def test_time_correlation_trivial_without_slots():
    H = _chan([Ray(1.0, m1=0.0)]).generate(n_slots=1)
    assert time_correlation(H).shape == (1,)


def test_svd_spectrum_rank_one_for_single_ray_single_rx():
    # one ray, one rx -> rank-1 channel: all energy in the first singular value
    H = _chan([Ray(1.0, m1=1.0)], n_rx=1).generate()
    spec = singular_value_spectrum(H)
    assert spec[0] == pytest.approx(1.0, abs=1e-9)


def test_svd_spectrum_richer_for_multipath():
    # NB: SyntheticRayChannel defaults a_rx = ones for every ray, so distinct
    # rays still share one receive signature (rank 1).  Give each ray its own
    # a_rx to build a genuinely higher-rank channel.
    rng = np.random.default_rng(0)
    los = _chan([Ray(1.0, m1=1.0, a_rx=rng.standard_normal(4))], n_rx=4).generate()
    rich = _chan(
        [Ray(1.0, m1=float(i), a_rx=rng.standard_normal(4) + 1j * rng.standard_normal(4))
         for i in range(6)],
        n_rx=4,
    ).generate()
    # the dominant eigenvalue fraction is larger for the rank-1 LoS-like channel
    assert singular_value_spectrum(los)[0] == pytest.approx(1.0, abs=1e-9)
    assert singular_value_spectrum(rich)[0] < 0.95


def test_spatial_covariance_concentrates_for_single_direction():
    # one fixed beam direction across all realisations -> rank-1 spatial
    # covariance (lambda1 ~ 1); many distinct fixed directions -> spread out.
    one = _chan([Ray(1.0, m1=2.0, m2=1.0)], n_rx=2).generate(n_slots=4)
    # distinct delays make each direction appear with a different per-tap weight,
    # so the covariance spans all the rays' directions (rather than collapsing to
    # one fixed superposition when every ray shares delay 0 and a_rx = ones)
    many = _chan(
        [Ray(1.0, m1=float(i), m2=float(i % 2), delay=float(i)) for i in range(8)], n_rx=2
    ).generate(n_slots=4)
    s_one = spatial_covariance_spectrum(one)
    s_many = spatial_covariance_spectrum(many)
    assert s_one.shape == (ANT.P,)
    assert s_one[0] == pytest.approx(1.0, abs=1e-6)  # single direction = rank 1
    assert s_many[0] < 0.9  # richer angular spread spreads the eigenvalues
    assert s_one[0] > s_many[0]


def test_taps_to_seconds():
    # one tap of a 256-pt / 30 kHz grid = 1/(256*30e3) ~= 130.2 ns
    assert taps_to_seconds(1, 256, 30e3) == pytest.approx(130.2e-9, rel=1e-3)


def test_coherence_lag_interpolates_and_saturates():
    assert coherence_lag(np.array([1.0, 1.0, 1.0])) == 3.0  # never crosses
    # crosses between lag 1 (0.6) and lag 2 (0.4): midpoint -> 1.5
    assert coherence_lag(np.array([1.0, 0.6, 0.4])) == pytest.approx(1.5)
