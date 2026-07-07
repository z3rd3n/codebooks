"""CSI feedback-delay semantics, pinned exactly with slot-marker stubs.

Construction: the channel's only active port *is* the absolute slot index
(H[slot] = e_slot), and the probe scheme's report interval ``s`` beams at
port ``(m - n_slots) + s`` -- the absolute slot that interval covers.  A
layer then earns log2(1 + rho) iff the harness applies it against exactly
the slot it was reported for, so the achieved SE is a closed-form function
of (measurement window, delay, delay_aware) and every window/indexing rule
in ``evaluate`` is asserted exactly, not statistically.
"""

import numpy as np
import pytest

from nr_csi.channel import RandomRayChannel
from nr_csi.codebooks import R16Type2Codebook, R18Type2Codebook
from nr_csi.codebooks.base import CodebookScheme
from nr_csi.config import AntennaConfig
from nr_csi.eval import delay_sweep, evaluate

RHO_DB = 10.0
RATE1 = float(np.log2(1 + 10 ** (RHO_DB / 10)))  # rate of one matched interval


class _SlotPortChannel:
    """H[slot, 0, 0, :] = e_slot: the active port names the absolute slot."""

    def __init__(self, n_ports: int, n3: int = 1):
        self.n_ports = n_ports
        self.n_rx = 1
        self.N3 = n3

    def generate(self, n_slots=1, rng=None):
        H = np.zeros((n_slots, self.N3, 1, self.n_ports), dtype=complex)
        for s in range(n_slots):
            H[s, :, 0, s] = 1.0
        return H


class _SlotMarkerScheme(CodebookScheme):
    """Report interval s beams at port base + s; records what select() saw."""

    name = "slot-marker"

    def __init__(self, n_ports: int, n_intervals: int, base: int, n3: int = 1):
        self.P = n_ports
        self.S = n_intervals
        self.base = base
        self.n3 = n3
        self.seen: list[np.ndarray] = []

    def select(self, H, rank=1):
        self.seen.append(np.asarray(H).copy())
        return None

    def precoder(self, pmi):
        W = np.zeros((self.S, self.n3, self.P, 1), dtype=complex)
        for s in range(self.S):
            W[s, :, self.base + s, 0] = 1.0
        return W

    def overhead_bits(self, pmi):
        return {"i": 1}


class TestWindowSemantics:
    def test_measurement_window_is_mean_of_first_m_slots(self):
        chan = _SlotPortChannel(n_ports=8)
        scheme = _SlotMarkerScheme(8, n_intervals=1, base=2)
        evaluate(scheme, chan, snr_db=[RHO_DB], rank=1, n_drops=1,
                 measurement_slots=3, rng=np.random.default_rng(0))
        expected = (np.eye(8)[0] + np.eye(8)[1] + np.eye(8)[2])[None, None, None, :] / 3
        assert np.allclose(scheme.seen[0], expected)

    def test_zero_delay_single_interval_scores_measured_slot(self):
        chan = _SlotPortChannel(n_ports=8)
        scheme = _SlotMarkerScheme(8, n_intervals=1, base=0)  # beams at slot 0
        res = evaluate(scheme, chan, snr_db=[RHO_DB], rank=1, n_drops=1,
                       rng=np.random.default_rng(0))
        assert res.se[0] == pytest.approx(RATE1)

    def test_delay_shifts_scoring_window_exactly(self):
        """Single-interval report from slot 0, scored on slot d: a stale beam
        earns exactly zero on this channel for every d > 0."""
        chan = _SlotPortChannel(n_ports=8)
        for d in (1, 2, 5):
            scheme = _SlotMarkerScheme(8, n_intervals=1, base=0)
            res = evaluate(scheme, chan, snr_db=[RHO_DB], rank=1, n_drops=1,
                           feedback_delay_slots=d, rng=np.random.default_rng(0))
            assert res.se[0] == pytest.approx(0.0, abs=1e-12)
            # ... and the scheme still measured only slot 0
            assert np.allclose(scheme.seen[0][0, 0, 0], np.eye(8)[0])

    def test_multi_interval_no_delay_aligns_interval_to_slot(self):
        """S_out == scoring slots: W[j] applied to slot j, all matched."""
        chan = _SlotPortChannel(n_ports=8)
        scheme = _SlotMarkerScheme(8, n_intervals=3, base=0)
        res = evaluate(scheme, chan, snr_db=[RHO_DB], rank=1, n_drops=1,
                       n_slots=3, rng=np.random.default_rng(0))
        assert res.se[0] == pytest.approx(RATE1)

    def test_delay_oblivious_multi_interval_misses_every_slot(self):
        """d > 0 without delay awareness: W[j] hits slot j+d, always wrong."""
        chan = _SlotPortChannel(n_ports=12)
        scheme = _SlotMarkerScheme(12, n_intervals=3, base=0)
        res = evaluate(scheme, chan, snr_db=[RHO_DB], rank=1, n_drops=1,
                       n_slots=3, feedback_delay_slots=2, rng=np.random.default_rng(0))
        assert res.se[0] == pytest.approx(0.0, abs=1e-12)

    def test_delay_aware_clamped_indexing_closed_form(self):
        """delay_aware: scoring index j uses interval min(j+d, S-1); intervals
        within the reported window match, the clamped tail does not, so
        SE = (S - d)/S * log2(1 + rho) exactly."""
        S = 4
        chan = _SlotPortChannel(n_ports=16)
        for d in (0, 1, 2, 3):
            scheme = _SlotMarkerScheme(16, n_intervals=S, base=0)
            res = evaluate(scheme, chan, snr_db=[RHO_DB], rank=1, n_drops=1,
                           n_slots=S, feedback_delay_slots=d, delay_aware=True,
                           rng=np.random.default_rng(0))
            assert res.se[0] == pytest.approx((S - d) / S * RATE1), f"delay {d}"


ANT = AntennaConfig.standard(4, 2)  # P = 16


class TestDelaySweepHelper:
    def test_static_channel_immune_to_delay(self):
        """Matched seeding + a static channel: aging is a no-op, so every
        delay must reproduce the zero-delay result bit-for-bit."""
        chan = RandomRayChannel(ANT, N3=4, n_rx=2, n_paths=4, max_doppler=0.0)
        cbk = R16Type2Codebook(ANT, N3=4, param_combination=4)
        out = delay_sweep(cbk, chan, delays=(0, 2, 4), seed=7,
                          snr_db=[10.0], rank=1, n_drops=3)
        assert list(out.keys()) == [0, 2, 4]
        for d in (2, 4):
            assert out[d].se == out[0].se
            assert out[d].sgcs == out[0].sgcs

    def test_rejects_conflicting_kwargs(self):
        chan = RandomRayChannel(ANT, N3=4, n_rx=2)
        cbk = R16Type2Codebook(ANT, N3=4, param_combination=4)
        with pytest.raises(ValueError, match="delay_sweep"):
            delay_sweep(cbk, chan, delays=(0,), feedback_delay_slots=1)
        with pytest.raises(ValueError, match="delay_sweep"):
            delay_sweep(cbk, chan, delays=(0,), rng=np.random.default_rng(0))

    def test_aging_degrades_static_codebook_on_fast_channel(self):
        chan = RandomRayChannel(ANT, N3=4, n_rx=2, n_paths=4,
                                max_doppler=1.0, doppler_period=8)
        cbk = R16Type2Codebook(ANT, N3=4, param_combination=4)
        out = delay_sweep(cbk, chan, delays=(0, 2), seed=0,
                          snr_db=[10.0], rank=1, n_drops=8)
        assert out[2].se[0] < out[0].se[0]
        assert out[2].sgcs < out[0].sgcs

    def test_r18_doppler_beats_static_r16_under_delay(self):
        """The R18 codebook's raison d'etre, now actually exercised: under
        CSI aging on a fast channel, delay-aware R18 prediction must beat the
        static R16 report; at zero delay they are near-equivalent."""
        chan = RandomRayChannel(ANT, N3=4, n_rx=2, n_paths=4,
                                max_doppler=1.0, doppler_period=8)
        r16 = R16Type2Codebook(ANT, N3=4, param_combination=6)
        r18 = R18Type2Codebook(ANT, N3=4, N4=4, param_combination=7)
        kw = dict(snr_db=[10.0], rank=1, n_drops=10)
        d = 2
        out16 = delay_sweep(r16, chan, delays=(0, d), seed=1,
                            measurement_slots=4, **kw)
        out18 = delay_sweep(r18, chan, delays=(0, d), seed=1,
                            n_slots=4, delay_aware=True, **kw)
        assert out18[d].sgcs > out16[d].sgcs + 0.05
        assert out18[d].se[0] > out16[d].se[0]
