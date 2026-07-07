"""Per-drop/per-user MU statistics and the stats helpers behind the
MU Pareto methodology (mean CI over drops + pooled cell-edge percentile)."""

import numpy as np
import pytest

from nr_csi.channel import RandomRayChannel
from nr_csi.codebooks import R15Type2Codebook
from nr_csi.codebooks.base import CodebookScheme
from nr_csi.config import AntennaConfig
from nr_csi.eval import bootstrap_ci, edge_rate, evaluate_mu, mean_ci
from nr_csi.metrics import su_rate

ANT = AntennaConfig.standard(4, 2)  # P = 16


class TestPerDropUserRates:
    def _res(self, n_users=2, n_drops=5, rank=1):
        cbk = R15Type2Codebook(ANT, N3=4, L=2)
        chan = RandomRayChannel(ANT, N3=4, n_rx=max(1, rank))
        return evaluate_mu(cbk, chan, n_users=n_users, snr_db=[0.0, 10.0],
                           n_drops=n_drops, rank=rank, rng=np.random.default_rng(0))

    def test_shape_is_drops_snr_users(self):
        res = self._res(n_users=3, n_drops=4)
        arr = np.asarray(res.per_drop_user_rates)
        assert arr.shape == (4, 2, 3)
        arr_full = np.asarray(res.per_drop_user_rates_full_csi)
        assert arr_full.shape == (4, 2, 3)

    def test_mean_over_drops_reproduces_sum_rate(self):
        """sum_rate must be exactly the mean over drops of the per-user sums --
        the samples and the summary are the same numbers."""
        res = self._res(n_users=2, n_drops=6)
        arr = np.asarray(res.per_drop_user_rates)  # (drops, snr, K)
        assert np.allclose(arr.sum(-1).mean(0), res.sum_rate, atol=1e-12)
        arr_full = np.asarray(res.per_drop_user_rates_full_csi)
        assert np.allclose(arr_full.sum(-1).mean(0), res.sum_rate_full_csi, atol=1e-12)

    def test_rates_nonnegative(self):
        res = self._res(n_users=2, n_drops=4, rank=2)
        assert np.all(np.asarray(res.per_drop_user_rates) >= 0)

    def test_overhead_bits_reported(self):
        res = self._res(n_users=2, n_drops=4)
        assert res.overhead_bits > 0


class _FixedDirectionScheme(CodebookScheme):
    """Reports a fixed (unnormalized) direction regardless of the channel."""

    name = "fixed-direction"

    def __init__(self, w: np.ndarray, n3: int):
        self._w = w  # (P,)
        self._n3 = n3

    def select(self, H, rank=1):
        return None

    def precoder(self, pmi):
        w = self._w / np.linalg.norm(self._w)
        return np.tile(w[None, None, :, None], (1, self._n3, 1, 1))

    def overhead_bits(self, pmi):
        return {"i": 4}


class _FixedChannel:
    """Deterministic single-slot channel, same H every drop."""

    def __init__(self, H: np.ndarray):
        self._H = H  # (1, N3, Nr, P)
        self.N3 = H.shape[1]
        self.n_rx = H.shape[2]
        self.n_ports = H.shape[3]

    def generate(self, n_slots=1, rng=None):
        return self._H.copy()


class TestSingleUserIdentity:
    def test_k1_v1_zf_reduces_to_su_rate_of_reported_beam(self):
        """K = 1, v = 1: ZF across one direction is that direction; the MU sum
        rate must equal su_rate of the reported beam at full power.  Pins the
        beam normalization and the rho/(K*v) power split end to end."""
        rng = np.random.default_rng(1)
        P, N3 = 8, 4
        H = rng.standard_normal((1, N3, 2, P)) + 1j * rng.standard_normal((1, N3, 2, P))
        w = rng.standard_normal(P) + 1j * rng.standard_normal(P)
        scheme = _FixedDirectionScheme(w, N3)
        chan = _FixedChannel(H)
        res = evaluate_mu(scheme, chan, n_users=1, snr_db=[10.0], n_drops=2)
        W = scheme.precoder(None)[0]  # (N3, P, 1), unit columns
        assert np.isclose(res.sum_rate[0], su_rate(H[0], W, 10.0), atol=1e-9)
        # single user, no interference: quantized == "full CSI" structure holds
        assert res.sum_rate[0] <= res.sum_rate_full_csi[0] + 1e-9


class TestMeanCi:
    def test_constant_samples_zero_width(self):
        m, hw = mean_ci([3.0, 3.0, 3.0, 3.0])
        assert m == 3.0 and hw == 0.0

    def test_single_sample_point_estimate(self):
        m, hw = mean_ci([2.5])
        assert m == 2.5 and hw == 0.0

    def test_width_shrinks_like_sqrt_n(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(6400)
        _, hw_small = mean_ci(x[:100])
        _, hw_big = mean_ci(x)
        # 64x the samples -> ~8x narrower; allow slack for sample-std noise
        assert hw_big < hw_small / 5

    def test_coverage_near_nominal(self):
        """95% CI covers the true mean about 95% of the time."""
        rng = np.random.default_rng(3)
        hits = 0
        n_rep = 400
        for _ in range(n_rep):
            x = rng.standard_normal(30) + 1.7
            m, hw = mean_ci(x, confidence=0.95)
            hits += (m - hw <= 1.7 <= m + hw)
        assert 0.90 <= hits / n_rep <= 0.99

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            mean_ci([])
        with pytest.raises(ValueError):
            mean_ci([1.0, 2.0], confidence=1.5)


class TestBootstrapCi:
    def test_deterministic_with_seeded_rng(self):
        x = np.arange(20.0)
        a = bootstrap_ci(x, rng=np.random.default_rng(0))
        b = bootstrap_ci(x, rng=np.random.default_rng(0))
        assert a == b

    def test_constant_samples_degenerate_interval(self):
        lo, hi = bootstrap_ci([5.0] * 10, rng=np.random.default_rng(1))
        assert lo == hi == 5.0

    def test_contains_point_estimate(self):
        rng = np.random.default_rng(4)
        x = rng.exponential(size=200)
        lo, hi = bootstrap_ci(x, stat=np.median, rng=np.random.default_rng(5))
        assert lo <= np.median(x) <= hi


class TestEdgeRate:
    def test_matches_pooled_percentile(self):
        rng = np.random.default_rng(6)
        rates = rng.uniform(0, 10, size=(50, 4))  # (drops, users)
        assert edge_rate(rates, q=5.0) == pytest.approx(np.percentile(rates.ravel(), 5.0))

    def test_pools_across_users_not_per_drop(self):
        """One weak user out of two: the pooled 5th percentile must reflect
        the weak user's rates, not the per-drop sums."""
        rates = np.stack([np.full(100, 10.0), np.full(100, 0.1)], axis=1)
        assert edge_rate(rates, q=5.0) == pytest.approx(0.1)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            edge_rate([])
