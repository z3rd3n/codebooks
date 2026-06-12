"""Anchors for the synthetic channel, ideal baselines, and SE/SGCS metrics."""

import numpy as np

from nr_csi.baselines import eigen_precoder, mmse, zf
from nr_csi.channel import Ray, SyntheticRayChannel
from nr_csi.config import AntennaConfig
from nr_csi.metrics import mu_rate, nmse, sgcs, su_rate, subspace_sgcs
from nr_csi.utils import dft

CFG = AntennaConfig.standard(4, 2)  # P = 16


def dual_beam(cfg, m1, m2, pol_phase=0.0):
    v = dft.spatial_beam(cfg, m1, m2)
    return np.concatenate([v, np.exp(1j * pol_phase) * v])


class TestSyntheticChannel:
    def test_on_grid_ray_matched_by_codebook_beam(self):
        ray = Ray(gain=1.0, m1=5, m2=2, pol_phase=np.pi / 2)
        H = SyntheticRayChannel(CFG, [ray], N3=1).generate()
        v_dual = dual_beam(CFG, 5, 2, np.pi / 2)
        # H = a v^H, so H v_dual = gain * P (matched beam collects full power)
        assert np.allclose(H[0, 0] @ v_dual, CFG.P)
        # any other grid beam from a different orthogonal position picks up nothing
        v_other = dual_beam(CFG, 5 + CFG.O1, 2, np.pi / 2)
        assert np.allclose(H[0, 0] @ v_other, 0.0, atol=1e-9)

    def test_delay_creates_linear_phase_across_frequency(self):
        N3 = 8
        ch = SyntheticRayChannel(CFG, [Ray(gain=1.0, m1=0, delay=3)], N3=N3)
        H = ch.generate()
        ratio = H[0, 1:, 0, 0] / H[0, :-1, 0, 0]
        assert np.allclose(ratio, np.exp(2j * np.pi * 3 / N3))

    def test_doppler_creates_linear_phase_across_slots(self):
        ch = SyntheticRayChannel(CFG, [Ray(gain=1.0, m1=0, doppler=1)], N3=1, doppler_period=4)
        H = ch.generate(n_slots=4)
        ratio = H[1:, 0, 0, 0] / H[:-1, 0, 0, 0]
        assert np.allclose(ratio, np.exp(2j * np.pi / 4))


class TestBaselinesAndSE:
    def test_eigen_precoder_rank1_channel(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        v = dual_beam(CFG, 3, 1)
        H = np.outer(a, v.conj())[None]  # (1, Nr, P)
        W = eigen_precoder(H, rank=1)
        assert W.shape == (1, CFG.P, 1)
        assert np.isclose(np.linalg.norm(W[0]), 1.0)
        assert sgcs(v[None, :, None] / np.linalg.norm(v), W) > 1 - 1e-12
        rho = 10.0
        expected = np.log2(1 + rho * np.linalg.norm(a) ** 2 * np.linalg.norm(v) ** 2)
        assert np.isclose(su_rate(H, W, rho), expected)

    def test_eigen_precoder_is_su_rate_upper_bound(self):
        rng = np.random.default_rng(1)
        H = rng.standard_normal((6, 2, CFG.P)) + 1j * rng.standard_normal((6, 2, CFG.P))
        W_opt = eigen_precoder(H, rank=2)
        for _ in range(20):
            W_rand = rng.standard_normal((6, CFG.P, 2)) + 1j * rng.standard_normal((6, CFG.P, 2))
            W_rand /= np.linalg.norm(W_rand, axis=(-2, -1), keepdims=True)
            # rank-2 eigen beamforming with equal split beats random precoding
            assert su_rate(H, W_opt, 10.0) >= su_rate(H, W_rand, 10.0) - 1e-9

    def test_zf_inverts_channel(self):
        rng = np.random.default_rng(2)
        H = rng.standard_normal((4, 8)) + 1j * rng.standard_normal((4, 8))
        assert np.allclose(H @ zf(H), np.eye(4), atol=1e-9)

    def test_mmse_approaches_zf_at_high_snr(self):
        rng = np.random.default_rng(3)
        H = rng.standard_normal((4, 8)) + 1j * rng.standard_normal((4, 8))
        assert np.allclose(mmse(H, 1e12), zf(H), atol=1e-6)

    def test_mu_rate_zf_removes_interference(self):
        rng = np.random.default_rng(4)
        K, P = 2, CFG.P
        H = rng.standard_normal((K, 1, 1, P)) + 1j * rng.standard_normal((K, 1, 1, P))
        Hstack = H[:, 0, 0, :]  # (K, P)
        W_zf = zf(Hstack)  # (P, K)
        W = np.stack([W_zf[:, k][None, :, None] / np.linalg.norm(W_zf[:, k]) for k in range(K)])
        rates = mu_rate(H, W, rho=100.0)
        for k in range(K):
            assert np.isclose(rates[k], su_rate(H[k], W[k], 100.0), atol=1e-6)


class TestSimilarity:
    def test_sgcs_phase_and_scale_invariant(self):
        rng = np.random.default_rng(5)
        w = rng.standard_normal((8, 1)) + 1j * rng.standard_normal((8, 1))
        assert np.isclose(sgcs(w, 3.7 * np.exp(1j * 0.83) * w), 1.0)

    def test_sgcs_orthogonal_is_zero(self):
        w1 = np.array([1, 0, 0, 0], dtype=complex)[:, None]
        w2 = np.array([0, 1, 0, 0], dtype=complex)[:, None]
        assert np.isclose(sgcs(w1, w2), 0.0)

    def test_nmse_zero_for_phase_rotated_copy(self):
        rng = np.random.default_rng(6)
        w = rng.standard_normal((8, 2)) + 1j * rng.standard_normal((8, 2))
        assert nmse(w, w * np.exp(1j * 1.1)) < 1e-12

    def test_nmse_positive_for_mismatch(self):
        w1 = np.array([1, 0], dtype=complex)[:, None]
        w2 = np.array([0, 1], dtype=complex)[:, None]
        assert np.isclose(nmse(w1, w2), 2.0)  # orthogonal unit vectors


class TestSubspaceSGCS:
    def _rand(self, rng, *shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    def test_rank1_equals_sgcs(self):
        rng = np.random.default_rng(10)
        w_ref = self._rand(rng, 3, 8, 1)
        w_hat = self._rand(rng, 3, 8, 1)
        assert np.isclose(subspace_sgcs(w_ref, w_hat), sgcs(w_ref, w_hat))

    def test_right_unitary_invariance(self):
        rng = np.random.default_rng(11)
        W_ref = self._rand(rng, 8, 2)
        W_hat = self._rand(rng, 8, 2)
        U = np.linalg.qr(self._rand(rng, 2, 2))[0]
        assert np.isclose(subspace_sgcs(W_ref, W_hat @ U), subspace_sgcs(W_ref, W_hat))
        # the column-wise SGCS is NOT invariant in general (that is the point)
        assert not np.isclose(sgcs(W_ref, W_hat @ U), sgcs(W_ref, W_hat))

    def test_at_least_columnwise_sgcs(self):
        rng = np.random.default_rng(12)
        for _ in range(20):
            W_ref = self._rand(rng, 8, 2)
            W_hat = self._rand(rng, 8, 2)
            assert subspace_sgcs(W_ref, W_hat) >= sgcs(W_ref, W_hat) - 1e-12

    def test_zero_precoder_is_zero(self):
        rng = np.random.default_rng(13)
        w = self._rand(rng, 8, 2)
        z = np.zeros((8, 2), dtype=complex)
        assert subspace_sgcs(w, z) == 0.0
        # a degenerate (duplicated-column) precoder spans only one direction:
        # the masked second QR column must not capture extra energy
        dup = np.stack([w[:, 0], w[:, 0]], axis=1)
        assert np.isclose(subspace_sgcs(w, dup), subspace_sgcs(w, w[:, :1]))

    def test_type1_rank2_subspace_gap(self):
        """S1 regression: Type I's rigid rank-2 pair spans a decent subspace
        but cannot rotate within it; the column-wise SGCS punishes that
        rotation while the subspace metric does not."""
        from nr_csi.channel import RandomRayChannel
        from nr_csi.codebooks import Type1Codebook

        chan = RandomRayChannel(CFG, N3=4, n_rx=2)
        cbk = Type1Codebook(CFG, N3=4)
        rng = np.random.default_rng(0)
        gaps = []
        for _ in range(5):
            H = chan.generate(n_slots=1, rng=rng)
            W = cbk.precoder(cbk.select(H, rank=2))
            W_ref = eigen_precoder(H, rank=2)
            gaps.append(subspace_sgcs(W_ref, W) - sgcs(W_ref, W))
        assert np.mean(gaps) >= 0.1


class TestMetricEdgeCases:
    def test_sgcs_zero_columns(self):
        w = np.array([1, 1j, 0, 0], dtype=complex)[:, None] / np.sqrt(2)
        z = np.zeros((4, 1), dtype=complex)
        assert sgcs(z, w) == 0.0
        assert sgcs(w, z) == 0.0
        assert sgcs(z, z) == 0.0

    def test_nmse_zero_reference(self):
        w = np.ones((4, 1), dtype=complex)
        z = np.zeros((4, 1), dtype=complex)
        assert nmse(z, w) == 0.0  # zero-reference columns are skipped, not inf
        assert np.isclose(nmse(w, z), 1.0)

    def test_su_rate_rank_deficient_precoder(self):
        rng = np.random.default_rng(7)
        H = rng.standard_normal((4, 2, CFG.P)) + 1j * rng.standard_normal((4, 2, CFG.P))
        w = rng.standard_normal(CFG.P) + 1j * rng.standard_normal(CFG.P)
        W = np.stack([w, w], axis=1) / np.linalg.norm(np.stack([w, w], axis=1))
        rate = su_rate(H, W[None], 10.0)
        assert np.isfinite(rate) and rate >= 0
        # a duplicated column buys nothing over the single beam at equal power
        w1 = w[:, None] / np.linalg.norm(w)
        assert rate <= su_rate(H, w1[None], 10.0) + 1e-9

    def test_mu_rate_single_user_equals_su_rate(self):
        rng = np.random.default_rng(8)
        H = rng.standard_normal((1, 3, 2, CFG.P)) + 1j * rng.standard_normal((1, 3, 2, CFG.P))
        W = rng.standard_normal((1, 3, CFG.P, 2)) + 1j * rng.standard_normal((1, 3, CFG.P, 2))
        W /= np.linalg.norm(W, axis=(-2, -1), keepdims=True)
        assert np.isclose(mu_rate(H, W, 5.0)[0], su_rate(H[0], W[0], 5.0), atol=1e-9)


class TestHarness:
    def _channel(self, **kw):
        from nr_csi.channel import RandomRayChannel

        return RandomRayChannel(CFG, N3=4, n_rx=2, **kw)

    def test_evaluate_r16_invariants(self):
        from nr_csi.codebooks import R16Type2Codebook
        from nr_csi.eval import evaluate

        cbk = R16Type2Codebook(CFG, N3=4, param_combination=2)
        res = evaluate(cbk, self._channel(), snr_db=[0.0, 10.0], rank=1,
                       n_drops=5, rng=np.random.default_rng(0))
        assert all(c <= u + 1e-9 for c, u in zip(res.se, res.se_upper_bound))
        assert res.se[0] < res.se[1]
        assert 0 < res.sgcs <= 1
        assert len(res.per_drop_sgcs) == 5
        assert res.overhead_bits > 0

    def test_evaluate_r18_multi_slot(self):
        """Harness-level R18 run (n_slots = N4), previously only hand-rolled
        in codebook tests."""
        from nr_csi.codebooks import R18Type2Codebook
        from nr_csi.eval import evaluate

        cbk = R18Type2Codebook(CFG, N3=4, N4=4, param_combination=5)
        chan = self._channel(max_doppler=1.0, doppler_period=4)
        res = evaluate(cbk, chan, snr_db=[10.0], rank=1, n_drops=4,
                       n_slots=4, rng=np.random.default_rng(1))
        assert all(c <= u + 1e-9 for c, u in zip(res.se, res.se_upper_bound))
        assert 0 < res.sgcs <= 1

    def test_evaluate_deterministic_with_seeded_rng(self):
        from nr_csi.codebooks import R15Type2Codebook
        from nr_csi.eval import evaluate

        cbk = R15Type2Codebook(CFG, N3=4, L=2)
        runs = [
            evaluate(cbk, self._channel(), snr_db=[10.0], rank=1, n_drops=3,
                     rng=np.random.default_rng(42))
            for _ in range(2)
        ]
        assert runs[0].se == runs[1].se
        assert runs[0].sgcs == runs[1].sgcs
        assert runs[0].overhead_bits == runs[1].overhead_bits
