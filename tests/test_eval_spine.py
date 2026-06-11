"""Anchors for the synthetic channel, ideal baselines, and SE/SGCS metrics."""

import numpy as np
import pytest

from nr_csi.baselines import eigen_precoder, mmse, zf
from nr_csi.channel import Ray, SyntheticRayChannel
from nr_csi.config import AntennaConfig
from nr_csi.metrics import mu_rate, nmse, sgcs, su_rate
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
