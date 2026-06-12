"""Closed-form anchors for the paper's Appendix A beamforming schemes
(Tables SUMIMO/MUMIMO, Algorithm WMMSE, power allocation) and the auto-RI
helper (plan C2.4/C2.6)."""

import numpy as np
import pytest

from nr_csi.baselines import (
    bd,
    ezf,
    gmd,
    harmonic_mean_allocation,
    water_filling,
    wmmse,
    zf,
)
from nr_csi.metrics.se import mu_rate


def rand_channel(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


class TestGMD:
    @pytest.mark.parametrize("shape,K", [((4, 8), 4), ((2, 6), 2), ((4, 8), 3)])
    def test_decomposition_properties(self, shape, K):
        rng = np.random.default_rng(0)
        H = rand_channel(rng, *shape)
        Q, R, P = gmd(H, K)
        # unitary factors
        assert np.allclose(Q.conj().T @ Q, np.eye(K), atol=1e-10)
        assert np.allclose(P.conj().T @ P, np.eye(K), atol=1e-10)
        # upper triangular with equal diagonal = geometric mean of singular values
        assert np.allclose(np.tril(R, -1), 0, atol=1e-10)
        s = np.linalg.svd(H, compute_uv=False)[:K]
        sbar = np.exp(np.mean(np.log(s)))
        assert np.allclose(np.diag(R).real, sbar, atol=1e-9)
        assert np.allclose(np.diag(R).imag, 0, atol=1e-9)
        # exact reconstruction of the rank-K truncation
        U, sv, Vh = np.linalg.svd(H, full_matrices=False)
        H_K = U[:, :K] @ np.diag(sv[:K]) @ Vh[:K]
        assert np.allclose(Q @ R @ P.conj().T, H_K, atol=1e-9)

    def test_equal_singular_values_identity(self):
        """If all singular values are equal, R is already diagonal."""
        rng = np.random.default_rng(1)
        A = np.linalg.qr(rand_channel(rng, 6, 6))[0][:, :3]
        B = np.linalg.qr(rand_channel(rng, 6, 6))[0][:, :3]
        H = 2.0 * A @ B.conj().T
        Q, R, P = gmd(H, 3)
        assert np.allclose(R, 2.0 * np.eye(3), atol=1e-9)


class TestEZF:
    def test_zero_eigen_leakage_without_regularization(self):
        """xi = 0: V_eff^H W = I, so user j's dominant eigendirection sees no
        energy from user k's beam."""
        rng = np.random.default_rng(2)
        H_users = rand_channel(rng, 3, 2, 12)
        W = ezf(H_users, n_streams=1, xi=0.0)
        for j in range(3):
            _, _, Vh = np.linalg.svd(H_users[j])
            v1 = Vh.conj().T[:, 0]
            for k in range(3):
                inner = v1.conj() @ W[:, k]
                if j == k:
                    assert abs(inner) > 1e-3
                else:
                    assert abs(inner) < 1e-10


class TestBD:
    def test_zero_interuser_leakage(self):
        rng = np.random.default_rng(3)
        K, Nr, Nt = 3, 2, 8
        H_users = rand_channel(rng, K, Nr, Nt)
        W = bd(H_users, n_streams=2)
        assert W.shape == (Nt, K * 2)
        for j in range(K):
            for k in range(K):
                block = H_users[j] @ W[:, k * 2:(k + 1) * 2]
                if j != k:
                    assert np.allclose(block, 0, atol=1e-10)
                else:
                    assert np.linalg.norm(block) > 1e-6

    def test_requires_enough_antennas(self):
        rng = np.random.default_rng(4)
        with pytest.raises(ValueError):
            bd(rand_channel(rng, 3, 2, 4), n_streams=1)


class TestWMMSE:
    def _sum_rate(self, H_users, W, D, rho):
        K = H_users.shape[0]
        # per-user precoders normalized to the framework convention
        Wn = np.stack([
            W[:, k * D:(k + 1) * D][None] / np.linalg.norm(W[:, k * D:(k + 1) * D])
            for k in range(K)
        ])
        return float(np.sum(mu_rate(H_users[:, None], Wn, rho / K)))

    def test_objective_improves_with_iterations(self):
        rng = np.random.default_rng(5)
        H_users = rand_channel(rng, 2, 2, 8)
        rho = 10.0
        rates = [
            self._sum_rate(H_users, wmmse(H_users, rho, n_streams=1, n_iter=it,
                                          rng=np.random.default_rng(0)), 1, rho)
            for it in (1, 5, 20)
        ]
        assert rates[0] <= rates[1] + 1e-6 <= rates[2] + 2e-6

    def test_beats_plain_zf_at_low_snr(self):
        """At low SNR ZF over-inverts; WMMSE trades interference for power."""
        rng = np.random.default_rng(6)
        H_users = rand_channel(rng, 2, 1, 4)
        rho = 0.5
        W_w = wmmse(H_users, rho, n_streams=1, n_iter=30)
        Hstack = H_users[:, 0, :]
        W_z = zf(Hstack)
        r_w = self._sum_rate(H_users, W_w, 1, rho)
        r_z = self._sum_rate(H_users, W_z, 1, rho)
        assert r_w >= r_z - 1e-9


class TestPowerAllocation:
    def test_water_filling_constraints_and_ordering(self):
        gains = np.array([4.0, 1.0, 0.25])
        p = water_filling(gains, p_total=3.0)
        assert np.isclose(p.sum(), 3.0)
        assert np.all(p >= 0)
        assert p[0] >= p[1] >= p[2]  # stronger subchannel gets at least as much

    def test_water_filling_drops_weak_channel(self):
        """A very weak subchannel gets zero power at low budget."""
        p = water_filling(np.array([10.0, 0.001]), p_total=0.5)
        assert p[1] == 0.0 and np.isclose(p[0], 0.5)

    def test_water_filling_two_channel_analytic(self):
        g = np.array([2.0, 1.0])
        pt = 4.0
        p = water_filling(g, pt)
        mu = (pt + (1 / g).sum()) / 2
        assert np.allclose(p, mu - 1 / g)

    def test_water_filling_equal_split_at_high_power(self):
        g = np.array([2.0, 1.0])
        p = water_filling(g, p_total=1e6)
        assert np.allclose(p / p.sum(), 0.5, atol=1e-3)

    def test_harmonic_mean_allocation(self):
        gains = np.array([4.0, 1.0])  # lambda = 2, 1
        p = harmonic_mean_allocation(gains, p_total=3.0)
        assert np.isclose(p.sum(), 3.0)
        assert np.allclose(p, [1.0, 2.0])  # P_i proportional to 1/lambda_i
        # resulting per-stream SNRs P_i*lambda_i^2 = beta*lambda_i
        snrs = p * gains
        assert snrs[0] > snrs[1]


class TestAutoRank:
    def test_select_rank_picks_se_maximizer(self):
        from nr_csi.codebooks import R16Type2Codebook
        from nr_csi.config import AntennaConfig
        from nr_csi.eval.harness import select_rank
        from nr_csi.metrics.se import su_rate

        ant = AntennaConfig.standard(4, 2)
        cbk = R16Type2Codebook(ant, N3=4, param_combination=5)
        rng = np.random.default_rng(7)
        H = rand_channel(rng, 1, 4, 4, ant.P)
        rank, pmi, W, se = select_rank(cbk, H, rho=100.0)
        assert rank == pmi.rank
        for r in (1, 2, 3, 4):
            se_r = su_rate(H, cbk.precoder(cbk.select(H, rank=r)), 100.0)
            assert se >= se_r - 1e-9
        assert rank > 1  # at rho = 100 on a rich channel, multiplexing wins

    def test_select_rank_honors_restriction(self):
        from nr_csi.codebooks import Type1Codebook
        from nr_csi.config import AntennaConfig
        from nr_csi.eval.harness import select_rank

        ant = AntennaConfig.standard(4, 2)
        r = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        cbk = Type1Codebook(ant, N3=2, rank_restriction=r)
        rng = np.random.default_rng(8)
        H = rand_channel(rng, 1, 2, 2, ant.P)
        rank, _, _, _ = select_rank(cbk, H, rho=100.0)
        assert rank == 1  # rank 2 prohibited even though SE would prefer it
