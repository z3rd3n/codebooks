"""Per-layer linear MMSE receiver rate (``su_rate_mmse``).

Anchors the closed form SINR_l = 1/[(I + rho G^H G)^{-1}]_{ll} - 1 against
the definition (explicit interference-plus-noise inversion), and the two
structural facts the metric exists for: it never exceeds the joint log-det
rate, and it matches it exactly when the effective layers are orthogonal.
"""

import numpy as np

from nr_csi.metrics import su_rate, su_rate_mmse


def _rand(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _brute_force_mmse_rate(H: np.ndarray, W: np.ndarray, rho: float) -> float:
    """Definition-level SINR: per layer, invert I + rho * (other layers)."""
    G = H @ W  # (..., Nr, v)
    lead = G.shape[:-2]
    nr, v = G.shape[-2], G.shape[-1]
    G = G.reshape(-1, nr, v)
    rates = []
    for g in G:
        total = 0.0
        for l in range(v):
            others = np.delete(g, l, axis=1)  # (Nr, v-1)
            Q = np.eye(nr) + rho * (others @ others.conj().T)
            sinr = rho * np.real(g[:, l].conj() @ np.linalg.solve(Q, g[:, l]))
            total += np.log2(1 + sinr)
        rates.append(total)
    return float(np.mean(np.array(rates).reshape(lead or (1,))))


class TestSuRateMmse:
    def test_rank1_equals_joint_rate(self):
        """v = 1: no inter-layer interference, MMSE = matched filter = joint."""
        rng = np.random.default_rng(0)
        H = _rand(rng, 5, 4, 2, 8)
        W = _rand(rng, 5, 4, 8, 1)
        W /= np.linalg.norm(W, axis=-2, keepdims=True)
        for rho in (0.1, 1.0, 10.0, 1000.0):
            assert np.isclose(su_rate_mmse(H, W, rho), su_rate(H, W, rho), rtol=1e-10)

    def test_matches_brute_force_sinr_definition(self):
        rng = np.random.default_rng(1)
        for v in (2, 3, 4):
            H = _rand(rng, 6, 3, 4, 8)
            W = _rand(rng, 6, 3, 8, v)
            W /= np.sqrt(v) * np.linalg.norm(W, axis=-2, keepdims=True)
            for rho in (1.0, 10.0):
                assert np.isclose(
                    su_rate_mmse(H, W, rho), _brute_force_mmse_rate(H, W, rho), rtol=1e-9
                )

    def test_never_exceeds_joint_decoding(self):
        rng = np.random.default_rng(2)
        for _ in range(30):
            v = int(rng.integers(2, 5))
            H = _rand(rng, 4, 2, 8)
            W = _rand(rng, 4, 8, v)
            W /= np.linalg.norm(W, axis=(-2, -1), keepdims=True)
            assert su_rate_mmse(H, W, 10.0) <= su_rate(H, W, 10.0) + 1e-9

    def test_equality_for_orthogonal_effective_layers(self):
        """Diagonal G^H G (orthogonal layers at the receiver): per-layer MMSE
        loses nothing vs joint decoding."""
        rng = np.random.default_rng(3)
        nr = 6
        Q = np.linalg.qr(_rand(rng, nr, 3))[0]  # orthonormal effective layers
        H = np.eye(nr)[None]  # (1, Nr, P) with P = Nr
        W = (Q * rng.uniform(0.5, 2.0, size=3))[None]  # scaled, still orthogonal
        for rho in (0.5, 10.0):
            assert np.isclose(su_rate_mmse(H, W, rho), su_rate(H, W, rho), rtol=1e-10)

    def test_strict_gap_for_correlated_layers(self):
        """Two nearly-colinear layers: joint decoding still separates them,
        a linear per-layer receiver mostly cannot."""
        h = np.array([1.0, 0.0], dtype=complex)
        g = np.array([1.0, 0.05], dtype=complex)
        H = np.eye(2, dtype=complex)[None]
        W = np.stack([h, g / np.linalg.norm(g)], axis=1)[None] / np.sqrt(2)
        joint = su_rate(H, W, 100.0)
        mmse = su_rate_mmse(H, W, 100.0)
        assert mmse < joint - 1.0  # multiple bits lost, not epsilon

    def test_zero_precoder_is_zero_rate(self):
        H = np.ones((1, 2, 4), dtype=complex)
        W = np.zeros((1, 4, 2), dtype=complex)
        assert su_rate_mmse(H, W, 10.0) == 0.0

    def test_monotone_in_snr(self):
        rng = np.random.default_rng(4)
        H = _rand(rng, 3, 2, 6)
        W = _rand(rng, 3, 6, 2)
        W /= np.linalg.norm(W, axis=(-2, -1), keepdims=True)
        rates = [su_rate_mmse(H, W, 10 ** (s / 10)) for s in (-10, 0, 10, 20, 30)]
        assert all(b >= a - 1e-12 for a, b in zip(rates, rates[1:]))


class TestEvaluateSeMmse:
    def test_evaluate_reports_se_mmse(self):
        from nr_csi.channel import RandomRayChannel
        from nr_csi.codebooks import R16Type2Codebook
        from nr_csi.config import AntennaConfig
        from nr_csi.eval import evaluate

        cfg = AntennaConfig.standard(4, 2)
        cbk = R16Type2Codebook(cfg, N3=4, param_combination=2)
        chan = RandomRayChannel(cfg, N3=4, n_rx=2)
        res = evaluate(cbk, chan, snr_db=[0.0, 10.0], rank=2, n_drops=4,
                       rng=np.random.default_rng(0))
        assert len(res.se_mmse) == 2
        # per-layer MMSE never beats joint decoding, and both grow with SNR
        assert all(m <= s + 1e-9 for m, s in zip(res.se_mmse, res.se))
        assert res.se_mmse[0] < res.se_mmse[1]

    def test_rank1_se_mmse_equals_se(self):
        from nr_csi.channel import RandomRayChannel
        from nr_csi.codebooks import Type1Codebook
        from nr_csi.config import AntennaConfig
        from nr_csi.eval import evaluate

        cfg = AntennaConfig.standard(4, 2)
        cbk = Type1Codebook(cfg, N3=4)
        chan = RandomRayChannel(cfg, N3=4, n_rx=2)
        res = evaluate(cbk, chan, snr_db=[10.0], rank=1, n_drops=3,
                       rng=np.random.default_rng(1))
        assert np.isclose(res.se_mmse[0], res.se[0], rtol=1e-10)
