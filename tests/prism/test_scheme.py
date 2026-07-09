"""PrismScheme: interface conformance, bit honesty, index header, truncation."""

import numpy as np
import pytest

from nr_csi.channel import RandomRayChannel
from nr_csi.config import AntennaConfig
from nr_csi.eval import evaluate
from nr_csi.prism import PrismCodec, PrismScheme, fit_mixture

ANT = AntennaConfig.standard(4, 2)  # P = 16
N3 = 8
D = ANT.P * N3


@pytest.fixture(scope="module")
def codec():
    # Two synthetic regimes with SLOWLY decaying spectra (r=24, 0.85^k): the
    # effective dimension must exceed the sketch size or a pooled basis
    # already captures everything and there is no regime structure to test.
    rng = np.random.default_rng(0)
    parts = []
    for c in range(2):
        basis = rng.standard_normal((D, 24)) + 1j * rng.standard_normal((D, 24))
        coeff = (rng.standard_normal((800, 24)) + 1j * rng.standard_normal((800, 24)))
        coeff *= 0.85 ** np.arange(24)
        g = coeff @ basis.T
        parts.append(g / np.linalg.norm(g, axis=1, keepdims=True))
    g = np.concatenate(parts)
    bases, sigmas, _, _ = fit_mixture(g, 2, m_max=32, m_ref=12, seed=2)
    return PrismCodec(ANT, N3, bases=tuple(bases), sigmas=tuple(sigmas))


@pytest.fixture(scope="module")
def scheme(codec):
    return PrismScheme(ANT, N3, codec, m=16, bits=3)


def _channel(seed=0):
    chan = RandomRayChannel(ANT, N3=N3, n_rx=2)
    gen = chan.generate

    def seeded(n_slots=1, rng=None):
        return gen(n_slots=n_slots, rng=rng or np.random.default_rng(seed))

    chan.generate = seeded
    return chan


class TestInterface:
    def test_select_precoder_shapes_and_norm(self, scheme):
        H = _channel().generate(n_slots=1)
        for rank in (1, 2):
            pmi = scheme.select(H, rank=rank)
            W = scheme.precoder(pmi)
            assert W.shape == (1, N3, ANT.P, rank)
            for t in range(N3):
                assert np.trace(W[0, t].conj().T @ W[0, t]).real == pytest.approx(1.0)

    def test_overhead_includes_index_and_matches_bitstream(self, scheme):
        H = _channel(1).generate(n_slots=1)
        for rank in (1, 2):
            pmi = scheme.select(H, rank=rank)
            bits = scheme.total_overhead_bits(pmi)
            assert bits == 1 + rank * 2 * 16 * 3  # K=2 -> 1 index bit, shared
            stream = scheme.pack(pmi)
            assert len(stream) == bits
            assert set(stream) <= {"0", "1"}
            pmi2 = scheme.unpack(stream, rank)
            assert pmi2["basis"] == pmi["basis"]
            np.testing.assert_array_equal(pmi2["u_hat"], pmi["u_hat"])
            np.testing.assert_allclose(scheme.precoder(pmi2), scheme.precoder(pmi))

    def test_deterministic(self, scheme):
        H = _channel(2).generate(n_slots=1)
        a, b = scheme.select(H), scheme.select(H)
        assert a["basis"] == b["basis"]
        assert a["streams"] == b["streams"]

    def test_m_out_of_range(self, codec):
        with pytest.raises(ValueError):
            PrismScheme(ANT, N3, codec, m=64)


class TestSelection:
    def _planted_channel(self, codec, rng, regime):
        """A rank-1 channel whose per-subband dominant right singular vector
        IS a vector from ``regime``'s principal subspace: for
        ``H[t] = u v_t^H`` the right singular vector is exactly ``v_t``."""
        coeff = rng.standard_normal(12) + 1j * rng.standard_normal(12)
        coeff *= 0.85 ** np.arange(12)
        g = coeff @ codec.bases[regime][:12].conj()
        g /= np.linalg.norm(g)
        V = codec.vec_to_targets(g)  # (N3, P)
        u = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        H = u[None, :, None] * V.conj()[:, None, :]  # (N3, rx, P)
        return H[None], g  # (slot=1, N3, rx, P)

    def test_selection_recovers_planted_regime(self, codec, scheme):
        rng = np.random.default_rng(7)
        hits = 0
        for trial in range(20):
            regime = trial % 2
            H, _ = self._planted_channel(codec, rng, regime)
            pmi = scheme.select(H, rank=1)
            hits += pmi["basis"] == regime
        assert hits >= 18  # near-orthogonal planted subspaces: selection wins

    def test_wrong_basis_reconstructs_worse(self, codec, scheme):
        """Force-decode planted-regime reports under the other basis: the
        selected basis must reconstruct better, by a wide margin."""
        from nr_csi.metrics.similarity import sgcs
        from nr_csi.codebooks._spatial import aligned_eigen_targets
        from nr_csi.ml.quantizer import dequantize_alloc, quantize_alloc

        rng = np.random.default_rng(8)
        gaps = []
        for trial in range(10):
            H, _ = self._planted_channel(codec, rng, trial % 2)
            pmi = scheme.select(H, rank=1)
            V = aligned_eigen_targets(H[-1], 1)
            right = sgcs(V[None], scheme.precoder(pmi))
            # honest wrong-basis report: re-project + re-quantize under it
            wrong_k = 1 - pmi["basis"]
            g = codec.targets_to_vec(V)
            u = codec.standardize(codec.project(g, scheme.m, wrong_k), wrong_k)
            bv = scheme.bits_vecs[wrong_k]
            s = quantize_alloc(u[0], bv)
            wrong_pmi = dict(pmi, basis=wrong_k,
                             u_hat=np.stack([dequantize_alloc(s, bv)]))
            wrong = sgcs(V[None], scheme.precoder(wrong_pmi))
            gaps.append(right - wrong)
        assert np.mean(gaps) > 0.3  # near-orthogonal regimes: large gap

    def test_truncated_report_decodes_same_basis(self, scheme):
        H = _channel(3).generate(n_slots=1)
        pmi = scheme.select(H, rank=1)
        short = scheme.truncate(pmi, 8)
        assert short["basis"] == pmi["basis"]
        assert scheme.total_overhead_bits(short) == 1 + 2 * 8 * 3
        W = scheme.precoder(short)
        assert np.all(np.isfinite(W))

    def test_monotone_in_prefix_length(self, codec):
        from nr_csi.metrics.similarity import sgcs
        from nr_csi.codebooks._spatial import aligned_eigen_targets

        full = PrismScheme(ANT, N3, codec, m=32, bits=4)
        scores = {m: [] for m in (8, 16, 32)}
        for seed in range(12):
            H = _channel(30 + seed).generate(n_slots=1)
            pmi = full.select(H, rank=1)
            V = aligned_eigen_targets(H[-1], 1)
            for m in scores:
                W = full.precoder(full.truncate(pmi, m))
                scores[m].append(sgcs(V[None], W))
        means = {m: np.mean(v) for m, v in scores.items()}
        assert means[8] < means[16] < means[32]


class TestHarnessIntegration:
    def test_evaluate_smoke(self, scheme):
        res = evaluate(scheme, _channel(42), snr_db=[10.0], rank=1, n_drops=4,
                       rng=np.random.default_rng(0))
        assert np.isfinite(res.se[0]) and 0.0 <= res.sgcs <= 1.0
        assert res.overhead_bits == 1 + 2 * 16 * 3
