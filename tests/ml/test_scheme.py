"""GlimpseScheme: CodebookScheme conformance, bit honesty, prefix property."""

import numpy as np
import pytest

from nr_csi.channel import RandomRayChannel
from nr_csi.config import AntennaConfig
from nr_csi.eval import evaluate
from nr_csi.ml import GlimpseScheme, LeastSquaresDecoder, OMPDecoder
from nr_csi.ml.projection import GlimpseCodec, fit_klt

ANT = AntennaConfig.standard(4, 2)  # P = 16 keeps the numpy tests light
N3 = 8


@pytest.fixture(scope="module")
def codec():
    return GlimpseCodec(ANT, N3, m_max=48, seed=0)


@pytest.fixture(scope="module")
def scheme(codec):
    return GlimpseScheme(ANT, N3, LeastSquaresDecoder(codec), m=24, bits=3, codec=codec)


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

    def test_overhead_matches_bitstream(self, scheme):
        H = _channel(1).generate(n_slots=1)
        for rank in (1, 2):
            pmi = scheme.select(H, rank=rank)
            bits = scheme.total_overhead_bits(pmi)
            assert bits == rank * 2 * 24 * 3  # allocation sums to the budget
            stream = scheme.pack(pmi)
            assert len(stream) == bits  # the serialize.py honesty convention
            assert set(stream) <= {"0", "1"}
            pmi2 = scheme.unpack(stream, rank)
            np.testing.assert_array_equal(pmi2["u_hat"], pmi["u_hat"])
            np.testing.assert_allclose(scheme.precoder(pmi2), scheme.precoder(pmi))

    def test_deterministic(self, scheme):
        H = _channel(2).generate(n_slots=1)
        a, b = scheme.select(H), scheme.select(H)
        np.testing.assert_array_equal(a["u_hat"], b["u_hat"])
        assert a["streams"] == b["streams"]

    def test_waterfill_and_uniform_same_budget(self):
        # a KLT codec with decaying coordinate variances (the regime where
        # water-filling differs from uniform)
        rng = np.random.default_rng(0)
        n, r = 3000, 12
        basis = rng.standard_normal((ANT.P * N3, r)) + 1j * rng.standard_normal(
            (ANT.P * N3, r))
        coeff = (rng.standard_normal((n, r)) + 1j * rng.standard_normal((n, r))) / (
            1 + np.arange(r))
        g = coeff @ basis.T
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        A, sigma = fit_klt(g, m_max=48)
        klt = GlimpseCodec(ANT, N3, m_max=48, basis_matrix=A, sigma=sigma)
        H = _channel(5).generate(n_slots=1)
        wf = GlimpseScheme(ANT, N3, LeastSquaresDecoder(klt), m=24, bits=3,
                           codec=klt, allocation="waterfill")
        un = GlimpseScheme(ANT, N3, LeastSquaresDecoder(klt), m=24, bits=3,
                           codec=klt, allocation="uniform")
        assert int(wf.bits_vec.sum()) == int(un.bits_vec.sum()) == 2 * 24 * 3
        assert wf.total_overhead_bits(wf.select(H)) == un.total_overhead_bits(un.select(H))
        # water-filling puts more bits on the leading (high-variance) coordinates
        assert wf.bits_vec[0] > un.bits_vec[0]
        assert wf.bits_vec[-1] < un.bits_vec[-1]

    def test_m_out_of_range(self, codec):
        with pytest.raises(ValueError):
            GlimpseScheme(ANT, N3, LeastSquaresDecoder(codec), m=64, codec=codec)


class TestPrefixProperty:
    def test_truncated_report_decodes(self, scheme):
        H = _channel(3).generate(n_slots=1)
        pmi = scheme.select(H, rank=1)
        short = scheme.truncate(pmi, 8)
        assert scheme.total_overhead_bits(short) == 2 * 8 * 3
        W = scheme.precoder(short)
        assert np.all(np.isfinite(W))

    def test_more_measurements_reconstruct_better(self, codec):
        """With the prior-free LS decoder, SGCS must be monotone in the
        prefix length on average (graceful degradation), and pinned near the
        m/D energy-capture law -- the floor any prior-informed decoder exists
        to beat: min-norm inversion recovers only the component of the target
        inside the m-dimensional measured subspace."""
        from nr_csi.metrics.similarity import sgcs

        ls = LeastSquaresDecoder(codec)
        scores = {m: [] for m in (8, 24, 48)}
        for seed in range(12):
            H = _channel(10 + seed).generate(n_slots=1)
            full = GlimpseScheme(ANT, N3, ls, m=48, bits=4, codec=codec)
            pmi = full.select(H, rank=1)
            from nr_csi.codebooks._spatial import aligned_eigen_targets

            V = aligned_eigen_targets(H[-1], 1)
            for m in scores:
                W = full.precoder(full.truncate(pmi, m))
                scores[m].append(sgcs(V[None], W))
        means = {m: np.mean(v) for m, v in scores.items()}
        assert means[8] < means[24] < means[48]
        D = codec.D
        for m, mu in means.items():
            assert mu == pytest.approx(m / D, abs=0.08), (m, mu)


class TestClassicalDecoders:
    def test_omp_beats_ls_on_sparse_vector(self, codec):
        """On an exactly sparse angle-delay vector, the sparsity prior must
        recover what least squares cannot (m < D)."""
        rng = np.random.default_rng(7)
        g = np.zeros(codec.D, complex)
        g[rng.choice(codec.D, 8, replace=False)] = (
            rng.standard_normal(8) + 1j * rng.standard_normal(8)
        )
        g /= np.linalg.norm(g)
        m = 32
        y = codec.project(g, m)  # physical measurements
        ls = LeastSquaresDecoder(codec)(y, m)
        omp = OMPDecoder(codec, sparsity=8)(y, m)

        def err(x):
            return np.linalg.norm(x - g) / np.linalg.norm(g)

        assert err(omp) < 0.05 < err(ls)


class TestHarnessIntegration:
    def test_evaluate_smoke(self, scheme):
        res = evaluate(scheme, _channel(42), snr_db=[10.0], rank=1, n_drops=4,
                       rng=np.random.default_rng(0))
        assert np.isfinite(res.se[0]) and 0.0 <= res.sgcs <= 1.0
        assert res.overhead_bits == 2 * 24 * 3
