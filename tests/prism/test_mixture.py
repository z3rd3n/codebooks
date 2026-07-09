"""PRISM mixture fitting: planted-cluster recovery, selection, persistence."""

import numpy as np
import pytest

from nr_csi.config import AntennaConfig
from nr_csi.ml.projection import fit_klt
from nr_csi.prism.mixture import (
    PrismCodec,
    captured_energy,
    fit_mixture,
    prism_encoder_flops,
)

ANT = AntennaConfig.standard(4, 2)  # P = 16 keeps tests light
N3 = 8
D = ANT.P * N3


def planted_population(rng, n_per, r=24, gap=0.85):
    """Two well-separated populations + labels.

    ``r`` and ``gap`` set each cluster's effective dimension: the spectrum
    must decay slowly enough that a *pooled* small sketch cannot cover the
    union of both subspaces, or there is nothing for a mixture to gain (a
    too-fast decay makes every cluster ~3-dimensional and the pooled KLT
    already captures everything).  The subspaces are independent random
    draws, hence nearly orthogonal at these dimensions.
    """
    pops, labels = [], []
    for c in range(2):
        basis = rng.standard_normal((D, r)) + 1j * rng.standard_normal((D, r))
        coeff = (rng.standard_normal((n_per, r)) + 1j * rng.standard_normal((n_per, r)))
        coeff *= gap ** np.arange(r)  # decaying spectrum
        g = coeff @ basis.T
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        pops.append(g)
        labels.append(np.full(n_per, c))
    perm = rng.permutation(2 * n_per)
    return np.concatenate(pops)[perm], np.concatenate(labels)[perm]


class TestFitMixture:
    def test_k1_equals_pooled_klt(self):
        rng = np.random.default_rng(0)
        g, _ = planted_population(rng, 400)
        bases, sigmas, assign, obj = fit_mixture(g, 1, m_max=24)
        A_ref, s_ref = fit_klt(g, 24)
        np.testing.assert_allclose(bases[0], A_ref)
        np.testing.assert_allclose(sigmas[0], s_ref)
        assert set(assign) == {0}

    def test_recovers_planted_clusters(self):
        rng = np.random.default_rng(1)
        g, labels = planted_population(rng, 600)
        bases, sigmas, assign, obj = fit_mixture(g, 2, m_max=16, m_ref=8, seed=3)
        # cluster assignment must match the planted labels up to permutation
        agree = np.mean(assign == labels)
        assert max(agree, 1 - agree) > 0.95
        # and the mixture must capture far more energy than the pooled KLT
        A1, _ = fit_klt(g, 16)
        pooled = float(np.mean(captured_energy(A1, g, 8)))
        assert obj > pooled + 0.1

    def test_objective_beats_or_matches_k1(self):
        rng = np.random.default_rng(2)
        g, _ = planted_population(rng, 300)
        _, _, _, obj1 = fit_mixture(g, 1, m_max=16, m_ref=8)
        _, _, _, obj2 = fit_mixture(g, 2, m_max=16, m_ref=8, seed=5)
        assert obj2 >= obj1 - 1e-9


class TestPrismCodec:
    @pytest.fixture(scope="class")
    def codec(self):
        rng = np.random.default_rng(4)
        g, _ = planted_population(rng, 500)
        bases, sigmas, _, _ = fit_mixture(g, 2, m_max=24, m_ref=12, seed=1)
        return PrismCodec(ANT, N3, bases=tuple(bases), sigmas=tuple(sigmas))

    def test_selection_prefers_matching_basis(self, codec):
        rng = np.random.default_rng(5)
        # a vector drawn inside basis 0's principal subspace
        g0 = rng.standard_normal(12) @ codec.bases[0][:12].conj()
        g0 = g0 / np.linalg.norm(g0)
        k = codec.select_basis(g0, m=12)
        e = [captured_energy(A, g0[None], 12)[0] for A in codec.bases]
        assert k == int(np.argmax(e))
        assert e[k] > 0.99

    def test_adjoint_is_pseudo_inverse(self, codec):
        rng = np.random.default_rng(6)
        g = rng.standard_normal(D) + 1j * rng.standard_normal(D)
        for k in range(codec.n_components):
            y = codec.project(g, 16, k)
            np.testing.assert_allclose(
                codec.project(codec.adjoint(y, 16, k), 16, k), y, atol=1e-10)

    def test_index_bits(self, codec):
        assert codec.index_bits == 1  # K = 2

    def test_save_load_round_trip(self, codec, tmp_path):
        codec.save(tmp_path / "prism")
        loaded = PrismCodec.load(tmp_path / "prism")
        assert loaded.n_components == codec.n_components
        for k in range(codec.n_components):
            np.testing.assert_allclose(loaded.bases[k], codec.bases[k])
            np.testing.assert_allclose(loaded.sigmas[k], codec.sigmas[k])


class TestComplexity:
    def test_direct_and_two_stage(self):
        f_direct = prism_encoder_flops(ANT, N3, m=16, n_components=4)
        f_two = prism_encoder_flops(ANT, N3, m=16, n_components=4, m_sel=4)
        assert f_direct["projection"] == 4 * 16 * D
        assert f_two["projection"] == 4 * 4 * D + 16 * D
        assert f_two["projection"] < f_direct["projection"]

    def test_still_cheaper_than_type2_search(self):
        from nr_csi.ml.projection import type2_select_flops

        for n1, n2 in ((4, 4), (16, 4)):
            ant = AntennaConfig.standard(n1, n2)
            ours = sum(prism_encoder_flops(ant, N3, m=16, n_components=4).values())
            theirs = sum(type2_select_flops(ant, N3, L=4, Mv=2).values())
            assert ours * 3 < theirs, (n1, n2, ours, theirs)
