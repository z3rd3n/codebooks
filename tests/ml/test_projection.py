"""GLIMPSE measurement projection: unitarity, nesting, transforms, FLOPs."""

import numpy as np
import pytest

from nr_csi.config import AntennaConfig
from nr_csi.ml.projection import (
    GlimpseCodec,
    encoder_flops,
    measurement_matrix,
    type2_select_flops,
)

ANT = AntennaConfig.standard(4, 4)  # P = 32
N3 = 8


@pytest.fixture(scope="module")
def codec() -> GlimpseCodec:
    return GlimpseCodec(ANT, N3, m_max=48, seed=0)


class TestMeasurementMatrix:
    def test_rows_orthonormal(self):
        A = measurement_matrix(64, 24, seed=3)
        np.testing.assert_allclose(A @ A.conj().T, np.eye(24), atol=1e-10)

    def test_prefix_nested_and_deterministic(self):
        A1 = measurement_matrix(64, 16, seed=5)
        A2 = measurement_matrix(64, 48, seed=5)
        np.testing.assert_array_equal(A1, A2[:16])  # rateless prefix property
        np.testing.assert_array_equal(A1, measurement_matrix(64, 16, seed=5))

    def test_seed_changes_matrix(self):
        assert not np.allclose(
            measurement_matrix(64, 8, seed=0), measurement_matrix(64, 8, seed=1)
        )

    def test_bounds(self):
        with pytest.raises(ValueError):
            measurement_matrix(64, 65)


class TestAngleDelayTransform:
    def test_round_trip(self, codec):
        rng = np.random.default_rng(0)
        V = rng.standard_normal((N3, ANT.P, 2)) + 1j * rng.standard_normal((N3, ANT.P, 2))
        g = codec.targets_to_vec(V)
        assert g.shape == (2, codec.D)
        np.testing.assert_allclose(codec.vec_to_targets(g), V, atol=1e-12)

    def test_unitary(self, codec):
        rng = np.random.default_rng(1)
        V = rng.standard_normal((N3, ANT.P)) + 1j * rng.standard_normal((N3, ANT.P))
        g = codec.targets_to_vec(V)
        assert g.shape == (codec.D,)
        np.testing.assert_allclose(np.linalg.norm(g), np.linalg.norm(V), rtol=1e-12)

    def test_on_grid_beam_is_sparse(self, codec):
        """A DFT beam constant across frequency maps to a single angle-delay
        coefficient per polarization -- the sparsity the decoder prior uses."""
        from nr_csi.utils import dft

        beam = dft.spatial_beam(ANT, 0, 0)  # (P/2,), one orthogonal-group beam
        v = np.concatenate([beam, beam])  # both polarizations
        V = np.repeat(v[None, :], N3, axis=0)  # constant across frequency
        g = codec.targets_to_vec(V)
        energy = np.abs(g) ** 2
        top2 = np.sort(energy)[::-1][:2]
        assert top2.sum() > 0.999 * energy.sum()  # 2 pols -> 2 coefficients


class TestProjection:
    def test_adjoint_consistency(self, codec):
        rng = np.random.default_rng(2)
        g = rng.standard_normal(codec.D) + 1j * rng.standard_normal(codec.D)
        y = codec.project(g, 24)
        # orthonormal rows: measuring the back-projection reproduces y
        np.testing.assert_allclose(codec.project(codec.adjoint(y, 24), 24), y, atol=1e-10)

    def test_standardize_round_trip(self, codec):
        rng = np.random.default_rng(3)
        y = 5.0 * (rng.standard_normal(16) + 1j * rng.standard_normal(16))
        np.testing.assert_allclose(codec.destandardize(codec.standardize(y)), y,
                                   atol=1e-12)

    def test_gaussianization(self, codec):
        """For the *random* basis, measurement entries are near-Gaussian
        regardless of the input's distribution (here: a sparse spiky vector) --
        the property that lets a single fixed scalar quantizer serve any
        deployment even without a fitted basis."""
        rng = np.random.default_rng(4)
        g = np.zeros(codec.D, complex)
        support = rng.choice(codec.D, 6, replace=False)
        g[support] = rng.standard_normal(6) * 10 + 1j * rng.standard_normal(6)
        u = codec.project(g, 48)
        parts = np.concatenate([u.real, u.imag])
        kurt = np.mean(parts**4) / np.mean(parts**2) ** 2
        assert abs(kurt - 3.0) < 1.2  # Gaussian kurtosis = 3


class TestKLT:
    def test_klt_captures_more_than_random(self, codec):
        """The fitted KLT basis captures far more energy per measurement than
        the distribution-blind random basis -- the reason GLIMPSE uses it."""
        from nr_csi.ml.projection import GlimpseCodec, fit_klt

        rng = np.random.default_rng(0)
        # A population with consistent low-dimensional structure (channels live
        # near a fixed angle-delay subspace) -- the regime KLT exploits and a
        # distribution-blind random basis cannot.
        n, r = 4000, 20
        D = codec.D
        basis = rng.standard_normal((D, r)) + 1j * rng.standard_normal((D, r))
        coeff = (rng.standard_normal((n, r)) + 1j * rng.standard_normal((n, r))) * \
            (1.0 / (1 + np.arange(r)))  # decaying spectrum
        g = coeff @ basis.T
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        A, sigma = fit_klt(g, m_max=48)
        klt = GlimpseCodec(ANT, N3, m_max=48, basis_matrix=A, sigma=sigma)
        gv = g[:500]
        m = 16

        def captured(cod):
            proj = cod.adjoint(cod.project(gv, m), m)
            num = np.abs(np.sum(gv.conj() * proj, axis=1)) ** 2
            den = np.sum(np.abs(gv) ** 2, 1) * np.sum(np.abs(proj) ** 2, 1)
            return float(np.mean(num / np.maximum(den, 1e-12)))

        assert captured(klt) > 2 * captured(codec)
        # variance-ordered: sigma is non-increasing
        assert np.all(np.diff(sigma) <= 1e-9)

    def test_klt_prefix_nested_and_saves(self, tmp_path):
        from nr_csi.ml.projection import GlimpseCodec, fit_klt

        rng = np.random.default_rng(1)
        g = rng.standard_normal((2000, ANT.P * N3)) + 1j * rng.standard_normal(
            (2000, ANT.P * N3))
        A, sigma = fit_klt(g, m_max=40)
        klt = GlimpseCodec(ANT, N3, m_max=40, basis_matrix=A, sigma=sigma)
        # prefix nesting: A[:m] is the first m principal rows
        np.testing.assert_array_equal(klt.A[:16], A[:16])
        # save/load round-trips the published constant
        klt.save(tmp_path / "codec")
        reloaded = GlimpseCodec.load(tmp_path / "codec")
        np.testing.assert_allclose(reloaded.A, klt.A)
        np.testing.assert_allclose(reloaded._sigma, klt._sigma)


class TestComplexity:
    def test_glimpse_encoder_cheaper_than_type2_search(self):
        """The fixed encoder costs a small fraction of the R16 UE search it
        replaces (shared eigen step excluded), and the advantage *grows* with
        the array: the search scans O1*O2 oversampled groups (~P^2) while the
        projection is m*D (~P)."""
        ratios = []
        for n1, n2 in ((4, 2), (4, 4), (16, 2)):
            ant = AntennaConfig.standard(n1, n2)
            ours = sum(encoder_flops(ant, N3, m=32).values())
            theirs = sum(type2_select_flops(ant, N3, L=4, Mv=2).values())
            assert ours * 3 < theirs, (n1, n2, ours, theirs)
            ratios.append(theirs / ours)
        assert ratios[0] < ratios[1] < ratios[2]  # P = 16 -> 32 -> 64
        assert ratios[2] > 15
