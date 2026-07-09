"""Fixed Lloyd-Max quantizer: known optima, distortion, bit-exact packing."""

import numpy as np
import pytest

from nr_csi.ml.quantizer import (
    dequantize,
    dequantize_complex,
    lloyd_max,
    pack_indices,
    quantize,
    quantize_complex,
    quantizer_mse,
    unpack_indices,
)


class TestLloydMax:
    def test_one_bit_is_sign_times_mean(self):
        levels, bounds = lloyd_max(1)
        np.testing.assert_allclose(levels, [-np.sqrt(2 / np.pi), np.sqrt(2 / np.pi)],
                                   atol=1e-9)
        np.testing.assert_allclose(bounds, [0.0], atol=1e-12)

    def test_two_bit_classic_values(self):
        levels, _ = lloyd_max(2)  # Max (1960): +-0.4528, +-1.5104
        np.testing.assert_allclose(np.abs(levels), [1.5104, 0.4528, 0.4528, 1.5104],
                                   atol=2e-4)

    def test_symmetry_and_monotonicity(self):
        for b in (1, 2, 3, 4, 5, 6):
            levels, bounds = lloyd_max(b)
            np.testing.assert_allclose(levels, -levels[::-1], atol=1e-9)
            assert np.all(np.diff(levels) > 0) and np.all(np.diff(bounds) > 0)

    def test_distortion_matches_classic_tables(self):
        # Max (1960) / Lloyd (1982) Gaussian distortions
        for bits, d_ref in ((1, 0.3634), (2, 0.1175), (3, 0.03454), (4, 0.009497)):
            assert quantizer_mse(bits) == pytest.approx(d_ref, rel=2e-3)

    def test_empirical_distortion(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200_000)
        for bits in (2, 3, 4):
            err = np.mean((dequantize(quantize(x, bits), bits) - x) ** 2)
            assert err == pytest.approx(quantizer_mse(bits), rel=2e-2)


class TestPacking:
    def test_round_trip_and_bit_exact_length(self):
        rng = np.random.default_rng(1)
        for bits in (2, 3, 4, 5):
            idx = rng.integers(0, 1 << bits, size=48).astype(np.uint8)
            stream = pack_indices(idx, bits)
            assert set(stream) <= {"0", "1"}
            assert len(stream) == idx.size * bits  # honest overhead accounting
            np.testing.assert_array_equal(unpack_indices(stream, bits), idx)

    def test_complex_round_trip(self):
        rng = np.random.default_rng(2)
        u = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        idx = quantize_complex(u, 3)
        assert idx.shape == (32,)
        u_hat = dequantize_complex(idx, 3)
        assert u_hat.shape == (16,)
        # dequantized values are the nearest levels of the true parts
        assert np.mean(np.abs(u_hat - u) ** 2) < 2.5 * quantizer_mse(3)

    def test_bad_stream_length(self):
        with pytest.raises(ValueError):
            unpack_indices("0101", 3)
