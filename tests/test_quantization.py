"""Exact-value anchors for the amplitude/phase quantization tables."""

import numpy as np
import pytest

from nr_csi.utils import quantization as qt


def test_r15_wideband_table_exact():
    """Table tabk1: k in 0..7 -> {0, sqrt(1/64), ..., sqrt(1/2), 1}."""
    expected = [0, np.sqrt(1 / 64), np.sqrt(1 / 32), np.sqrt(1 / 16),
                np.sqrt(1 / 8), np.sqrt(1 / 4), np.sqrt(1 / 2), 1]
    assert np.allclose(qt.R15_WB_AMP, expected)


def test_r15_subband_table_exact():
    assert np.allclose(qt.R15_SB_AMP, [np.sqrt(1 / 2), 1.0])


def test_r16_reference_table_exact():
    """Table tabmapkuan: k=0 reserved; spot-check every listed value."""
    assert np.isnan(qt.R16_REF_AMP[0])
    expected = {
        1: 1 / np.sqrt(128), 2: (1 / 8192) ** 0.25, 3: 1 / 8, 4: (1 / 2048) ** 0.25,
        5: 1 / (2 * np.sqrt(8)), 6: (1 / 512) ** 0.25, 7: 1 / 4, 8: (1 / 128) ** 0.25,
        9: 1 / np.sqrt(8), 10: (1 / 32) ** 0.25, 11: 1 / 2, 12: (1 / 8) ** 0.25,
        13: 1 / np.sqrt(2), 14: (1 / 2) ** 0.25, 15: 1,
    }
    for k, v in expected.items():
        assert np.isclose(qt.R16_REF_AMP[k], v), f"k={k}"


def test_r16_differential_table_exact():
    """Table tabmapzhai: k in 0..7 -> {1/(8 sqrt 2), 1/8, ..., 1/sqrt(2), 1}."""
    expected = [1 / (8 * np.sqrt(2)), 1 / 8, 1 / (4 * np.sqrt(2)), 1 / 4,
                1 / (2 * np.sqrt(2)), 1 / 2, 1 / np.sqrt(2), 1]
    assert np.allclose(qt.R16_DIFF_AMP, expected)


@pytest.mark.parametrize("table", [qt.R15_WB_AMP, qt.R15_SB_AMP, qt.R16_DIFF_AMP])
def test_amplitude_quantizer_idempotent(table):
    """Quantizing a table value returns its own index."""
    for k, v in enumerate(table):
        assert qt.quantize_amplitude(v, table) == k


def test_amplitude_quantizer_never_selects_reserved():
    ks = qt.quantize_amplitude(np.linspace(0, 1, 101), qt.R16_REF_AMP)
    assert (ks >= 1).all()


@pytest.mark.parametrize("n_psk", [4, 8, 16])
def test_phase_quantizer_roundtrip(n_psk):
    for c in range(n_psk):
        angle = 2 * np.pi * c / n_psk
        assert qt.quantize_phase(angle, n_psk) == c
        assert np.isclose(qt.phase_value(c, n_psk), np.exp(1j * angle))
    # wrap-around: angles just below 2*pi quantize back to c=0
    assert qt.quantize_phase(2 * np.pi - 1e-9, n_psk) == 0
    # negative angles supported
    assert qt.quantize_phase(-2 * np.pi / n_psk, n_psk) == n_psk - 1
