"""Exhaustive finite-domain tests for the paper's Algorithms 1-4 and bases."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from math import comb

import numpy as np
import pytest

from nr_csi.codebooks import R16Type2Codebook, R18Type2Codebook
from nr_csi.codebooks.fetype2_r17 import R17Type2Codebook
from nr_csi.config import (
    SUPPORTED_N1N2,
    AntennaConfig,
    SubbandConfig,
    m_v,
    n3_for_bwp,
    subband_size_options,
)
from nr_csi.utils import combinatorics as codec
from nr_csi.utils import dft
from nr_csi.utils import quantization as quant
from tests.paper_oracles import paper_combo_decode, paper_combo_index, spatial_beam

CODEC_CASES = [
    (n_total, k)
    for n_total in range(1, 13)
    for k in range(1, min(n_total, 6) + 1)
]


@pytest.mark.parametrize("n_total,k", CODEC_CASES)
def test_generic_combinadic_is_exhaustive_bijection(n_total, k):
    expected_indices = set(range(comb(n_total, k)))
    actual_indices = set()
    for values in itertools.combinations(range(n_total), k):
        expected = paper_combo_index(values, n_total)
        actual = codec.combo_to_index(values, n_total)
        assert actual == expected
        assert codec.index_to_combo(actual, n_total, k) == list(values)
        assert paper_combo_decode(actual, n_total, k) == list(values)
        actual_indices.add(actual)
    assert actual_indices == expected_indices


@pytest.mark.parametrize(
    "indices,n_total",
    [
        ((-1,), 4),
        ((4,), 4),
        ((0, 0), 4),
        ((0, 1, 1), 4),
        ((0, 5), 5),
        ((-2, 3), 8),
    ],
)
def test_generic_combinadic_rejects_every_invalid_index_set(indices, n_total):
    with pytest.raises(ValueError):
        codec.combo_to_index(indices, n_total)


BEAM_CASES = [
    (shape, L)
    for shape in sorted(SUPPORTED_N1N2)
    for L in (1, 2, 3, 4)
    if L <= math.prod(shape)
]


@pytest.mark.parametrize("shape,L", BEAM_CASES)
def test_algorithm1_exhaustive_for_supported_array(shape, L):
    N1, N2 = shape
    for flat in itertools.combinations(range(N1 * N2), L):
        n1 = [n % N1 for n in flat]
        n2 = [n // N1 for n in flat]
        index = codec.encode_beam_combination(n1, n2, N1, N2)
        assert index == paper_combo_index(flat, N1 * N2)
        assert codec.decode_beam_combination(index, N1, N2, L) == (n1, n2)


@pytest.mark.parametrize("O1,O2", [(4, 1), (4, 4)])
def test_algorithm2_exhaustive_group_mapping(O1, O2):
    for groups in itertools.combinations(range(O1 * O2), 4):
        r1 = [g % O1 for g in groups]
        r2 = [g // O1 for g in groups]
        beta1 = codec.encode_restriction_groups(r1, r2, O1, O2)
        assert beta1 == paper_combo_index(groups, O1 * O2)
        assert codec.decode_restriction_groups(beta1, O1, O2) == (list(groups), r1, r2)


SMALL_TAP_CASES = [
    (4, 1),
    (4, 2),
    (5, 2),
    (5, 3),
    (8, 2),
    (8, 3),
    (8, 4),
    (12, 2),
    (12, 3),
    (12, 4),
    (18, 3),
    (18, 5),
    (19, 3),
    (19, 5),
]


@pytest.mark.parametrize("N3,Mv", SMALL_TAP_CASES)
def test_algorithm3_exhaustive_at_and_below_19(N3, Mv):
    for rest in itertools.combinations(range(1, N3), Mv - 1):
        taps = [0, *rest]
        i16, i15 = codec.encode_taps(taps, N3, Mv)
        assert i15 is None
        assert i16 == paper_combo_index(tuple(t - 1 for t in rest), N3 - 1)
        assert codec.decode_taps(i16, N3, Mv) == taps


LARGE_TAP_CASES = [
    (20, 2),
    (20, 3),
    (20, 4),
    (24, 2),
    (24, 3),
    (24, 4),
    (36, 2),
    (36, 3),
    (36, 5),
]


@pytest.mark.parametrize("N3,Mv", LARGE_TAP_CASES)
def test_algorithm3_exhaustive_two_level_code_space(N3, Mv):
    for i15 in range(2 * Mv):
        m_initial = i15 if i15 == 0 else i15 - 2 * Mv
        for i16 in range(comb(2 * Mv - 1, Mv - 1)):
            taps = codec.decode_taps(i16, N3, Mv, i15)
            assert taps[0] == 0
            assert len(taps) == Mv
            assert len(set(taps)) == Mv
            encoded, encoded_i15 = codec.encode_taps(
                taps, N3, Mv, m_initial=m_initial
            )
            assert (encoded, encoded_i15) == (i16, i15)


@pytest.mark.parametrize("N3,Mv", [(20, 2), (20, 5), (36, 3), (36, 5)])
def test_algorithm3_large_n3_requires_window_indicator(N3, Mv):
    with pytest.raises(ValueError, match="required"):
        codec.decode_taps(0, N3, Mv)


PORT_CASES = [
    (P, L)
    for P in (4, 8, 12, 16, 24, 32)
    for L in (1, 2, 4)
    if L <= P // 2
]


@pytest.mark.parametrize("P,L", PORT_CASES)
def test_algorithm4_exhaustive_free_port_mapping(P, L):
    for ports in itertools.combinations(range(P // 2), L):
        index = codec.encode_ports(list(ports), P)
        assert index == paper_combo_index(ports, P // 2)
        assert codec.decode_ports(index, P, L) == list(ports)


DFT_GROUP_CASES = [
    (shape, q1, q2)
    for shape, (O1, O2) in sorted(SUPPORTED_N1N2.items())
    for q1, q2 in {(0, 0), (O1 - 1, O2 - 1)}
]


@pytest.mark.parametrize("shape,q1,q2", DFT_GROUP_CASES)
def test_spatial_dft_group_matches_equation_and_parseval(shape, q1, q2):
    antenna = AntennaConfig.standard(*shape)
    group = dft.orthogonal_group(antenna, q1, q2)
    expected = []
    for n in range(antenna.n_ports_per_pol):
        n1, n2 = n % antenna.N1, n // antenna.N1
        expected.append(
            spatial_beam(
                antenna,
                antenna.O1 * n1 + q1,
                antenna.O2 * n2 + q2,
            )
        )
    expected = np.stack(expected)
    assert np.allclose(group, expected, atol=1e-13)
    assert np.allclose(
        group.conj() @ group.T,
        antenna.n_ports_per_pol * np.eye(antenna.n_ports_per_pol),
        atol=1e-10,
    )
    vector = np.arange(antenna.n_ports_per_pol) + 1j * np.arange(
        antenna.n_ports_per_pol
    )[::-1]
    projections = group.conj() @ vector
    assert np.isclose(
        np.sum(np.abs(projections) ** 2),
        antenna.n_ports_per_pol * np.sum(np.abs(vector) ** 2),
    )


STEERING_CASES = [
    (N, O, index)
    for N, O in ((1, 1), (2, 4), (3, 4), (4, 1), (8, 4), (16, 4))
    for index in (-2, -1, 0, 1, O * N - 1)
]


@pytest.mark.parametrize("N,O,index", STEERING_CASES)
def test_steering_periodicity_and_closed_form(N, O, index):
    expected = np.exp(2j * np.pi * index * np.arange(N) / (O * N))
    assert np.allclose(dft.steering(N, O, index), expected)
    assert np.allclose(dft.steering(N, O, index + O * N), expected)


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 8, 12, 19, 20, 36])
def test_frequency_and_time_dft_full_parseval(N):
    expected = np.exp(
        2j * np.pi * np.outer(np.arange(N), np.arange(N)) / max(N, 1)
    )
    assert np.allclose(dft.freq_basis(N, np.arange(N)), expected)
    assert np.allclose(dft.time_basis(N, np.arange(N)), expected)
    assert np.allclose(expected.conj() @ expected.T, N * np.eye(N), atol=1e-9)


AMPLITUDE_TABLES = [
    pytest.param(quant.R15_WB_AMP, id="r15-wideband"),
    pytest.param(quant.R15_SB_AMP, id="r15-subband"),
    pytest.param(quant.R16_DIFF_AMP, id="r16-differential"),
    pytest.param(quant.R16_REF_AMP[1:], id="r16-reference"),
]
AMPLITUDE_BOUNDARIES = [
    (table, i)
    for table in (
        quant.R15_WB_AMP,
        quant.R15_SB_AMP,
        quant.R16_DIFF_AMP,
        quant.R16_REF_AMP[1:],
    )
    for i in range(len(table) - 1)
]


@pytest.mark.parametrize("table", AMPLITUDE_TABLES)
def test_amplitude_quantizer_is_nearest_neighbor_everywhere(table):
    samples = np.linspace(float(table[0]), float(table[-1]), 1001)
    expected = np.argmin(np.abs(samples[:, None] - table[None, :]), axis=1)
    actual = quant.quantize_amplitude(samples, table)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("table,index", AMPLITUDE_BOUNDARIES)
def test_amplitude_quantizer_boundary_and_midpoint_tie(table, index):
    midpoint = (table[index] + table[index + 1]) / 2
    epsilon = np.spacing(midpoint) * 8
    assert int(quant.quantize_amplitude(midpoint - epsilon, table)) == index
    assert int(quant.quantize_amplitude(midpoint + epsilon, table)) == index + 1
    assert int(quant.quantize_amplitude(midpoint, table)) == index


PHASE_CASES = [(n_psk, c) for n_psk in (4, 8, 16) for c in range(n_psk)]


@pytest.mark.parametrize("n_psk,c", PHASE_CASES)
def test_phase_quantizer_voronoi_cell_and_periodicity(n_psk, c):
    center = 2 * np.pi * c / n_psk
    half_step = np.pi / n_psk
    for offset in (-0.99 * half_step, 0.0, 0.99 * half_step):
        assert int(quant.quantize_phase(center + offset, n_psk)) == c
        assert int(quant.quantize_phase(center + offset + 4 * np.pi, n_psk)) == c
    assert np.isclose(quant.phase_value(c, n_psk), np.exp(1j * center))


SUBBAND_CASES = [
    (n_rb, size, R)
    for n_rb in (24, 72, 73, 144, 145, 275)
    for size in subband_size_options(n_rb)
    for R in (1, 2)
]


@pytest.mark.parametrize("n_rb,size,R", SUBBAND_CASES)
def test_subband_table_edges_and_n3_equation(n_rb, size, R):
    assert n3_for_bwp(n_rb, size, R) == math.ceil(n_rb / size) * R


@pytest.mark.parametrize(
    "fraction,N3,R",
    [
        (Fraction(1, 4), 3, 1),
        (Fraction(1, 8), 18, 1),
        (Fraction(1, 4), 36, 2),
        (Fraction(1, 2), 19, 1),
        (Fraction(1, 16), 36, 2),
    ],
)
def test_mv_matches_paper_ceiling_equation(fraction, N3, R):
    assert m_v(fraction, N3, R) == math.ceil(float(fraction) * N3 / R)


@pytest.mark.parametrize("n_subbands", [0, -1])
def test_subband_configuration_rejects_nonpositive_count(n_subbands):
    with pytest.raises(ValueError):
        SubbandConfig(n_subbands=n_subbands)


@pytest.mark.parametrize("R", [0, 3])
def test_r16_and_r18_reject_r_outside_protocol_domain(R):
    antenna = AntennaConfig.standard(4, 2)
    with pytest.raises(ValueError):
        R16Type2Codebook(antenna, N3=12, R=R)
    with pytest.raises(ValueError):
        R18Type2Codebook(antenna, N3=12, N4=4, R=R)


def test_r17_rejects_two_taps_when_only_one_frequency_unit_exists():
    antenna = AntennaConfig.standard(4, 2)
    with pytest.raises(ValueError):
        R17Type2Codebook(antenna, N3=1, param_combination=5)
