"""Spec-anchored conformance tests against 3GPP TS 38.214 v19.3.0 (``j30``).

Why this module exists
----------------------
``test_config_tables.py`` and ``test_quantization.py`` pin the implementation
against values transcribed from *the tutorial paper*.  That catches a typo in
``config.py``/``quantization.py`` -- but it cannot catch a case where the paper
itself diverges from the 3GPP specification, because both copies would agree.

This module closes that gap: every constant below is transcribed **directly
from TS 38.214 v19.3.0** (the ``38214-j30.md`` conversion under
``specvert/specs``), with the clause/table number cited next to it.  Values
that the spec renders as equation images (rather than text) were read off the
rendered images in ``specs/assets`` and are flagged ``# OCR``.  If the
implementation, the paper, or a future spec edition ever drifts from the
letter of TS 38.214, the relevant assertion fails with its citation.

Scope: every implemented codebook type (Type I single/multi-panel R15,
Type II R15, eType II R16, feType II R17, eType II Doppler/predicted R18).
"""

from __future__ import annotations

from fractions import Fraction as F
from math import comb

import numpy as np
import pytest

from nr_csi.codebooks.etype2_r16 import R16Type2Codebook, decode_i18, encode_i18
from nr_csi.codebooks.etype2_r18 import R18Type2Codebook
from nr_csi.codebooks.fetype2_r17 import R17Type2Codebook
from nr_csi.codebooks.type1 import i13_offsets, i13_offsets_rank34
from nr_csi.codebooks.type1_multipanel import i13_offsets_multipanel_rank34
from nr_csi.codebooks.type2_r15 import R15Type2Codebook, k2_cap
from nr_csi.config import (
    R16_PARAM_COMBOS,
    R17_PARAM_COMBOS,
    R18_PARAM_COMBOS,
    SUPPORTED_N1N2,
    SUPPORTED_NG_N1N2,
    AntennaConfig,
    m_v,
)
from nr_csi.utils import quantization as qt

# ---------------------------------------------------------------------------
# Type I single-panel -- Table 5.2.2.2.1-2 (supported (N1,N2) -> (O1,O2))
# ---------------------------------------------------------------------------

# Transcribed verbatim from TS 38.214 Table 5.2.2.2.1-2 (j30 lines 5549-5617).
# Columns: P_CSI-RS | (N1,N2) | (O1,O2).
SPEC_TABLE_5_2_2_2_1_2 = [
    (4, (2, 1), (4, 1)),
    (8, (2, 2), (4, 4)),
    (8, (4, 1), (4, 1)),
    (12, (3, 2), (4, 4)),
    (12, (6, 1), (4, 1)),
    (16, (4, 2), (4, 4)),
    (16, (8, 1), (4, 1)),
    (24, (4, 3), (4, 4)),
    (24, (6, 2), (4, 4)),
    (24, (12, 1), (4, 1)),
    (32, (4, 4), (4, 4)),
    (32, (8, 2), (4, 4)),
    (32, (16, 1), (4, 1)),
]


def test_table_5_2_2_2_1_2_supported_configs():
    spec = {n1n2: o1o2 for _, n1n2, o1o2 in SPEC_TABLE_5_2_2_2_1_2}
    assert SUPPORTED_N1N2 == spec
    # P_CSI-RS = 2*N1*N2 column is self-consistent in the spec table.
    for p, (n1, n2), _ in SPEC_TABLE_5_2_2_2_1_2:
        assert 2 * n1 * n2 == p
        assert AntennaConfig.standard(n1, n2).P == p


# ---------------------------------------------------------------------------
# Type I single-panel -- Table 5.2.2.2.1-3 (rank-2 i13 -> (k1,k2))
# Table 5.2.2.2.1-4 (rank-3/4, P < 16).  Both OCR-confirmed from j30 assets.
# ---------------------------------------------------------------------------


def spec_rank2_offsets(N1, N2, O1, O2):
    """TS 38.214 Table 5.2.2.2.1-3, four (N1,N2) categories.  # OCR (assets
    image211-221: headers N1>N2>1 / N1=N2 / N1=2,N2=1 / N1>2,N2=1)."""
    if N1 == 2 and N2 == 1:
        return [(0, 0), (O1, 0)]
    if N1 > 2 and N2 == 1:
        return [(0, 0), (O1, 0), (2 * O1, 0), (3 * O1, 0)]
    if N1 == N2:
        return [(0, 0), (O1, 0), (0, O2), (O1, O2)]
    if N1 > N2 > 1:
        return [(0, 0), (O1, 0), (0, O2), (2 * O1, 0)]
    raise AssertionError("unreachable")


# TS 38.214 Table 5.2.2.2.1-4 (P < 16).  # OCR (assets image213/222/223/224/225
# headers (2,1)/(4,1)/(6,1)/(2,2)/(3,2); values 2O1=image226, 3O1=image227,
# 4O1=image228).
SPEC_TABLE_5_2_2_2_1_4 = {
    (2, 1): lambda O1, O2: [(O1, 0)],
    (4, 1): lambda O1, O2: [(O1, 0), (2 * O1, 0), (3 * O1, 0)],
    (6, 1): lambda O1, O2: [(O1, 0), (2 * O1, 0), (3 * O1, 0), (4 * O1, 0)],
    (2, 2): lambda O1, O2: [(O1, 0), (0, O2), (O1, O2)],
    (3, 2): lambda O1, O2: [(O1, 0), (0, O2), (O1, O2), (2 * O1, 0)],
}


@pytest.mark.parametrize("n1,n2", sorted(SUPPORTED_N1N2))
def test_table_5_2_2_2_1_3_rank2_offsets(n1, n2):
    O1, O2 = SUPPORTED_N1N2[(n1, n2)]
    assert i13_offsets(n1, n2, O1, O2) == spec_rank2_offsets(n1, n2, O1, O2)


@pytest.mark.parametrize("n1,n2", sorted(SPEC_TABLE_5_2_2_2_1_4))
def test_table_5_2_2_2_1_4_rank34_small_offsets(n1, n2):
    O1, O2 = SUPPORTED_N1N2[(n1, n2)]
    assert i13_offsets_rank34(n1, n2, O1, O2) == SPEC_TABLE_5_2_2_2_1_4[(n1, n2)](O1, O2)


def test_table_5_2_2_2_1_4_only_lists_below_16_ports():
    # The small-port branch is defined only for the (N1,N2) with 2*N1*N2 < 16;
    # 16-port (4,2)/(8,1) use the half-split "large" codebook instead.
    for (n1, n2) in SPEC_TABLE_5_2_2_2_1_4:
        assert 2 * n1 * n2 < 16


# ---------------------------------------------------------------------------
# Type I multi-panel -- Table 5.2.2.2.2-1 (supported (Ng,N1,N2) -> (O1,O2))
# and Table 5.2.2.2.2-2 (rank-3/4 offsets).  Both transcribed from j30.
# ---------------------------------------------------------------------------

# TS 38.214 Table 5.2.2.2.2-1 (j30 lines 6838-6881): P_CSI-RS = 2*Ng*N1*N2.
SPEC_TABLE_5_2_2_2_2_1 = {
    (2, 2, 1): (4, 1),
    (2, 4, 1): (4, 1),
    (4, 2, 1): (4, 1),
    (2, 2, 2): (4, 4),
    (2, 8, 1): (4, 1),
    (4, 4, 1): (4, 1),
    (2, 4, 2): (4, 4),
    (4, 2, 2): (4, 4),
}

# TS 38.214 Table 5.2.2.2.2-2 (rank 3/4).  # OCR (assets image213/222/391/224/392
# headers (2,1)/(4,1)/(8,1)/(2,2)/(4,2)).
SPEC_TABLE_5_2_2_2_2_2 = {
    (2, 1): lambda O1, O2: [(O1, 0)],
    (4, 1): lambda O1, O2: [(O1, 0), (2 * O1, 0), (3 * O1, 0)],
    (8, 1): lambda O1, O2: [(O1, 0), (2 * O1, 0), (3 * O1, 0), (4 * O1, 0)],
    (2, 2): lambda O1, O2: [(O1, 0), (0, O2), (O1, O2)],
    (4, 2): lambda O1, O2: [(O1, 0), (0, O2), (O1, O2), (2 * O1, 0)],
}


def test_table_5_2_2_2_2_1_multipanel_configs():
    assert SUPPORTED_NG_N1N2 == SPEC_TABLE_5_2_2_2_2_1
    for (ng, n1, n2) in SPEC_TABLE_5_2_2_2_2_1:
        assert AntennaConfig.standard(n1, n2, Ng=ng).P == 2 * ng * n1 * n2


@pytest.mark.parametrize("n1,n2", sorted(SPEC_TABLE_5_2_2_2_2_2))
def test_table_5_2_2_2_2_2_multipanel_rank34_offsets(n1, n2):
    O1, O2 = (4, 1) if n2 == 1 else (4, 4)
    assert (
        i13_offsets_multipanel_rank34(n1, n2, O1, O2)
        == SPEC_TABLE_5_2_2_2_2_2[(n1, n2)](O1, O2)
    )


# ---------------------------------------------------------------------------
# Type II R15 -- amplitude tables and the subband-coefficient cap.
# ---------------------------------------------------------------------------


def test_table_5_2_2_2_3_2_wideband_amplitude():
    """TS 38.214 Table 5.2.2.2.3-2: k^(1) in 0..7 -> p^(1).

    p^(1) = sqrt(1/4)^(7-k), i.e. {0, 1/8, 1/(4 sqrt2), 1/4, 1/(2 sqrt2),
    1/2, 1/sqrt2, 1} for k=0..7 (k=0 maps to 0)."""
    expected = [
        0.0,
        np.sqrt(1 / 64),
        np.sqrt(1 / 32),
        np.sqrt(1 / 16),
        np.sqrt(1 / 8),
        np.sqrt(1 / 4),
        np.sqrt(1 / 2),
        1.0,
    ]
    assert np.allclose(qt.R15_WB_AMP, expected)


def test_table_5_2_2_2_3_3_subband_amplitude():
    """TS 38.214 Table 5.2.2.2.3-3: k^(2) in {0,1} -> {1/sqrt2, 1}."""
    assert np.allclose(qt.R15_SB_AMP, [np.sqrt(1 / 2), 1.0])


def test_table_5_2_2_2_3_4_subband_coefficient_cap():
    """TS 38.214 Table 5.2.2.2.3-4 (j30 lines 7636-7655): K2 = 4,4,6 for L=2,3,4."""
    assert (k2_cap(2), k2_cap(3), k2_cap(4)) == (4, 4, 6)


@pytest.mark.parametrize("n_psk", [4, 8])
def test_r15_phase_alphabet_sizes_accepted(n_psk):
    """Clause 5.2.2.2.3: phaseAlphabetSize in {4, 8}."""
    R15Type2Codebook(AntennaConfig.standard(4, 2), N3=1, L=4, n_psk=n_psk)


@pytest.mark.parametrize("n_psk", [2, 16, 32])
def test_r15_invalid_phase_alphabet_rejected(n_psk):
    with pytest.raises(ValueError):
        R15Type2Codebook(AntennaConfig.standard(4, 2), N3=1, L=4, n_psk=n_psk)


# ---------------------------------------------------------------------------
# Combinatorial coefficients C(x,y) -- Tables 5.2.2.2.3-1 / 5.2.2.2.5-4.
# ---------------------------------------------------------------------------


def test_table_5_2_2_2_3_1_combinatorial_coefficients():
    """TS 38.214 Table 5.2.2.2.3-1 (x=0..15, y=1..4), transcribed verbatim
    (j30 lines 7525-7540).  C(x,y) is the ordinary binomial coefficient
    (0 when x < y), which is the base of the combinatorial number system used
    by Algorithm 1/3/4 (``math.comb``)."""
    rows = {
        0: [0, 0, 0, 0],
        1: [1, 0, 0, 0],
        2: [2, 1, 0, 0],
        3: [3, 3, 1, 0],
        4: [4, 6, 4, 1],
        5: [5, 10, 10, 5],
        6: [6, 15, 20, 15],
        7: [7, 21, 35, 35],
        8: [8, 28, 56, 70],
        9: [9, 36, 84, 126],
        10: [10, 45, 120, 210],
        11: [11, 55, 165, 330],
        12: [12, 66, 220, 495],
        13: [13, 78, 286, 715],
        14: [14, 91, 364, 1001],
        15: [15, 105, 455, 1365],
    }
    for x, vals in rows.items():
        for y, c in enumerate(vals, start=1):
            assert comb(x, y) == c, f"C({x},{y})"


def test_table_5_2_2_2_5_4_combinatorial_high_columns():
    """TS 38.214 Table 5.2.2.2.5-4 extends C(x,y) to y=1..9 (j30 lines
    8155-8168); spot-check the y in {5..9} columns the R16 tap codec needs."""
    assert comb(9, 5) == 126
    assert comb(13, 6) == 1716
    assert comb(13, 7) == 1716
    assert comb(13, 9) == 715


# ---------------------------------------------------------------------------
# eType II R16 -- Table 5.2.2.2.5-1, amplitude tables, ranges, K0, i_{1,8,l}.
# ---------------------------------------------------------------------------

# TS 38.214 Table 5.2.2.2.5-1 (j30 lines 7858-7927): idx -> (L, p_v12, p_v34, beta).
SPEC_TABLE_5_2_2_2_5_1 = {
    1: (2, F(1, 4), F(1, 8), F(1, 4)),
    2: (2, F(1, 4), F(1, 8), F(1, 2)),
    3: (4, F(1, 4), F(1, 8), F(1, 4)),
    4: (4, F(1, 4), F(1, 8), F(1, 2)),
    5: (4, F(1, 4), F(1, 4), F(3, 4)),
    6: (4, F(1, 2), F(1, 4), F(1, 2)),
    7: (6, F(1, 4), None, F(1, 2)),
    8: (6, F(1, 4), None, F(3, 4)),
}


@pytest.mark.parametrize("idx", sorted(SPEC_TABLE_5_2_2_2_5_1))
def test_table_5_2_2_2_5_1_r16_param_combos(idx):
    L, p12, p34, beta = SPEC_TABLE_5_2_2_2_5_1[idx]
    c = R16_PARAM_COMBOS[idx]
    assert (c.L, c.p_v12, c.p_v34, c.beta) == (L, p12, p34, beta)


def test_table_5_2_2_2_5_3_differential_amplitude():
    """TS 38.214 Table 5.2.2.2.5-3 (j30 lines 8087-8124): k^(2) in 0..7 ->
    {1/(8 sqrt2), 1/8, 1/(4 sqrt2), 1/4, 1/(2 sqrt2), 1/2, 1/sqrt2, 1}."""
    expected = [
        1 / (8 * np.sqrt(2)),
        1 / 8,
        1 / (4 * np.sqrt(2)),
        1 / 4,
        1 / (2 * np.sqrt(2)),
        1 / 2,
        1 / np.sqrt(2),
        1.0,
    ]
    assert np.allclose(qt.R16_DIFF_AMP, expected)


def test_table_5_2_2_2_5_2_reference_amplitude():
    """TS 38.214 Table 5.2.2.2.5-2: k^(1)=0 is Reserved, k^(1)=15 -> 1, and
    the 15 levels are the 1/4-octave geometric ladder 2^(-(15-k)/4)."""
    assert np.isnan(qt.R16_REF_AMP[0])  # k=0 Reserved
    assert np.isclose(qt.R16_REF_AMP[15], 1.0)
    for k in range(1, 16):
        assert np.isclose(qt.R16_REF_AMP[k], 2.0 ** (-(15 - k) / 4)), f"k={k}"


def test_r16_coefficient_index_ranges():
    """Clause 5.2.2.2.5: k^(1) in {1..15} (4-bit, 0 reserved), k^(2) in {0..7}
    (3-bit), phase c in {0..15} (4-bit, N_PSK=16)."""
    assert R16Type2Codebook.N_PSK == 16
    assert len(qt.R16_REF_AMP) == 16 and np.isnan(qt.R16_REF_AMP[0])  # k^(1): 1..15
    assert len(qt.R16_DIFF_AMP) == 8  # k^(2): 0..7


@pytest.mark.parametrize("N3", [12, 18, 36])
@pytest.mark.parametrize("idx", sorted(SPEC_TABLE_5_2_2_2_5_1))
def test_r16_K0_uses_M1_not_Mv(N3, idx):
    """Clause 5.2.2.2.5: K0 = ceil(beta * 2L * M1) -- the rank-1 tap count M1,
    NOT the per-rank M_v (j30 line 8005: ``Let K0 = ceil(beta 2L M_1)``)."""
    L, p12, _, beta = SPEC_TABLE_5_2_2_2_5_1[idx]
    M1 = m_v(p12, N3, 1)
    cbk = R16Type2Codebook(AntennaConfig.standard(4, 2), N3=N3, param_combination=idx)
    assert cbk.K0 == int(np.ceil(beta * 2 * L * M1))


def test_r16_Mv_formula():
    """Clause 5.2.2.2.5: M_v = ceil(p_v * N3 / R)."""
    assert m_v(F(1, 4), 18, 1) == 5
    assert m_v(F(1, 8), 18, 1) == 3
    assert m_v(F(1, 4), 36, 2) == 5


def test_r16_i18_strongest_coefficient_dual_mode():
    """Clause 5.2.2.2.5 (j30 line 8043): the strongest-coefficient indicator is
    piecewise --
        v = 1: i_{1,8,l} = sum_{i=0}^{i*} k^(3)_{1,i,0} - 1   (cumulative count)
        v > 1: i_{1,8,l} = i*_l                               (beam index)
    """
    # f=0 bitmap row (Mv=1, 2L=8) with nonzero coefficients at i in {0,2,3,5}.
    bitmap = np.zeros((1, 8), dtype=bool)
    bitmap[0, [0, 2, 3, 5]] = True
    i_star = 3
    # rank 1: positions 0,2,3 are set up to and including i*=3 -> count 3, minus 1.
    assert encode_i18(bitmap, i_star, rank=1) == 2
    assert decode_i18(2, bitmap, rank=1) == i_star
    # rank > 1: the indicator is the beam index directly.
    assert encode_i18(bitmap, i_star, rank=2) == i_star
    assert decode_i18(i_star, bitmap, rank=2) == i_star


# ---------------------------------------------------------------------------
# feType II R17 -- Table 5.2.2.2.7-1 and the K1 = alpha*P relation.
# ---------------------------------------------------------------------------

# TS 38.214 Table 5.2.2.2.7-1 (j30 lines 8470-8479): idx -> (M, alpha, beta).
SPEC_TABLE_5_2_2_2_7_1 = {
    1: (1, F(3, 4), F(1, 2)),
    2: (1, F(1, 1), F(1, 2)),
    3: (1, F(1, 1), F(3, 4)),
    4: (1, F(1, 1), F(1, 1)),
    5: (2, F(1, 2), F(1, 2)),
    6: (2, F(3, 4), F(1, 2)),
    7: (2, F(1, 1), F(1, 2)),
    8: (2, F(1, 1), F(3, 4)),
}


@pytest.mark.parametrize("idx", sorted(SPEC_TABLE_5_2_2_2_7_1))
def test_table_5_2_2_2_7_1_r17_param_combos(idx):
    M, alpha, beta = SPEC_TABLE_5_2_2_2_7_1[idx]
    c = R17_PARAM_COMBOS[idx]
    assert (c.M, c.alpha, c.beta) == (M, alpha, beta)


@pytest.mark.parametrize("idx", sorted(SPEC_TABLE_5_2_2_2_7_1))
def test_r17_K1_equals_alpha_P_and_L_is_half(idx):
    """Clause 5.2.2.2.7: L = K1/2 with K1 = alpha * P_CSI-RS; K0 = ceil(beta K1 M)."""
    M, alpha, beta = SPEC_TABLE_5_2_2_2_7_1[idx]
    ant = AntennaConfig.standard(4, 2)  # P = 16
    cbk = R17Type2Codebook(ant, N3=12, param_combination=idx)
    assert cbk.K1 == int(alpha * ant.P)
    assert cbk.L == cbk.K1 // 2
    assert cbk.N_PSK == 16
    assert cbk.K0 == int(np.ceil(beta * cbk.K1 * M))


# ---------------------------------------------------------------------------
# eType II Doppler R18 -- Table 5.2.2.2.10-1 and K0 (with the Q Doppler factor).
# ---------------------------------------------------------------------------

# TS 38.214 Table 5.2.2.2.10-1 (j30 lines 9892-9966): idx -> (L, p_v12, p_v34, beta).
SPEC_TABLE_5_2_2_2_10_1 = {
    1: (2, F(1, 8), F(1, 16), F(1, 4)),
    2: (2, F(1, 4), F(1, 8), F(1, 2)),
    3: (4, F(1, 4), F(1, 8), F(1, 4)),
    4: (4, F(1, 4), F(1, 4), F(1, 4)),
    5: (4, F(1, 4), F(1, 4), F(1, 2)),
    6: (4, F(1, 4), F(1, 4), F(3, 4)),
    7: (4, F(1, 2), F(1, 4), F(1, 2)),
    8: (6, F(1, 4), None, F(1, 2)),
    9: (6, F(1, 4), None, F(3, 4)),
}


@pytest.mark.parametrize("idx", sorted(SPEC_TABLE_5_2_2_2_10_1))
def test_table_5_2_2_2_10_1_r18_param_combos(idx):
    L, p12, p34, beta = SPEC_TABLE_5_2_2_2_10_1[idx]
    c = R18_PARAM_COMBOS[idx]
    assert (c.L, c.p_v12, c.p_v34, c.beta) == (L, p12, p34, beta)


@pytest.mark.parametrize("idx", sorted(SPEC_TABLE_5_2_2_2_10_1))
@pytest.mark.parametrize("N4,Q", [(1, 1), (4, 2)])
def test_r18_K0_scales_with_Q(idx, N4, Q):
    """Clause 5.2.2.2.10: Q = 1 (N4=1) degenerates to R16; Q = 2 otherwise, with
    K0 = ceil(2 * beta * L * M1 * Q)."""
    L, p12, _, beta = SPEC_TABLE_5_2_2_2_10_1[idx]
    M1 = m_v(p12, 18, 1)
    cbk = R18Type2Codebook(AntennaConfig.standard(4, 2), N3=18, N4=N4, param_combination=idx)
    assert cbk.Q == Q
    assert cbk.K0 == int(np.ceil(2 * beta * L * M1 * Q))
