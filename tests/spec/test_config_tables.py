"""Literal row-by-row anchors for every configuration table in the paper.

Each table is re-transcribed here independently (values read from the paper's
tabNO / tabp / tabp17 / tabpredic / tabCSS) and asserted against the module
constants, so a typo in either copy fails loudly.
"""

from fractions import Fraction as F

import pytest

from nr_csi.config import (
    R16_PARAM_COMBOS,
    R17_PARAM_COMBOS,
    R18_PARAM_COMBOS,
    SUBBAND_SIZES,
    SUPPORTED_N1N2,
    SUPPORTED_NG_N1N2,
    AntennaConfig,
    m_v,
)

# Table tabNO: P_CSI-RS -> [(N1, N2, O1, O2)], all 13 rows.
TABNO = {
    4: [(2, 1, 4, 1)],
    8: [(2, 2, 4, 4), (4, 1, 4, 1)],
    12: [(3, 2, 4, 4), (6, 1, 4, 1)],
    16: [(4, 2, 4, 4), (8, 1, 4, 1)],
    24: [(4, 3, 4, 4), (6, 2, 4, 4), (12, 1, 4, 1)],
    32: [(4, 4, 4, 4), (8, 2, 4, 4), (16, 1, 4, 1)],
}

# Table tabp: paramCombination-r16 -> (L, p_v12, p_v34, beta).
TABP_R16 = {
    1: (2, F(1, 4), F(1, 8), F(1, 4)),
    2: (2, F(1, 4), F(1, 8), F(1, 2)),
    3: (4, F(1, 4), F(1, 8), F(1, 4)),
    4: (4, F(1, 4), F(1, 8), F(1, 2)),
    5: (4, F(1, 4), F(1, 4), F(3, 4)),
    6: (4, F(1, 2), F(1, 4), F(1, 2)),
    7: (6, F(1, 4), None, F(1, 2)),
    8: (6, F(1, 4), None, F(3, 4)),
}

# Table tabp17: paramCombination-r17 -> (M, alpha, beta).
TABP_R17 = {
    1: (1, F(3, 4), F(1, 2)),
    2: (1, F(1, 1), F(1, 2)),
    3: (1, F(1, 1), F(3, 4)),
    4: (1, F(1, 1), F(1, 1)),
    5: (2, F(1, 2), F(1, 2)),
    6: (2, F(3, 4), F(1, 2)),
    7: (2, F(1, 1), F(1, 2)),
    8: (2, F(1, 1), F(3, 4)),
}

# Table tabpredic: paramCombination-Doppler-r18 -> (L, p_v12, p_v34, beta).
TABP_R18 = {
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

# Table tabCSS: BWP size (RBs) -> allowed subband sizes (RBs).
TABCSS = [
    (24, 72, (4, 8)),
    (73, 144, (8, 16)),
    (145, 275, (16, 32)),
]


class TestTabNO:
    def test_all_rows_and_p_csirs_grouping(self):
        flat = {}
        for p, rows in TABNO.items():
            for n1, n2, o1, o2 in rows:
                assert 2 * n1 * n2 == p  # the P_CSI-RS column is consistent
                flat[(n1, n2)] = (o1, o2)
        assert SUPPORTED_N1N2 == flat

    def test_row_count(self):
        assert len(SUPPORTED_N1N2) == 13

    @pytest.mark.parametrize("n1,n2", sorted(SUPPORTED_N1N2))
    def test_standard_constructor_round_trip(self, n1, n2):
        ant = AntennaConfig.standard(n1, n2)
        assert (ant.O1, ant.O2) == SUPPORTED_N1N2[(n1, n2)]
        assert ant.P == 2 * n1 * n2
        assert ant.P in TABNO


def test_type1_multipanel_table_rows():
    expected = {
        (2, 2, 1): (4, 1),
        (2, 4, 1): (4, 1),
        (4, 2, 1): (4, 1),
        (2, 2, 2): (4, 4),
        (2, 8, 1): (4, 1),
        (4, 4, 1): (4, 1),
        (2, 4, 2): (4, 4),
        (4, 2, 2): (4, 4),
    }
    assert SUPPORTED_NG_N1N2 == expected


class TestParamComboTables:
    @pytest.mark.parametrize("idx", sorted(TABP_R16))
    def test_r16_rows(self, idx):
        c = R16_PARAM_COMBOS[idx]
        L, p12, p34, beta = TABP_R16[idx]
        assert (c.L, c.p_v12, c.p_v34, c.beta) == (L, p12, p34, beta)

    def test_r16_row_count(self):
        assert sorted(R16_PARAM_COMBOS) == list(range(1, 9))

    @pytest.mark.parametrize("idx", sorted(TABP_R17))
    def test_r17_rows(self, idx):
        c = R17_PARAM_COMBOS[idx]
        M, alpha, beta = TABP_R17[idx]
        assert (c.M, c.alpha, c.beta) == (M, alpha, beta)

    def test_r17_row_count(self):
        assert sorted(R17_PARAM_COMBOS) == list(range(1, 9))

    @pytest.mark.parametrize("idx", sorted(TABP_R18))
    def test_r18_rows(self, idx):
        c = R18_PARAM_COMBOS[idx]
        L, p12, p34, beta = TABP_R18[idx]
        assert (c.L, c.p_v12, c.p_v34, c.beta) == (L, p12, p34, beta)

    def test_r18_row_count(self):
        assert sorted(R18_PARAM_COMBOS) == list(range(1, 10))

    def test_rank34_unsupported_raises(self):
        for idx in (7, 8):
            with pytest.raises(ValueError):
                R16_PARAM_COMBOS[idx].p_v(3)
        for idx in (8, 9):
            with pytest.raises(ValueError):
                R18_PARAM_COMBOS[idx].p_v(4)


class TestTabCSS:
    def test_rows(self):
        assert len(SUBBAND_SIZES) == len(TABCSS)
        for (rng, sizes), (lo, hi, exp_sizes) in zip(SUBBAND_SIZES, TABCSS):
            assert rng.start == lo and rng.stop == hi + 1
            assert sizes == exp_sizes

    def test_coverage_is_contiguous(self):
        covered = sorted(rb for rng, _ in SUBBAND_SIZES for rb in rng)
        assert covered == list(range(24, 276))


class TestDerivedQuantities:
    def test_paper_worked_example_273_rb(self):
        """273 RB BWP, subband size 16 -> 18 subbands -> M_v in {3, 5}."""
        import math

        n_subbands = math.ceil(273 / 16)
        assert n_subbands == 18
        N3 = n_subbands  # R = 1
        assert m_v(F(1, 4), N3, R=1) == 5
        assert m_v(F(1, 8), N3, R=1) == 3

    @pytest.mark.parametrize("N3,expect", [(12, 3), (18, 5), (36, 9)])
    def test_m_v_quarter(self, N3, expect):
        assert m_v(F(1, 4), N3, R=1) == expect

    def test_m_v_with_r2(self):
        # R = 2: M_v = ceil(p_v * N3 / R); N3 = n_subbands * R
        assert m_v(F(1, 4), 36, R=2) == 5

    @pytest.mark.parametrize("N3", [12, 18, 36])
    def test_r16_k0_all_combos(self, N3):
        """K0 = ceil(beta * 2L * M1) for every paramCombination-r16."""
        import math

        from nr_csi.codebooks.etype2_r16 import R16Type2Codebook

        ant = AntennaConfig.standard(4, 4)  # P = 32: every combination allowed
        for idx, (L, p12, _, beta) in TABP_R16.items():
            M1 = math.ceil(p12 * N3)
            expected = math.ceil(beta * 2 * L * M1)
            ri = [1, 1, 0, 0] if idx in (7, 8) else None
            cbk = R16Type2Codebook(ant, N3=N3, param_combination=idx, ri_restriction=ri)
            assert cbk.K0 == expected, f"combo {idx}, N3={N3}"

    @pytest.mark.parametrize("N3", [12, 18, 36])
    def test_r18_k0_all_combos(self, N3):
        """K0 = ceil(2 * beta * L * M1 * Q) for every Doppler-r18 combination."""
        import math

        from nr_csi.codebooks.etype2_r18 import R18Type2Codebook

        ant = AntennaConfig.standard(4, 4)  # P = 32: every combination allowed
        for idx, (L, p12, _, beta) in TABP_R18.items():
            M1 = math.ceil(p12 * N3)
            ri = [1, 1, 0, 0] if idx in (8, 9) else None
            cbk = R18Type2Codebook(ant, N3=N3, N4=4, param_combination=idx, ri_restriction=ri)
            assert cbk.Q == 2
            assert cbk.K0 == math.ceil(2 * beta * L * M1 * 2), f"combo {idx}, N3={N3}"
            cbk1 = R18Type2Codebook(ant, N3=N3, N4=1, param_combination=idx, ri_restriction=ri)
            assert cbk1.Q == 1
            assert cbk1.K0 == math.ceil(2 * beta * L * M1), f"combo {idx}, N3={N3}, N4=1"
