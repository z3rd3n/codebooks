"""Overhead anchors: Table bit1/bit2 formulas, f2 qualitative claims,
consistency between nominal formulas and the codebook classes."""

import numpy as np
import pytest

from nr_csi.config import AntennaConfig
from nr_csi.metrics.overhead import f2_comparison, r15_bits, r16_bits, r18_bits

F2_ANT = AntennaConfig.standard(16, 1)  # N1N2 = 16, O1O2 = 4 (paper Fig. f2 setup)


class TestTableFormulas:
    """Spot-check the per-element bit counts of Tables bit1/bit2 (v=2)."""

    def test_r15_elements(self):
        bits = r15_bits(F2_ANT, L=4, v=2, N3=18, n_psk=4, K2=6, Ml=8)
        assert bits["i11"] == 2  # ceil(log2(4))
        assert bits["i12"] == 11  # ceil(log2 C(16,4)) = ceil(log2 1820)
        assert bits["i13"] == 2 * 3  # ceil(log2 8)
        assert bits["i14"] == 2 * 21  # 3*(2L-1)
        # per subband: (min(Ml,K2)-1)*log2(4) + 2*(Ml-min) = 5*2 + 2*2 = 14
        assert bits["i21"] == 2 * 18 * 14
        assert bits["i22"] == 2 * 18 * 5  # min(Ml,K2)-1
    def test_r16_elements_small_and_large_n3(self):
        bits = r16_bits(F2_ANT, L=4, v=2, N3=18, Mv=5, K_nz=20)
        assert bits["i16"] == 2 * int(np.ceil(np.log2(2380)))  # C(17,4)
        assert "i15" not in bits
        assert bits["i17"] == 2 * 2 * 4 * 5  # v*2L*Mv
        assert bits["i18"] == 2 * 3
        assert (bits["i23"], bits["i24"], bits["i25"]) == (8, 54, 72)
        big = r16_bits(F2_ANT, L=4, v=2, N3=36, Mv=5, K_nz=20)
        assert big["i15"] == int(np.ceil(np.log2(10)))
        assert big["i16"] == 2 * int(np.ceil(np.log2(126)))  # C(9,4)

    def test_r18_elements(self):
        bits = r18_bits(F2_ANT, L=4, v=2, N3=18, Mv=5, Q=2, N4=4, K_nz=20)
        assert bits["i17"] == 2 * 2 * 4 * 5 * 2  # x Q
        assert bits["i18"] == 2 * 4  # ceil(log2(2L*Q)) = log2(16)
        assert bits["i110"] == 2 * 2  # ceil(log2(3))


class TestF2Claims:
    """Qualitative claims of the paper's Fig. f2 discussion."""

    def test_ordering_and_growth(self):
        data = f2_comparison(F2_ANT)
        r15, r16, r18 = (data[k] for k in ("R15 Regular", "R16 Regular", "R18 Regular"))
        # L >= 2 (proper Type II range): R15 > R16 > R18.  At L=1 the fixed
        # K_NZ=20 coefficient payload makes R16 cost more than R15 under
        # spec-faithful accounting -- one of the places where the paper's
        # Fig. f2 bars are not derivable from its own bit tables (README).
        for i in range(1, 4):
            assert r15[i] > r16[i] > r18[i]
        assert r18[0] < r16[0]
        for series in (r15, r16, r18):
            assert all(b > a for a, b in zip(series, series[1:]))  # grows with L

    def test_l1_inversion_r16_exceeds_r15(self):
        """S8 lock-in: at L = 1 the Table bit1/bit2 formulas make R16 cost
        MORE than R15 -- R16's fixed machinery (i16 tap-combination index,
        K_NZ coefficient payload) is not amortized by a single beam.  The
        paper's Fig. f2 bars show the opposite ordering, which is not
        derivable from its own bit tables (erratum 4 territory; see README
        errata).

        Caveat (spec audit): the inversion is a property of the paper's
        fixed K^NZ = 20 convention, which is spec-infeasible at L = 1 where
        K^NZ <= 2*K0 = 2*ceil(beta*2L*M1) = 10; at any feasible budget R16
        stays below R15."""
        data = f2_comparison(F2_ANT)
        assert data["R16 Regular"][0] > data["R15 Regular"][0]
        # spec-feasible maximum at L = 1: K0 = 5, K^NZ <= 10 -> no inversion
        feasible = 4 * sum(r16_bits(F2_ANT, L=1, v=2, N3=18, Mv=5, K_nz=10).values())
        assert feasible < data["R15 Regular"][0]  # 464 < 488

    def test_r15_gap_grows_with_subband_count(self):
        """Per-subband reporting is R15's cost driver; FD compression decouples
        R16 from N3 (up to the tap-combination index)."""
        ratios = []
        for N3 in (6, 12, 18):
            r15 = sum(r15_bits(F2_ANT, L=4, N3=N3).values())
            r16 = sum(r16_bits(F2_ANT, L=4, N3=N3).values())
            ratios.append(r15 / r16)
        assert ratios[0] < ratios[1] < ratios[2]
        assert ratios[2] > 2.5  # at N3=18 the R15 overhead clearly dominates

    def test_r18_compression_vs_repeated_r16(self):
        """One R18 report covering N4 intervals stays cheaper than N4 R16
        reports, and the advantage grows with N4."""
        for L in (2, 4):
            prev = None
            for N4 in (2, 4, 8):
                r16_total = N4 * sum(r16_bits(F2_ANT, L=L).values())
                r18_total = sum(r18_bits(F2_ANT, L=L, N4=N4).values())
                ratio = r16_total / r18_total
                assert ratio > 1
                if prev is not None:
                    assert ratio > prev
                prev = ratio


class TestF2GoldenValues:
    """Frozen per-element bit values for the paper's f2 configuration
    ((16,1), v=2, N3=18, Mv=5, K_NZ=20, N_PSK=4, K2=6, Q=2, N4=4), every
    L in {1..4}.  Locks the Table bit1/bit2 formulas against regressions."""

    R15_GOLDEN = {
        1: {"i11": 2, "i12": 4, "i13": 2, "i14": 6, "i21": 72, "i22": 36},
        2: {"i11": 2, "i12": 7, "i13": 4, "i14": 18, "i21": 216, "i22": 108},
        3: {"i11": 2, "i12": 10, "i13": 6, "i14": 30, "i21": 360, "i22": 180},
        4: {"i11": 2, "i12": 11, "i13": 6, "i14": 42, "i21": 504, "i22": 180},
    }
    R16_GOLDEN = {
        1: {"i11": 2, "i12": 4, "i16": 24, "i17": 20, "i18": 2,
            "i23": 8, "i24": 54, "i25": 72},
        2: {"i11": 2, "i12": 7, "i16": 24, "i17": 40, "i18": 4,
            "i23": 8, "i24": 54, "i25": 72},
        3: {"i11": 2, "i12": 10, "i16": 24, "i17": 60, "i18": 6,
            "i23": 8, "i24": 54, "i25": 72},
        4: {"i11": 2, "i12": 11, "i16": 24, "i17": 80, "i18": 6,
            "i23": 8, "i24": 54, "i25": 72},
    }
    R18_GOLDEN = {
        1: {"i11": 2, "i12": 4, "i16": 24, "i17": 40, "i18": 4, "i110": 4,
            "i23": 8, "i24": 54, "i25": 72},
        2: {"i11": 2, "i12": 7, "i16": 24, "i17": 80, "i18": 6, "i110": 4,
            "i23": 8, "i24": 54, "i25": 72},
        3: {"i11": 2, "i12": 10, "i16": 24, "i17": 120, "i18": 8, "i110": 4,
            "i23": 8, "i24": 54, "i25": 72},
        4: {"i11": 2, "i12": 11, "i16": 24, "i17": 160, "i18": 8, "i110": 4,
            "i23": 8, "i24": 54, "i25": 72},
    }

    @pytest.mark.parametrize("L", [1, 2, 3, 4])
    def test_r15_golden(self, L):
        assert r15_bits(F2_ANT, L=L, v=2, N3=18, n_psk=4, K2=6) == self.R15_GOLDEN[L]

    @pytest.mark.parametrize("L", [1, 2, 3, 4])
    def test_r16_golden(self, L):
        assert r16_bits(F2_ANT, L=L, v=2, N3=18, Mv=5, K_nz=20) == self.R16_GOLDEN[L]

    @pytest.mark.parametrize("L", [1, 2, 3, 4])
    def test_r18_golden(self, L):
        got = r18_bits(F2_ANT, L=L, v=2, N3=18, Mv=5, Q=2, N4=4, K_nz=20)
        assert got == self.R18_GOLDEN[L]

    def test_r15_paper_worked_example(self):
        """The L=4 column the paper walks through: i21 = 2*18*14, i22 = 2*18*5."""
        bits = self.R15_GOLDEN[4]
        assert bits["i21"] == 2 * 18 * 14
        assert bits["i22"] == 2 * 18 * 5
        assert bits["i14"] == 2 * 3 * 7


class TestNominalMatchesCodebookClasses:
    """The pure-formula module and the codebook classes must agree."""

    def test_r16_class_consistency(self):
        from nr_csi.codebooks.etype2_r16 import R16Type2Codebook

        ant = AntennaConfig.standard(4, 2)
        cbk = R16Type2Codebook(ant, N3=12, param_combination=4)
        rng = np.random.default_rng(0)
        H = rng.standard_normal((1, 12, 2, ant.P)) + 1j * rng.standard_normal((1, 12, 2, ant.P))
        pmi = cbk.select(H, rank=2)
        actual = cbk.overhead_bits(pmi)
        nominal = r16_bits(ant, L=cbk.L, v=2, N3=12, Mv=cbk.Mv(2), K_nz=int(pmi.i17.sum()))
        assert actual == nominal

    def test_r18_class_consistency(self):
        from nr_csi.codebooks.etype2_r18 import R18Type2Codebook

        ant = AntennaConfig.standard(4, 2)
        cbk = R18Type2Codebook(ant, N3=8, N4=4, param_combination=3)
        rng = np.random.default_rng(1)
        H = rng.standard_normal((4, 8, 2, ant.P)) + 1j * rng.standard_normal((4, 8, 2, ant.P))
        pmi = cbk.select(H, rank=2)
        actual = cbk.overhead_bits(pmi)
        nominal = r18_bits(
            ant, L=cbk.L, v=2, N3=8, Mv=cbk.Mv(2), Q=2, N4=4, K_nz=int(pmi.i17.sum())
        )
        assert actual == nominal
