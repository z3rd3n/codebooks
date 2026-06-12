"""PMI composition conformance: which information elements exist per
(codebook, configuration).

The paper enumerates the exact i1/i2 element sets: R15 Type I/II, R16
eq. (a85), R16-PS eq. (a86) (no i12, explicit i11), R17 eq. (a104)
(no i11/i15, unified i16), R18 eq. (a127) (adds i110).  ``overhead_bits``
keys must match these sets exactly for every configuration, including the
conditional drops (R17 alpha=1 drops i12, M=1 drops i16; R18 N4=1 drops
i110; R16 N3<=19 drops i15; R15 SA=false drops i22).
"""

import numpy as np

from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
)
from nr_csi.config import AntennaConfig

ANT = AntennaConfig.standard(4, 2)


def random_channel(rng, n_slots, N3, n_rx=2, P=ANT.P):
    return rng.standard_normal((n_slots, N3, n_rx, P)) + 1j * rng.standard_normal(
        (n_slots, N3, n_rx, P)
    )


def keys_of(scheme, rank=1, n_slots=1, N3=None, seed=0):
    rng = np.random.default_rng(seed)
    H = random_channel(rng, n_slots, N3 if N3 is not None else scheme.N3)
    pmi = scheme.select(H, rank=rank)
    return set(scheme.overhead_bits(pmi).keys())


class TestType1Composition:
    def test_rank1(self):
        assert keys_of(Type1Codebook(ANT, N3=4), rank=1) == {"i11", "i12", "i2"}

    def test_rank2_adds_i13(self):
        assert keys_of(Type1Codebook(ANT, N3=4), rank=2) == {"i11", "i12", "i13", "i2"}


class TestR15Type2Composition:
    def test_regular_no_subband_amplitude(self):
        cbk = R15Type2Codebook(ANT, N3=4, L=2, subband_amplitude=False)
        assert keys_of(cbk) == {"i11", "i12", "i13", "i14", "i21"}

    def test_regular_with_subband_amplitude(self):
        cbk = R15Type2Codebook(ANT, N3=4, L=2, subband_amplitude=True)
        assert keys_of(cbk) == {"i11", "i12", "i13", "i14", "i21", "i22"}

    def test_port_selection_drops_i12(self):
        """The PS variant reports the initial port as i11 and has no beam
        combination index (paper Appendix r15ps)."""
        cbk = R15Type2Codebook(ANT, N3=4, L=2, port_selection=True, d=2)
        assert keys_of(cbk) == {"i11", "i13", "i14", "i21"}


class TestR16Composition:
    I2 = {"i23", "i24", "i25"}

    def test_regular_small_n3(self):
        """Eq. (a85) with the N3 <= 19 single-level tap indication."""
        cbk = R16Type2Codebook(ANT, N3=8, param_combination=2)
        assert keys_of(cbk) == {"i11", "i12", "i16", "i17", "i18"} | self.I2

    def test_regular_large_n3_adds_i15(self):
        cbk = R16Type2Codebook(ANT, N3=24, param_combination=2)
        assert keys_of(cbk) == {"i11", "i12", "i15", "i16", "i17", "i18"} | self.I2

    def test_port_selection_drops_i12(self):
        """Eq. (a86): no i12, explicit (port-window) i11."""
        cbk = R16Type2Codebook(ANT, N3=8, param_combination=2, port_selection=True, d=2)
        assert keys_of(cbk) == {"i11", "i16", "i17", "i18"} | self.I2


class TestR17Composition:
    I2 = {"i23", "i24", "i25"}

    def test_full_composition(self):
        """Eq. (a104): alpha < 1 and M = 2 with window N = 4 -> i12 and i16."""
        cbk = R17Type2Codebook(ANT, N3=8, param_combination=6, N_window=4)  # M=2, a=3/4
        assert keys_of(cbk) == {"i12", "i16", "i17", "i18"} | self.I2

    def test_alpha_1_drops_i12(self):
        cbk = R17Type2Codebook(ANT, N3=8, param_combination=7, N_window=4)  # M=2, a=1
        assert keys_of(cbk) == {"i16", "i17", "i18"} | self.I2

    def test_m1_drops_i16(self):
        cbk = R17Type2Codebook(ANT, N3=8, param_combination=2)  # M=1, a=1
        assert keys_of(cbk) == {"i17", "i18"} | self.I2

    def test_m2_window2_drops_i16(self):
        """M = 2 with N = 2: the second tap is implied, i16 not reported."""
        cbk = R17Type2Codebook(ANT, N3=8, param_combination=7, N_window=2)
        assert keys_of(cbk) == {"i17", "i18"} | self.I2

    def test_never_reports_i11_or_i15(self):
        """R17 has no consecutive-port window start and no two-level taps."""
        for combo, window in ((1, 4), (5, 4), (8, 2)):
            cbk = R17Type2Codebook(ANT, N3=8, param_combination=combo, N_window=window)
            assert not keys_of(cbk) & {"i11", "i15"}


class TestR18Composition:
    I2 = {"i23", "i24", "i25"}

    def test_n4_gt_1_adds_i110(self):
        """Eq. (a127) = (a85) plus the Doppler-shift offsets i110."""
        cbk = R18Type2Codebook(ANT, N3=8, N4=4, param_combination=3)
        keys = keys_of(cbk, n_slots=4)
        assert keys == {"i11", "i12", "i16", "i17", "i18", "i110"} | self.I2

    def test_n4_1_drops_i110(self):
        cbk = R18Type2Codebook(ANT, N3=8, N4=1, param_combination=3)
        assert keys_of(cbk, n_slots=1) == {"i11", "i12", "i16", "i17", "i18"} | self.I2

    def test_large_n3_adds_i15(self):
        cbk = R18Type2Codebook(ANT, N3=24, N4=2, param_combination=3)
        keys = keys_of(cbk, n_slots=2)
        assert keys == {"i11", "i12", "i15", "i16", "i17", "i18", "i110"} | self.I2

    def test_n4_2_i110_costs_zero_bits(self):
        """N4 = 2 leaves a single candidate second shift: i110 present, 0 bits."""
        cbk = R18Type2Codebook(ANT, N3=8, N4=2, param_combination=3)
        rng = np.random.default_rng(0)
        H = random_channel(rng, 2, 8)
        pmi = cbk.select(H, rank=1)
        assert cbk.overhead_bits(pmi)["i110"] == 0
