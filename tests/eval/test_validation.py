"""Negative tests: constructor guards, channel/config mismatches, and
malformed-PMI rejection in ``precoder()``."""

import numpy as np
import pytest

from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
)
from nr_csi.config import AntennaConfig

ANT = AntennaConfig.standard(4, 2)
SMALL = AntennaConfig.standard(2, 1)  # N1*N2 = 2


def random_channel(rng, n_slots, N3, n_rx=2, P=ANT.P):
    return rng.standard_normal((n_slots, N3, n_rx, P)) + 1j * rng.standard_normal(
        (n_slots, N3, n_rx, P)
    )


class TestConstructorGuards:
    def test_r15_l_exceeds_group_size(self):
        with pytest.raises(ValueError, match="orthogonal group"):
            R15Type2Codebook(SMALL, N3=4, L=4)

    def test_r16_l_exceeds_group_size(self):
        # at P = 4 the explicit 5.2.2.2.5 configuration bar (combinations 3-8)
        # fires before the generic L > N1*N2 check
        with pytest.raises(ValueError, match="P_CSI-RS=4"):
            R16Type2Codebook(SMALL, N3=4, param_combination=3)  # L = 4

    def test_r16_ps_l_exceeds_ports_per_pol(self):
        """L=4 port-selection vectors alias onto SMALL's 2 ports/polarization
        (v_m's "m mod P_CSI-RS/2" wraparound, Table 5.2.2.2.6-2): two of the four
        beam indices land on the same physical port and get identical LS-fitted
        coefficients, so gamma_{t,l} (which sums |coefficient|^2 per beam index as
        if they were orthogonal) undercounts the coherent sum at that port and the
        precoder comes out at exactly 2x its intended power. The base Type II PS
        codebook (38.214 5.2.2.2.4) avoids this by pinning L=2 when P_CSI-RS=4;
        the eType II PS table has no such bar, so the constructor must enforce it."""
        with pytest.raises(ValueError, match="ports per polarization"):
            R16Type2Codebook(SMALL, N3=4, param_combination=3, port_selection=True, d=2)

    def test_r18_l_exceeds_group_size(self):
        with pytest.raises(ValueError, match="orthogonal group"):
            R18Type2Codebook(SMALL, N3=4, N4=2, param_combination=3)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: R16Type2Codebook(ANT, N3=0, param_combination=1),
            lambda: R17Type2Codebook(ANT, N3=0, param_combination=1),
            lambda: R18Type2Codebook(ANT, N3=0, N4=2, param_combination=1),
        ],
    )
    def test_nonpositive_n3(self, make):
        with pytest.raises(ValueError):
            make()

    def test_r15_bad_l_and_psk(self):
        with pytest.raises(ValueError):
            R15Type2Codebook(ANT, N3=4, L=5)
        with pytest.raises(ValueError):
            R15Type2Codebook(ANT, N3=4, L=4, n_psk=16)

    def test_ps_sampling_size(self):
        with pytest.raises(ValueError):
            R15Type2Codebook(ANT, N3=4, L=2, port_selection=True, d=3)  # d > L

    def test_r17_window(self):
        with pytest.raises(ValueError):
            R17Type2Codebook(ANT, N3=8, param_combination=7, N_window=3)

    def test_r18_n4(self):
        with pytest.raises(ValueError):
            R18Type2Codebook(ANT, N3=8, N4=3, param_combination=3)

    def test_type1_mode2_n2_1_is_supported(self):
        Type1Codebook(AntennaConfig.standard(8, 1), N3=4, mode=2)


class TestChannelMismatch:
    @pytest.mark.parametrize(
        "cbk",
        [
            Type1Codebook(ANT, N3=4),
            R15Type2Codebook(ANT, N3=4, L=2),
            R16Type2Codebook(ANT, N3=4, param_combination=1),
            R17Type2Codebook(ANT, N3=4, param_combination=5),
        ],
        ids=lambda c: c.name,
    )
    def test_wrong_n3_rejected(self, cbk):
        H = random_channel(np.random.default_rng(0), 1, 6)
        with pytest.raises(ValueError, match="frequency units"):
            cbk.select(H, rank=1)

    def test_r18_wrong_n3_and_n4(self):
        cbk = R18Type2Codebook(ANT, N3=4, N4=2, param_combination=1)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="slot intervals"):
            cbk.select(random_channel(rng, 1, 4), rank=1)
        with pytest.raises(ValueError, match="frequency units"):
            cbk.select(random_channel(rng, 2, 6), rank=1)

    def test_rank_exceeds_channel_rank(self):
        cbk = R16Type2Codebook(ANT, N3=4, param_combination=1)
        H = random_channel(np.random.default_rng(0), 1, 4, n_rx=1)
        with pytest.raises(ValueError, match="exceeds channel rank"):
            cbk.select(H, rank=2)

    def test_unsupported_rank(self):
        rng = np.random.default_rng(0)
        H = random_channel(rng, 1, 4, n_rx=4)
        with pytest.raises(ValueError):
            Type1Codebook(ANT, N3=4).select(H, rank=9)
        with pytest.raises(ValueError):
            R15Type2Codebook(ANT, N3=4, L=2).select(H, rank=3)
        with pytest.raises(ValueError):
            R16Type2Codebook(ANT, N3=4, param_combination=1).select(H, rank=5)


def valid_pmi(cbk, rank=1, n_slots=1, seed=0):
    rng = np.random.default_rng(seed)
    H = random_channel(rng, n_slots, cbk.N3)
    return cbk.select(H, rank=rank)


class TestMalformedPMIRejection:
    def test_type1_out_of_range_fields(self):
        cbk = Type1Codebook(ANT, N3=4)
        pmi = valid_pmi(cbk)
        G1, G2 = ANT.n_beams
        for field, bad in (("i11", G1), ("i11", -1), ("i12", G2)):
            good = getattr(pmi, field)
            setattr(pmi, field, bad)
            with pytest.raises(ValueError, match="malformed PMI"):
                cbk.precoder(pmi)
            setattr(pmi, field, good)
        pmi.i2 = pmi.i2 + 4  # beyond n_i2 - 1
        with pytest.raises(ValueError, match="malformed PMI"):
            cbk.precoder(pmi)

    def test_type1_wrong_i2_shape(self):
        cbk = Type1Codebook(ANT, N3=4)
        pmi = valid_pmi(cbk)
        pmi.i2 = pmi.i2[:2]
        with pytest.raises(ValueError, match="malformed PMI"):
            cbk.precoder(pmi)

    def test_r15_violations(self):
        cbk = R15Type2Codebook(ANT, N3=4, L=2)
        pmi = valid_pmi(cbk)
        i_star = pmi.i13[0]
        pmi.k1[0, i_star] = 5  # strongest coefficient must keep k1 = 7
        with pytest.raises(ValueError, match="k1=7"):
            cbk.precoder(pmi)
        pmi.k1[0, i_star] = 7
        pmi.k1 = pmi.k1[:, :3]  # wrong shape
        with pytest.raises(ValueError, match="shape"):
            cbk.precoder(pmi)

    def test_r15_out_of_range_indices(self):
        cbk = R15Type2Codebook(ANT, N3=4, L=2)
        pmi = valid_pmi(cbk)
        pmi.q1 = ANT.O1
        with pytest.raises(ValueError, match="q1"):
            cbk.precoder(pmi)
        pmi.q1 = 0
        from math import comb

        pmi.i12 = comb(ANT.N1 * ANT.N2, 2)
        with pytest.raises(ValueError, match="i12"):
            cbk.precoder(pmi)

    def test_r16_violations(self):
        cbk = R16Type2Codebook(ANT, N3=8, param_combination=2)
        pmi = valid_pmi(cbk)
        from nr_csi.codebooks.etype2_r16 import decode_i18

        i_star = decode_i18(pmi.i18[0], pmi.i17[0], pmi.rank)
        pmi.i17[0, 0, i_star] = False  # strongest bit cleared
        with pytest.raises(ValueError, match="malformed PMI"):
            cbk.precoder(pmi)
        pmi.i17[0, 0, i_star] = True
        pmi.k1[0, 0] = 0  # reserved reference amplitude
        with pytest.raises(ValueError, match="k1"):
            cbk.precoder(pmi)

    def test_r16_i15_consistency(self):
        small = R16Type2Codebook(ANT, N3=8, param_combination=2)
        pmi = valid_pmi(small)
        pmi.i15 = 0  # must not be reported for N3 <= 19
        with pytest.raises(ValueError, match="i15"):
            small.precoder(pmi)
        big = R16Type2Codebook(ANT, N3=24, param_combination=2)
        pmi = valid_pmi(big)
        pmi.i15 = None  # required for N3 > 19
        with pytest.raises(ValueError, match="i15"):
            big.precoder(pmi)

    def test_r17_violations(self):
        cbk = R17Type2Codebook(ANT, N3=8, param_combination=2)  # M=1, alpha=1
        pmi = valid_pmi(cbk)
        pmi.i16 = 1  # i16 must not be reported when M = 1
        with pytest.raises(ValueError, match="i16"):
            cbk.precoder(pmi)
        pmi.i16 = None
        pmi.i12 = 0  # i12 must not be reported when alpha = 1
        with pytest.raises(ValueError, match="i12"):
            cbk.precoder(pmi)

    def test_r17_phase_out_of_range(self):
        cbk = R17Type2Codebook(ANT, N3=8, param_combination=5)
        pmi = valid_pmi(cbk)
        pmi.c[0, 0, 0] = 16
        with pytest.raises(ValueError, match="c values"):
            cbk.precoder(pmi)

    def test_r18_violations(self):
        cbk = R18Type2Codebook(ANT, N3=8, N4=4, param_combination=3)
        pmi = valid_pmi(cbk, n_slots=4)
        pmi.i110[0] = 3  # offset beyond N4 - 2
        with pytest.raises(ValueError, match="i110"):
            cbk.precoder(pmi)
        pmi.i110[0] = 0
        pmi.k2 = pmi.k2[:, :, :1]  # wrong shape
        with pytest.raises(ValueError, match="shape"):
            cbk.precoder(pmi)

    def test_valid_pmi_passes(self):
        """Sanity: an unmodified selected PMI reconstructs fine."""
        for cbk, n_slots in (
            (Type1Codebook(ANT, N3=4), 1),
            (R15Type2Codebook(ANT, N3=4, L=2), 1),
            (R16Type2Codebook(ANT, N3=8, param_combination=2), 1),
            (R17Type2Codebook(ANT, N3=8, param_combination=5), 1),
            (R18Type2Codebook(ANT, N3=8, N4=2, param_combination=1), 2),
        ):
            W = cbk.precoder(valid_pmi(cbk, n_slots=n_slots))
            assert np.all(np.isfinite(W))
