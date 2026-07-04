"""RRC configuration guards added for spec-exactness, TS 38.214 v19.4.0.

Covers:

* RI-restriction bitmaps on the base classes: typeII-RI-Restriction /
  typeII-PortSelectionRI-Restriction (R15, 2-bit), typeII-RI-Restriction-r16 /
  typeII-PortSelectionRI-Restriction-r16 (4-bit),
  typeII-PortSelectionRI-Restriction-r17 (4-bit);
* the paramCombination configuration bars of 5.2.2.2.5 / 5.2.2.2.7 /
  5.2.2.2.10;
* the R16 average-amplitude codebook subset restriction of Table 5.2.2.2.5-6
  (n1-n2-codebookSubsetRestriction-r16).
"""

from __future__ import annotations

import numpy as np
import pytest

from nr_csi.codebooks import (
    R15Type2Codebook,
    R16AmplitudeRestriction,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
)
from nr_csi.config import R16_PARAM_COMBOS, AntennaConfig
from nr_csi.utils import quantization as qt

ANT16 = AntennaConfig.standard(4, 2)
ANT32 = AntennaConfig.standard(4, 4)
ANT4 = AntennaConfig.standard(2, 1)


def _chan(rng, N3, nr, P, n_slots=1):
    z = rng.standard_normal((n_slots, N3, nr, P)) + 1j * rng.standard_normal(
        (n_slots, N3, nr, P)
    )
    return z / np.sqrt(2)


# ---------------------------------------------------------------------------
# RI-restriction bitmaps
# ---------------------------------------------------------------------------


class TestRIRestrictions:
    def test_r15_regular(self):
        cbk = R15Type2Codebook(ANT16, N3=4, L=2, ri_restriction=[1, 0])
        H = _chan(np.random.default_rng(0), 4, 2, ANT16.P)
        with pytest.raises(ValueError, match="typeII-RI-Restriction"):
            cbk.select(H, rank=2)
        cbk.select(H, rank=1)  # rank 1 still allowed

    def test_r15_port_selection_name(self):
        cbk = R15Type2Codebook(
            ANT16, N3=4, L=2, port_selection=True, d=2, ri_restriction=[0, 1]
        )
        H = _chan(np.random.default_rng(1), 4, 2, ANT16.P)
        with pytest.raises(ValueError, match="typeII-PortSelectionRI-Restriction"):
            cbk.select(H, rank=1)

    def test_r15_bitmap_width(self):
        with pytest.raises(ValueError, match="2 bits"):
            R15Type2Codebook(ANT16, N3=4, L=2, ri_restriction=[1, 1, 1, 1])

    def test_r16(self):
        cbk = R16Type2Codebook(
            ANT16, N3=4, param_combination=2, ri_restriction=[0, 1, 1, 1]
        )
        H = _chan(np.random.default_rng(2), 4, 2, ANT16.P)
        with pytest.raises(ValueError, match="typeII-RI-Restriction-r16"):
            cbk.select(H, rank=1)
        cbk.select(H, rank=2)

    def test_r16_port_selection_name(self):
        cbk = R16Type2Codebook(
            ANT16, N3=4, param_combination=2, port_selection=True,
            ri_restriction=[0, 1, 1, 1],
        )
        H = _chan(np.random.default_rng(3), 4, 2, ANT16.P)
        with pytest.raises(
            ValueError, match="typeII-PortSelectionRI-Restriction-r16"
        ):
            cbk.select(H, rank=1)

    def test_r17(self):
        cbk = R17Type2Codebook(
            ANT16, N3=4, param_combination=2, ri_restriction=[1, 1, 1, 0]
        )
        H = _chan(np.random.default_rng(4), 4, 4, ANT16.P)
        with pytest.raises(
            ValueError, match="typeII-PortSelectionRI-Restriction-r17"
        ):
            cbk.select(H, rank=4)

    def test_bitmap_width_r16_r17(self):
        with pytest.raises(ValueError, match="4 bits"):
            R16Type2Codebook(ANT16, N3=4, param_combination=2, ri_restriction=[1, 1])
        with pytest.raises(ValueError, match="4 bits"):
            R17Type2Codebook(ANT16, N3=4, param_combination=2, ri_restriction=[1, 1])


# ---------------------------------------------------------------------------
# paramCombination bars
# ---------------------------------------------------------------------------


class TestR16Bars:
    @pytest.mark.parametrize("pc", [3, 4, 5, 6, 7, 8])
    def test_p4_bars_3_to_8(self, pc):
        with pytest.raises(ValueError, match="P_CSI-RS=4"):
            R16Type2Codebook(ANT4, N3=4, param_combination=pc)

    @pytest.mark.parametrize("shape", [(6, 1), (4, 2), (4, 3)])  # P = 12, 16, 24
    @pytest.mark.parametrize("pc", [7, 8])
    def test_L6_needs_32_ports(self, shape, pc):
        ant = AntennaConfig.standard(*shape)
        with pytest.raises(ValueError, match="P_CSI-RS >= 32"):
            R16Type2Codebook(
                ant, N3=4, param_combination=pc, ri_restriction=[1, 1, 0, 0]
            )

    @pytest.mark.parametrize("pc", [7, 8])
    def test_L6_needs_rank12_and_R1(self, pc):
        with pytest.raises(ValueError, match="ranks 3-4"):
            R16Type2Codebook(ANT32, N3=4, param_combination=pc)
        with pytest.raises(ValueError, match="R=2"):
            R16Type2Codebook(
                ANT32, N3=4, param_combination=pc, R=2,
                ri_restriction=[1, 1, 0, 0],
            )
        R16Type2Codebook(
            ANT32, N3=4, param_combination=pc, ri_restriction=[1, 1, 0, 0]
        )

    def test_custom_combo_bypasses_bars(self):
        """The ``combo`` escape hatch is for generalized sweeps and is not
        subject to the standardized-table bars."""
        R16Type2Codebook(ANT16, N3=4, combo=R16_PARAM_COMBOS[7])


class TestR17Bars:
    @pytest.mark.parametrize("shape", [(2, 1), (6, 1)])  # P = 4, 12
    @pytest.mark.parametrize("pc", [1, 6])
    def test_pc1_6_barred_at_4_and_12_ports(self, shape, pc):
        ant = AntennaConfig.standard(*shape)
        with pytest.raises(ValueError, match="bars combinations 1 and 6"):
            R17Type2Codebook(ant, N3=4, param_combination=pc)

    @pytest.mark.parametrize("pc", [7, 8])
    def test_pc7_8_barred_at_32_ports(self, pc):
        with pytest.raises(ValueError, match="P_CSI-RS=32"):
            R17Type2Codebook(ANT32, N3=4, param_combination=pc)
        R17Type2Codebook(ANT16, N3=4, param_combination=pc)  # fine at 16

    def test_pc5_p4_conditional_on_ranks(self):
        with pytest.raises(ValueError, match="ranks 3-4"):
            R17Type2Codebook(ANT4, N3=4, param_combination=5)
        R17Type2Codebook(
            ANT4, N3=4, param_combination=5, ri_restriction=[1, 1, 0, 0]
        )


class TestR18Bars:
    @pytest.mark.parametrize("pc", [4, 5, 6, 7, 8, 9])
    def test_p4_bars_4_to_9(self, pc):
        with pytest.raises(ValueError, match="P_CSI-RS=4"):
            R18Type2Codebook(ANT4, N3=4, N4=2, param_combination=pc)

    @pytest.mark.parametrize("pc", [8, 9])
    def test_L6_needs_32_ports_rank12_R1(self, pc):
        with pytest.raises(ValueError, match="P_CSI-RS >= 32"):
            R18Type2Codebook(
                ANT16, N3=4, N4=2, param_combination=pc,
                ri_restriction=[1, 1, 0, 0],
            )
        with pytest.raises(ValueError, match="ranks 3-4"):
            R18Type2Codebook(ANT32, N3=4, N4=2, param_combination=pc)
        with pytest.raises(ValueError, match="R=2"):
            R18Type2Codebook(
                ANT32, N3=4, N4=2, param_combination=pc, R=2,
                ri_restriction=[1, 1, 0, 0],
            )
        R18Type2Codebook(
            ANT32, N3=4, N4=2, param_combination=pc, ri_restriction=[1, 1, 0, 0]
        )

    def test_p4_pc3_still_caught_by_L_check(self):
        # pc 3 (L=4) is not in the explicit P=4 bar list; the generic
        # L > N1*N2 check still rejects it
        with pytest.raises(ValueError, match="orthogonal group"):
            R18Type2Codebook(ANT4, N3=4, N4=2, param_combination=3)


# ---------------------------------------------------------------------------
# R16 average-amplitude subset restriction (Table 5.2.2.2.5-6)
# ---------------------------------------------------------------------------


def _amplitude_violations(cbk, pmi, caps):
    """Restricted beams whose average reported amplitude exceeds gamma."""
    out = []
    for li in range(pmi.rank):
        p1 = qt.R16_REF_AMP[pmi.k1[li]]
        for i in range(2 * cbk.L):
            k3 = pmi.i17[li][:, i]
            if not k3.any():
                continue
            amps = p1[i // cbk.L] * qt.R16_DIFF_AMP[pmi.k2[li][:, i]]
            avg = np.sqrt(np.mean(amps[k3] ** 2))
            if avg > caps[i] + 1e-9:
                out.append((li, i, avg, caps[i]))
    return out


class TestR16AmplitudeRestriction:
    def _restriction(self, gamma_codepoints):
        b2 = np.full((4, ANT16.N1 * ANT16.N2), gamma_codepoints)
        return R16AmplitudeRestriction(beta1=0, b2=b2)

    def test_gamma_table(self):
        assert np.allclose(
            R16AmplitudeRestriction.GAMMA, [0, 0.5, np.sqrt(0.5), 1.0]
        )

    def test_b2_shape_validated(self):
        with pytest.raises(ValueError, match="B2"):
            R16Type2Codebook(
                ANT16, N3=4, param_combination=4,
                restriction=R16AmplitudeRestriction(beta1=0, b2=np.ones((4, 3))),
            )

    def test_ps_variant_rejects_restriction(self):
        with pytest.raises(ValueError, match="regular codebook only"):
            R16Type2Codebook(
                ANT16, N3=4, param_combination=4, port_selection=True,
                restriction=self._restriction(3),
            )

    def test_prohibited_groups_avoided(self):
        restr = self._restriction(0)  # gamma = 0: 4 groups fully prohibited
        cbk = R16Type2Codebook(ANT16, N3=8, param_combination=4, restriction=restr)
        g = restr.restricted_groups(ANT16.O1, ANT16.O2)
        for seed in range(4):
            H = _chan(np.random.default_rng(seed), 8, 2, ANT16.P)
            pmi = cbk.select(H, rank=1)
            assert ANT16.O1 * pmi.q2 + pmi.q1 not in g

    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_average_amplitude_cap_enforced(self, rank):
        b2 = np.full((4, ANT16.N1 * ANT16.N2), 1)  # gamma = 1/2 ...
        b2[:, ::2] = 3  # ... except even beams unrestricted
        restr = R16AmplitudeRestriction(beta1=0, b2=b2)
        cbk = R16Type2Codebook(ANT16, N3=8, param_combination=4, restriction=restr)
        for seed in range(3):
            H = _chan(np.random.default_rng(10 + seed), 8, 4, ANT16.P)
            pmi = cbk.select(H, rank=rank)
            caps = cbk._selected_gamma_caps(pmi)
            assert _amplitude_violations(cbk, pmi, caps) == []
            cbk.precoder(pmi)  # report must stay well-formed

    def test_unrestricted_group_unaffected(self):
        restr = self._restriction(1)
        cbk = R16Type2Codebook(ANT16, N3=8, param_combination=4, restriction=restr)
        free = R16Type2Codebook(ANT16, N3=8, param_combination=4)
        H = _chan(np.random.default_rng(20), 8, 2, ANT16.P)
        pmi_r, pmi_f = cbk.select(H, rank=1), free.select(H, rank=1)
        g = restr.restricted_groups(ANT16.O1, ANT16.O2)
        if ANT16.O1 * pmi_f.q2 + pmi_f.q1 not in g:
            # the free choice was already outside the restricted groups:
            # the restricted codebook must report the identical PMI
            assert (pmi_r.q1, pmi_r.q2, pmi_r.i12) == (pmi_f.q1, pmi_f.q2, pmi_f.i12)
            assert np.array_equal(pmi_r.k2, pmi_f.k2)

    def test_strongest_coefficient_prefers_unrestricted_beam(self):
        b2 = np.full((4, ANT16.N1 * ANT16.N2), 1)
        b2[:, 0] = 3  # exactly one unrestricted beam per group
        restr = R16AmplitudeRestriction(beta1=0, b2=b2)
        cbk = R16Type2Codebook(ANT16, N3=8, param_combination=4, restriction=restr)
        from nr_csi.utils import combinatorics as cb

        for seed in range(4):
            H = _chan(np.random.default_rng(30 + seed), 8, 2, ANT16.P)
            pmi = cbk.select(H, rank=1)
            caps = cbk._selected_gamma_caps(pmi)
            if ANT16.O1 * pmi.q2 + pmi.q1 not in restr.restricted_groups(4, 4):
                continue
            from nr_csi.codebooks.etype2_r16 import decode_i18

            i_star = decode_i18(pmi.i18[0], pmi.i17[0], 1)
            if (caps == 1.0).any():
                assert caps[i_star] == 1.0
