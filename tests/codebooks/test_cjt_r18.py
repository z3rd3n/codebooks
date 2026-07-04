"""R18 CJT codebooks, TS 38.214 5.2.2.2.8 (typeII-CJT-r18) and 5.2.2.2.9
(typeII-CJT-PortSelection-r18).

Covers:

* the configuration tables 5.2.2.2.8-1/-2/-3 and 5.2.2.2.9-1/-2/-3 and their
  "not expected to be configured" bars;
* the N_TRP = 1 degeneracy: CJT mode2 reproduces the R16 eType II / R17
  feType II PS reports and precoders bit-for-bit;
* the mode1 inter-TRP delay offsets i_{1,9} (recovery of a synthetic
  inter-TRP delay, O3 ramp reconstruction, mode1 >= mode2);
* K0 coefficient budgets, strongest-coefficient conventions, resource
  selection, and overhead accounting.
"""

from __future__ import annotations

import numpy as np
import pytest

from nr_csi.codebooks.cjt_r18 import (
    CJT_ALLOWED,
    CJT_L_COMBOS,
    CJT_PS_ALLOWED,
    CJT_PS_ALPHA_COMBOS,
    R18CJTCodebook,
    R18CJTPortSelectionCodebook,
)
from nr_csi.codebooks.etype2_r16 import R16Type2Codebook
from nr_csi.codebooks.fetype2_r17 import R17Type2Codebook
from nr_csi.config import AntennaConfig
from nr_csi.metrics.similarity import sgcs

ANT = AntennaConfig.standard(4, 2)  # P = 16 per resource


def _chan(rng, N3, nr, P):
    z = rng.standard_normal((1, N3, nr, P)) + 1j * rng.standard_normal((1, N3, nr, P))
    return z / np.sqrt(2)


def _delayed_two_trp_channel(rng, N3, nr, P, delay: int, sparse: bool = False):
    """Resource 2 sees the same propagation as resource 1 shifted by an
    integer inter-TRP delay (a pure frequency ramp).  ``sparse`` draws a
    tap-sparse base channel (compressible by the M_v delay taps) instead of
    an i.i.d. one."""
    if sparse:
        from nr_csi.channel import RandomRayChannel

        chan = RandomRayChannel(ANT, N3=N3, n_rx=nr, n_paths=3, max_delay=0.0)
        base = chan.generate(n_slots=1, rng=rng)
    else:
        base = _chan(rng, N3, nr, P)
    ramp = np.exp(2j * np.pi * np.arange(N3) * delay / N3)
    return np.concatenate([base, base * ramp[None, :, None, None]], axis=-1)


# ---------------------------------------------------------------------------
# configuration guards
# ---------------------------------------------------------------------------


class TestCJTConfigGuards:
    def test_table_8_3_pairs_enforced(self):
        # N_TRP=2, pcL=1 only allows paramCombination-CJT-r18 = 1
        R18CJTCodebook(ANT, N3=8, n_trp=2, param_combination_L=1, param_combination=1)
        with pytest.raises(ValueError, match="Table 5.2.2.2.8-3"):
            R18CJTCodebook(
                ANT, N3=8, n_trp=2, param_combination_L=1, param_combination=2
            )

    def test_p4_bars(self):
        small = AntennaConfig.standard(2, 1)  # P = 4
        with pytest.raises(ValueError, match="P_CSI-RS=4"):
            R18CJTCodebook(
                small, N3=8, n_trp=2, param_combination_L=2, param_combination=1
            )

    def test_ntrp1_L6_bars(self):
        # pcL=3 (L=6) needs P >= 32
        with pytest.raises(ValueError, match="P_CSI-RS >= 32"):
            R18CJTCodebook(
                ANT, N3=8, n_trp=1, param_combination_L=3, param_combination=4
            )
        ant32 = AntennaConfig.standard(4, 4)
        # ... and ranks 3-4 disallowed
        with pytest.raises(ValueError, match="ranks 3-4"):
            R18CJTCodebook(
                ant32, N3=8, n_trp=1, param_combination_L=3, param_combination=4
            )
        R18CJTCodebook(
            ant32, N3=8, n_trp=1, param_combination_L=3, param_combination=4,
            ri_restriction=[1, 1, 0, 0],
        )

    def test_nl_multiple_combinations(self):
        cbk = R18CJTCodebook(
            ANT, N3=8, n_trp=3, param_combination_L=[1, 2], param_combination=1
        )
        assert cbk.N_L == 2
        with pytest.raises(ValueError, match="1, 2 or 4"):
            R18CJTCodebook(
                ANT, N3=8, n_trp=3, param_combination_L=[1, 2, 3], param_combination=1
            )

    def test_ps_alpha_bars(self):
        small = AntennaConfig.standard(2, 1)  # P = 4: alpha = 3/4 barred
        with pytest.raises(ValueError, match="P_CSI-RS=4"):
            R18CJTPortSelectionCodebook(
                small, N3=8, n_trp=1, param_combination_alpha=2, param_combination=1
            )
        ant32 = AntennaConfig.standard(4, 4)
        with pytest.raises(ValueError, match="P_CSI-RS=32"):
            R18CJTPortSelectionCodebook(
                ant32, N3=8, n_trp=1, param_combination_alpha=3, param_combination=4
            )

    def test_ps_table_9_3_pairs_enforced(self):
        with pytest.raises(ValueError, match="Table 5.2.2.2.9-3"):
            R18CJTPortSelectionCodebook(
                ANT, N3=8, n_trp=2, param_combination_alpha=2, param_combination=4
            )

    def test_restricted_cmr_fixes_n0(self):
        with pytest.raises(ValueError, match="N0 = N_TRP"):
            R18CJTCodebook(
                ANT, N3=8, n_trp=2, param_combination_L=1, param_combination=1,
                restricted_cmr=True, n0=1,
            )


# ---------------------------------------------------------------------------
# N_TRP = 1 degeneracy
# ---------------------------------------------------------------------------


class TestSingleTRPDegeneracy:
    @pytest.mark.parametrize("rank", [1, 2, 3, 4])
    def test_cjt_equals_r16_for_one_trp(self, rank):
        """N_TRP=1 + mode2 + (pcL=2 -> L=4, pc=4 -> p_v=(1/4,1/8), beta=1/2)
        is exactly R16 eType II paramCombination 4."""
        rng = np.random.default_rng(40 + rank)
        H = _chan(rng, 8, 4, ANT.P)
        cjt = R18CJTCodebook(
            ANT, N3=8, n_trp=1, param_combination_L=2, param_combination=4,
            mode="mode2",
        )
        r16 = R16Type2Codebook(ANT, N3=8, param_combination=4)
        p_cjt = cjt.select(H, rank=rank)
        p_r16 = r16.select(H, rank=rank)
        assert (p_cjt.q1[0], p_cjt.q2[0], p_cjt.i12[0]) == (
            p_r16.q1, p_r16.q2, p_r16.i12
        )
        assert p_cjt.i16 == p_r16.i16 and p_cjt.i18 == p_r16.i18
        assert np.array_equal(p_cjt.i17, p_r16.i17)
        assert np.array_equal(p_cjt.k1, p_r16.k1)
        assert np.array_equal(p_cjt.k2, p_r16.k2)
        assert np.array_equal(p_cjt.c, p_r16.c)
        assert np.allclose(cjt.precoder(p_cjt), r16.precoder(p_r16))

    @pytest.mark.parametrize("rank", [1, 2])
    def test_cjt_ps_equals_r17_for_one_trp(self, rank):
        """N_TRP=1 + mode2 + (alpha=1, M=1, beta=1/2) is exactly R17 feType II
        PS paramCombination 2."""
        rng = np.random.default_rng(50 + rank)
        H = _chan(rng, 8, 4, ANT.P)
        cjt = R18CJTPortSelectionCodebook(
            ANT, N3=8, n_trp=1, param_combination_alpha=3, param_combination=1,
            mode="mode2",
        )
        r17 = R17Type2Codebook(ANT, N3=8, param_combination=2)
        p_cjt = cjt.select(H, rank=rank)
        p_r17 = r17.select(H, rank=rank)
        assert p_cjt.i12 == (None,) and p_r17.i12 is None
        assert p_cjt.i18 == p_r17.i18
        assert np.array_equal(p_cjt.i17, p_r17.i17)
        assert np.array_equal(p_cjt.k1, p_r17.k1)
        assert np.array_equal(p_cjt.k2, p_r17.k2)
        assert np.array_equal(p_cjt.c, p_r17.c)
        assert np.allclose(cjt.precoder(p_cjt), r17.precoder(p_r17))


# ---------------------------------------------------------------------------
# inter-TRP delay offsets (mode1)
# ---------------------------------------------------------------------------


class TestInterTRPOffsets:
    N3 = 12

    def test_offset_recovered_on_synthetic_delay(self):
        """The target precoder *cancels* the channel's inter-TRP ramp
        (conjugate convention), so a channel delay +delta on resource 2 is
        pre-compensated with psi_2 = (N3 - delta) mod N3."""
        rng = np.random.default_rng(7)
        delay = 3
        H = _delayed_two_trp_channel(rng, self.N3, 2, ANT.P, delay, sparse=True)
        cbk = R18CJTCodebook(
            ANT, N3=self.N3, n_trp=2, param_combination_L=1, param_combination=1,
            mode="mode1", O3=4,
        )
        pmi = cbk.select(H, rank=1)
        assert pmi.i19 == (((self.N3 - delay) % self.N3) * cbk.O3,)

    def test_mode1_beats_mode2_on_delayed_trp(self):
        """With M_v = 1 (N3 = 8, p_v = 1/8) the single common tap cannot carry
        two different TRP delays -- only the mode1 ramp can pre-compensate.
        The base channel is frequency-flat so the inter-TRP ramp is the only
        delay structure (pcL=4 / pc=2 keeps the coefficient budget generous)."""
        rng = np.random.default_rng(8)
        N3 = 8
        H = _delayed_two_trp_channel(rng, N3, 2, ANT.P, 3, sparse=True)
        from nr_csi.baselines.ideal import eigen_precoder

        targets = eigen_precoder(H[-1], rank=1)
        scores = {}
        for mode in ("mode1", "mode2"):
            cbk = R18CJTCodebook(
                ANT, N3=N3, n_trp=2, param_combination_L=4,
                param_combination=2, mode=mode,
            )
            assert cbk.Mv(1) == 1
            W = cbk.precoder(cbk.select(H, rank=1))
            scores[mode] = np.mean([sgcs(targets[t], W[0, t]) for t in range(N3)])
        assert scores["mode1"] >= scores["mode2"] + 0.2

    def test_mode2_reports_no_offsets(self):
        rng = np.random.default_rng(9)
        H = _chan(rng, self.N3, 2, 2 * ANT.P)
        cbk = R18CJTCodebook(
            ANT, N3=self.N3, n_trp=2, param_combination_L=1, param_combination=1,
            mode="mode2",
        )
        pmi = cbk.select(H, rank=1)
        assert pmi.i19 == ()
        assert "i19" not in cbk.overhead_bits(pmi)


# ---------------------------------------------------------------------------
# budgets, structure, selection
# ---------------------------------------------------------------------------


class TestBudgetsAndStructure:
    @pytest.mark.parametrize("n_trp,pcl,pc", [(2, 4, 2), (3, 5, 3), (4, 4, 2)])
    def test_k0_budgets(self, n_trp, pcl, pc):
        rng = np.random.default_rng(n_trp)
        cbk = R18CJTCodebook(
            ANT, N3=8, n_trp=n_trp, param_combination_L=pcl, param_combination=pc
        )
        H = _chan(rng, 8, 4, n_trp * ANT.P)
        for rank in (1, 2, 4):
            pmi = cbk.select(H, rank=rank)
            Ls = cbk.Ls(pmi)
            K0 = cbk.K0_of(Ls)
            assert int(pmi.i17.sum()) <= 2 * K0
            for li in range(rank):
                assert int(pmi.i17[li].sum()) <= K0

    def test_column_norms_and_shape(self):
        rng = np.random.default_rng(11)
        cbk = R18CJTCodebook(
            ANT, N3=8, n_trp=3, param_combination_L=1, param_combination=1
        )
        H = _chan(rng, 8, 4, 3 * ANT.P)
        for rank in (1, 3):
            pmi = cbk.select(H, rank=rank)
            W = cbk.precoder(pmi)
            assert W.shape == (1, 8, 3 * ANT.P, rank)
            assert np.allclose(np.linalg.norm(W, axis=-2), 1 / np.sqrt(rank))

    def test_resource_selection_prefers_strong_trp(self):
        rng = np.random.default_rng(12)
        weak = 0.05 * _chan(rng, 8, 2, ANT.P)
        strong = _chan(rng, 8, 2, ANT.P)
        H = np.concatenate([weak, strong], axis=-1)
        cbk = R18CJTCodebook(
            ANT, N3=8, n_trp=2, param_combination_L=1, param_combination=1,
            restricted_cmr=False, n0=1,
        )
        pmi = cbk.select(H, rank=1)
        assert pmi.resources == (1,)
        W = cbk.precoder(pmi)
        assert np.allclose(W[0, :, : ANT.P, :], 0)  # unselected TRP silent
        bits = cbk.overhead_bits(pmi)
        assert bits["cmr"] == 2

    def test_combination_selection_reported(self):
        rng = np.random.default_rng(13)
        # N_TRP=3: pcL 1 ({2,2,2}) and 5 ({4,4,4}) both admit pc = 1
        cbk = R18CJTCodebook(
            ANT, N3=8, n_trp=3, param_combination_L=[1, 5], param_combination=1
        )
        H = _chan(rng, 8, 2, 3 * ANT.P)
        pmi = cbk.select(H, rank=1)
        assert pmi.i_L == 1  # {4,4,4} has more spatial DoF than {2,2,2}
        assert cbk.overhead_bits(pmi)["i_L"] == 1

    def test_ri_restriction(self):
        cbk = R18CJTCodebook(
            ANT, N3=8, n_trp=2, param_combination_L=1, param_combination=1,
            ri_restriction=[1, 0, 1, 1],
        )
        H = _chan(np.random.default_rng(14), 8, 4, 2 * ANT.P)
        with pytest.raises(ValueError, match="typeII-RI-Restriction-r18"):
            cbk.select(H, rank=2)

    def test_ps_alpha_one_omits_i12(self):
        rng = np.random.default_rng(15)
        cbk = R18CJTPortSelectionCodebook(
            ANT, N3=8, n_trp=2, param_combination_alpha=5, param_combination=2
        )
        H = _chan(rng, 8, 2, 2 * ANT.P)
        pmi = cbk.select(H, rank=1)
        assert pmi.i12 == (None, None)
        assert "i12" not in cbk.overhead_bits(pmi)

    def test_ps_mixed_alpha_selects_ports(self):
        rng = np.random.default_rng(16)
        # {1/2, 1}: resource 1 selects 4 of 8 port pairs, resource 2 all
        cbk = R18CJTPortSelectionCodebook(
            ANT, N3=8, n_trp=2, param_combination_alpha=2, param_combination=1
        )
        H = _chan(rng, 8, 2, 2 * ANT.P)
        pmi = cbk.select(H, rank=1)
        assert pmi.i12[0] is not None and pmi.i12[1] is None
        assert cbk.Ls(pmi) == [4, 8]
        W = cbk.precoder(pmi)
        assert np.allclose(np.linalg.norm(W, axis=-2), 1.0)

    def test_ps_m2_window_taps(self):
        rng = np.random.default_rng(17)
        cbk = R18CJTPortSelectionCodebook(
            ANT, N3=8, n_trp=2, param_combination_alpha=1, param_combination=4,
            N_window=4,
        )
        H = _chan(rng, 8, 2, 2 * ANT.P)
        pmi = cbk.select(H, rank=1)
        taps = cbk.taps(pmi)
        assert taps[0] == 0 and taps[1] in (1, 2, 3)
        assert cbk.overhead_bits(pmi)["i16"] == 2


class TestTableTranscriptions:
    def test_L_combos_shapes(self):
        for n_trp, rows in CJT_L_COMBOS.items():
            for combo in rows.values():
                assert len(combo) == n_trp
                assert all(L in (2, 4, 6) for L in combo)
        assert CJT_L_COMBOS[1][3] == (6,)
        assert CJT_L_COMBOS[3][5] == (4, 4, 4)

    def test_alpha_combos_shapes(self):
        for n_trp, rows in CJT_PS_ALPHA_COMBOS.items():
            for combo in rows.values():
                assert len(combo) == n_trp
        assert CJT_PS_ALPHA_COMBOS[4][4] == (1, 1, 1, 1)

    def test_allowed_tables_cover_all_rows(self):
        for n_trp in (1, 2, 3, 4):
            assert set(CJT_ALLOWED[n_trp]) == set(CJT_L_COMBOS[n_trp])
            assert set(CJT_PS_ALLOWED[n_trp]) == set(CJT_PS_ALPHA_COMBOS[n_trp])
