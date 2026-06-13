"""Focused correctness extensions closing genuine gaps in the suite.

Four independent angles, all reusing the existing equation-level oracles
(``tests.paper_oracles``) and config constants -- no production code is
exercised that the oracles also touch:

* end-to-end rank-3/4 rejection through the *public* codebook API (not just
  the ``R16ParamCombo.p_v`` table guard);
* spatial-DFT reconstruction for *every* supported ``(N1,N2)`` geometry,
  comparing the production precoder against the paper-equation oracle;
* genuine inter-layer orthonormality of the Type I precoder (Gram = I/rank),
  strengthening the existing column-norm-only checks;
* the restriction-group combinatorial codec at ``O1*O2 != 16`` (only the
  ``O1*O2 = 16`` case was previously covered).
"""

from __future__ import annotations

import numpy as np
import pytest

from nr_csi.codebooks import (
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
)
from nr_csi.config import SUPPORTED_N1N2, AntennaConfig
from nr_csi.utils import combinatorics as cb
from tests.paper_oracles import r16_precoder, r17_precoder, r18_precoder


def complex_channel(seed: int, n_slots: int, N3: int, n_rx: int, P: int) -> np.ndarray:
    """i.i.d. complex-Gaussian channel [slot, freq, rx, port] (matches the
    helper in test_paper_reconstruction_oracles.py)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_slots, N3, n_rx, P)) + 1j * rng.standard_normal(
        (n_slots, N3, n_rx, P)
    )


# ---------------------------------------------------------------------------
# Class 1 -- end-to-end rank-3/4 rejection through the public codebook API
# ---------------------------------------------------------------------------


class TestUnsupportedRankRejection:
    """``paramCombination`` rows with ``p_v34 = None`` must reject ranks 3-4 at
    every public entry point, not only at the ``R16ParamCombo.p_v`` table guard
    covered by test_config_tables.py."""

    # (4,2): N1*N2 = 8 >= L = 6, so the L=6 combos are constructible.
    N3 = 12

    @pytest.mark.parametrize("combo", [7, 8])
    def test_r16_rank34_rejected(self, combo):
        antenna = AntennaConfig.standard(4, 2)
        cbk = R16Type2Codebook(antenna, N3=self.N3, param_combination=combo)
        with pytest.raises(ValueError):
            cbk.Mv(3)
        with pytest.raises(ValueError):
            cbk.Mv(4)
        # n_rx = 4 lets the eigen-target step succeed so selection actually
        # reaches the Mv() call that raises (rather than failing earlier).
        H = complex_channel(11000 + combo, 1, self.N3, 4, antenna.P)
        with pytest.raises(ValueError):
            cbk.select(H, rank=3)

    @pytest.mark.parametrize("combo", [5, 6])
    def test_r16_rank34_supported(self, combo):
        antenna = AntennaConfig.standard(4, 2)
        cbk = R16Type2Codebook(antenna, N3=self.N3, param_combination=combo)
        assert cbk.Mv(3) >= 1
        assert cbk.Mv(4) >= 1

    @pytest.mark.parametrize("combo", [8, 9])
    def test_r18_rank34_rejected(self, combo):
        antenna = AntennaConfig.standard(4, 2)
        cbk = R18Type2Codebook(antenna, N3=self.N3, N4=2, param_combination=combo)
        with pytest.raises(ValueError):
            cbk.Mv(3)
        with pytest.raises(ValueError):
            cbk.Mv(4)
        H = complex_channel(12000 + combo, 2, self.N3, 4, antenna.P)
        with pytest.raises(ValueError):
            cbk.select(H, rank=3)

    @pytest.mark.parametrize("combo", [4, 5, 6, 7])
    def test_r18_rank34_supported(self, combo):
        antenna = AntennaConfig.standard(4, 2)
        cbk = R18Type2Codebook(antenna, N3=self.N3, N4=2, param_combination=combo)
        assert cbk.Mv(3) >= 1
        assert cbk.Mv(4) >= 1


# ---------------------------------------------------------------------------
# Class 2 -- spatial-DFT reconstruction for every supported array geometry
# ---------------------------------------------------------------------------

ALL_SHAPES = sorted(SUPPORTED_N1N2)
# Smallest (4-port), a 16-port, a 32-port, and the 32-port single-row array.
RANK2_SHAPES = {(2, 1), (4, 2), (4, 4), (16, 1)}
_N3 = 8  # <= 19 keeps the delay codec single-level (no i15)


def _r16_combo(antenna: AntennaConfig) -> int:
    """Largest paramCombination-r16 whose L fits the N1*N2 beam group."""
    npp = antenna.n_ports_per_pol
    if npp >= 6:
        return 7  # L = 6
    if npp >= 4:
        return 4  # L = 4
    return 1  # L = 2


def _r18_combo(antenna: AntennaConfig) -> int:
    npp = antenna.n_ports_per_pol
    if npp >= 6:
        return 8  # L = 6
    if npp >= 4:
        return 3  # L = 4
    return 2  # L = 2


def _build_case(family: str, antenna: AntennaConfig, rank: int, seed: int):
    """Return (codebook, oracle_fn, channel) for one reconstruction node."""
    n_rx = max(rank, 2)
    if family == "r16":
        cbk = R16Type2Codebook(antenna, N3=_N3, param_combination=_r16_combo(antenna))
        H = complex_channel(seed, 1, _N3, n_rx, antenna.P)
        return cbk, r16_precoder, H
    if family == "r16_ps":
        cbk = R16Type2Codebook(
            antenna, N3=_N3, param_combination=_r16_combo(antenna), port_selection=True, d=1
        )
        H = complex_channel(seed, 1, _N3, n_rx, antenna.P)
        return cbk, r16_precoder, H
    if family == "r17":
        # combo 5: alpha = 1/2 (free port selection exercised via i12), M = 2.
        cbk = R17Type2Codebook(antenna, N3=_N3, param_combination=5, N_window=4)
        H = complex_channel(seed, 1, _N3, n_rx, antenna.P)
        return cbk, r17_precoder, H
    if family == "r18":
        cbk = R18Type2Codebook(antenna, N3=_N3, N4=2, param_combination=_r18_combo(antenna))
        H = complex_channel(seed, 2, _N3, n_rx, antenna.P)
        return cbk, r18_precoder, H
    raise AssertionError(f"unknown family {family}")


_FAMILIES = ("r16", "r16_ps", "r17", "r18")


def _recon_cases():
    cases = []
    for fi, family in enumerate(_FAMILIES):
        for si, shape in enumerate(ALL_SHAPES):
            for rank in (1, 2) if shape in RANK2_SHAPES else (1,):
                seed = 90000 + 1000 * fi + 10 * si + rank
                cases.append((family, shape, rank, seed))
    return cases


class TestReconstructionAcrossAllArrays:
    """For every supported geometry, the production precoder must equal the
    independent paper-equation oracle bit-for-bit.  Because the oracle builds
    beams straight from m = O*n + q (paper_oracles.regular_basis), this also
    pins down the beam-grid oversampling formula for all 13 arrays."""

    @pytest.mark.parametrize(
        "family,shape,rank,seed",
        _recon_cases(),
        ids=lambda v: v if isinstance(v, str) else repr(v),
    )
    def test_production_precoder_matches_paper_oracle(self, family, shape, rank, seed):
        antenna = AntennaConfig.standard(*shape)
        cbk, oracle, H = _build_case(family, antenna, rank, seed)
        pmi = cbk.select(H, rank=rank)
        assert np.allclose(cbk.precoder(pmi), oracle(cbk, pmi), atol=1e-10)


# ---------------------------------------------------------------------------
# Class 3 -- Type I inter-layer orthonormality (Gram = I / rank)
# ---------------------------------------------------------------------------

# (shape, mode); mode 2 requires N2 > 1.  Shapes span every i13_offsets branch:
# N1==2,N2==1 / N1>2,N2==1 / N1==N2 / N1>N2>1.
TYPE1_ORTHO_CASES = [
    ((2, 1), 1),
    ((4, 1), 1),
    ((2, 2), 1),
    ((2, 2), 2),
    ((4, 2), 1),
    ((4, 2), 2),
    ((4, 4), 1),
    ((4, 4), 2),
]


class TestType1Orthonormality:
    """Type I rank-2 precoders are genuinely orthonormal (DFT beam + +/- co-
    phasing), unlike the quantized Type II families.  Asserting the full Gram
    matrix per subband strengthens the existing column-norm-only checks."""

    @pytest.mark.parametrize("shape,mode", TYPE1_ORTHO_CASES)
    def test_rank2_columns_are_orthonormal_per_subband(self, shape, mode):
        antenna = AntennaConfig.standard(*shape)
        cbk = Type1Codebook(antenna, N3=3, mode=mode)
        H = complex_channel(13000 + 10 * mode + antenna.P, 1, 3, 2, antenna.P)
        pmi = cbk.select(H, rank=2)
        W = cbk.precoder(pmi)  # (1, N3, P, 2)
        for t in range(cbk.N3):
            gram = W[0, t].conj().T @ W[0, t]
            assert np.allclose(gram, np.eye(2) / 2, atol=1e-12)


# ---------------------------------------------------------------------------
# Class 4 -- restriction-group codec at O1*O2 != 16
# ---------------------------------------------------------------------------


class TestRestrictionGroupCodecEdges:
    """test_combinatorics.py only round-trips the restriction-group codec at
    O1*O2 = 16.  At O1*O2 = 4 there is exactly one way to pick the 4 groups
    (C(4,4) = 1), the degenerate single-selection edge; the round-trip must
    still hold for both 1-D group layouts."""

    @pytest.mark.parametrize("O1,O2", [(4, 1), (1, 4)])
    def test_restriction_group_roundtrip_oo4(self, O1, O2):
        g_combo = list(range(O1 * O2))  # the only 4-of-4 selection
        r1 = [g % O1 for g in g_combo]
        r2 = [g // O1 for g in g_combo]
        beta1 = cb.encode_restriction_groups(r1, r2, O1, O2)
        assert beta1 == 0  # index of the single C(4,4)=1 combination
        g, d1, d2 = cb.decode_restriction_groups(beta1, O1, O2)
        assert g == g_combo
        assert (d1, d2) == (r1, r2)
