"""Independent closed-form oracle for the R16 eType II Port-Selection
precoder (TS 38.214 Table 5.2.2.2.6-2, whose gamma_{t,l} is given in
5.2.2.2.5-5), at the smallest antenna size (P=4, ports-per-polarization=2)
-- exactly the geometry where the port-aliasing bug (L > P/2) was found.

``_oracle_precoder`` recomputes W straight from the spec's own summation
and gamma formula, using only the PMI's raw integer fields and the spec's
amplitude/phase lookup tables (``R16_REF_AMP``, ``R16_DIFF_AMP``,
``phase_value`` -- data tables, not codebook logic). It deliberately does
NOT call ``R16Type2Codebook._basis``, ``_layer_coefficients``, or
``ls_coefficients``, so it exercises the assembly/normalization arithmetic
independently of the implementation's own helpers.
"""

import numpy as np
import pytest

from nr_csi.codebooks.etype2_r16 import R16Type2Codebook
from nr_csi.config import AntennaConfig
from nr_csi.utils import combinatorics as cb
from nr_csi.utils import quantization as qt

SMALL = AntennaConfig.standard(2, 1)  # N1=2, N2=1 -> P=4, ports-per-pol = 2


def _oracle_precoder(cbk: R16Type2Codebook, pmi) -> np.ndarray:
    half = cbk.antenna.P // 2
    L = cbk.L
    N3 = cbk.N3
    Mv = cbk.Mv(pmi.rank)
    W = np.zeros((N3, cbk.antenna.P, pmi.rank), dtype=complex)
    for li in range(pmi.rank):
        taps = cb.decode_taps(pmi.i16[li], N3, Mv, pmi.i15)  # remapped n3^(f)
        p1 = qt.R16_REF_AMP[pmi.k1[li]]  # (2,) per polarization
        p2 = qt.R16_DIFF_AMP[pmi.k2[li]]  # (Mv, 2L)
        phi = qt.phase_value(pmi.c[li], cbk.N_PSK)  # (Mv, 2L)
        bitmap = pmi.i17[li]  # (Mv, 2L) bool
        for t in range(N3):
            y = np.array([np.exp(2j * np.pi * t * n3 / N3) for n3 in taps])  # (Mv,)
            gamma = 0.0
            per_pol = [np.zeros(half, dtype=complex), np.zeros(half, dtype=complex)]
            for i in range(2 * L):
                pol, beam = divmod(i, L)
                # v_m: a `half`-element column vector with 1 at (m mod half),
                # m = i11_ps * d + beam (TS 38.214 Table 5.2.2.2.6-2)
                port = (pmi.i11_ps * cbk.d + beam) % half
                s = complex(sum(
                    y[f] * p2[f, i] * phi[f, i] for f in range(Mv) if bitmap[f, i]
                ))
                gamma += (p1[pol] ** 2) * abs(s) ** 2
                per_pol[pol][port] += p1[pol] * s
            gamma = gamma if gamma > 0 else 1.0
            W[t, :, li] = np.concatenate(per_pol) / np.sqrt(gamma)
    return W / np.sqrt(pmi.rank)


class TestR16PsClosedFormOracle:
    @pytest.mark.parametrize("combo", [1, 2])  # the only PS combos valid at P=4 (L=2)
    @pytest.mark.parametrize("rank", [1, 2])
    def test_matches_spec_formula_at_p4(self, combo, rank):
        cbk = R16Type2Codebook(SMALL, N3=8, param_combination=combo, port_selection=True, d=1)
        rng = np.random.default_rng(100 + combo * 10 + rank)
        H = rng.standard_normal((1, 8, 2, 4)) + 1j * rng.standard_normal((1, 8, 2, 4))
        pmi = cbk.select(H, rank=rank)
        W_impl = cbk.precoder(pmi)[0]  # (N3, P, rank)
        W_oracle = _oracle_precoder(cbk, pmi)
        assert np.allclose(W_impl, W_oracle, atol=1e-10)
        # the fundamental invariant the L>P/2 aliasing bug broke:
        power = np.sum(np.abs(W_impl) ** 2, axis=(-2, -1))
        assert np.allclose(power, 1.0, atol=1e-9)

    def test_l4_rejected_at_p4(self):
        """combo 3 (L=4) at P=4 (ports-per-pol=2): beam indices i and i+2
        (for i in {0,1}) both alias to physical port i via v_m's "m mod
        P/2" wraparound and the LS fit gives them IDENTICAL coefficients,
        so gamma (which sums |coefficient|^2 per beam index as if they were
        orthogonal) undercounts the coherent sum at that port by exactly
        2x. Must be rejected at construction, matching the base Type II PS
        codebook's explicit 'L=2 when P_CSI-RS=4' rule (38.214 5.2.2.2.4)."""
        with pytest.raises(ValueError, match="ports per polarization"):
            R16Type2Codebook(SMALL, N3=8, param_combination=3, port_selection=True, d=1)
