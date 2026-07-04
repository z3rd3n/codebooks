"""Property-based randomized invariants over all supported antenna
configurations and codebooks (plan B1).

For every (N1, N2) of Table tabNO, every codebook family, and ranks 1-2
(3-4 separately for the R16+ families): random seeded channels, then

* precoder shape, finiteness, tr(W^H W) = 1 per (interval, t), per-layer
  column norm 1/sqrt(v);
* every PMI integer field within its spec range (the ``validate_pmi``
  helpers, which ``precoder`` also runs);
* ``overhead_bits`` deterministic across calls, all entries >= 0;
* select -> precoder -> select on the reconstructed precoder's own channel
  must not lose fit (SGCS non-decreasing up to quantizer noise).
"""

import numpy as np
import pytest

from nr_csi.baselines.ideal import eigen_precoder
from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18PredictedPortSelectionCodebook,
    R18Type2Codebook,
    Type1Codebook,
    Type1MultiPanelCodebook,
)
from nr_csi.codebooks.validate import (
    validate_r15,
    validate_r16,
    validate_r17,
    validate_r18,
    validate_type1,
    validate_type1_multipanel,
)
from nr_csi.config import SUPPORTED_N1N2, AntennaConfig
from nr_csi.metrics.similarity import sgcs

N3 = 4
ALL_CONFIGS = sorted(SUPPORTED_N1N2)
IDEMPOTENCE_CONFIGS = [(2, 1), (4, 2), (16, 1), (3, 2), (4, 4)]


def make_schemes(ant: AntennaConfig):
    return [
        (Type1Codebook(ant, N3=N3), validate_type1),
        (R15Type2Codebook(ant, N3=N3, L=2), validate_r15),
        (R15Type2Codebook(ant, N3=N3, L=2, port_selection=True, d=2), validate_r15),
        (R16Type2Codebook(ant, N3=N3, param_combination=2), validate_r16),
        # port-selection: L must not exceed ports-per-polarization (else the
        # v_m = 1{m mod P/2} basis aliases distinct beam indices onto the same
        # physical port); combo 2 (L=2) always fits, combo 4 (L=4) only fits
        # when P > 4.
        (
            R16Type2Codebook(
                ant, N3=N3,
                param_combination=2 if ant.n_ports_per_pol <= 2 else 4,
                port_selection=True, d=1,
            ),
            validate_r16,
        ),
        # combo 5 at P = 4 requires ranks 3-4 disallowed (5.2.2.2.7); the
        # tests here use ranks 1-2 only.
        (
            R17Type2Codebook(
                ant, N3=N3, param_combination=5,
                ri_restriction=[1, 1, 0, 0] if ant.P == 4 else None,
            ),
            validate_r17,
        ),
        (R18Type2Codebook(ant, N3=N3, N4=2, param_combination=2), validate_r18),
    ]


def random_channel(rng, n_slots, P, n_rx=2):
    return rng.standard_normal((n_slots, N3, n_rx, P)) + 1j * rng.standard_normal(
        (n_slots, N3, n_rx, P)
    )


def check_precoder(W, rank, P, n_intervals):
    assert W.shape == (n_intervals, N3, P, rank)
    assert np.all(np.isfinite(W))
    col_norms = np.linalg.norm(W, axis=-2)  # (S, N3, v)
    assert np.allclose(col_norms, 1.0 / np.sqrt(rank), atol=1e-9)


@pytest.mark.parametrize("n1,n2", ALL_CONFIGS, ids=str)
@pytest.mark.parametrize("rank", [1, 2])
def test_all_configs_all_codebooks(n1, n2, rank):
    ant = AntennaConfig.standard(n1, n2)
    rng = np.random.default_rng(1000 * n1 + n2 + rank)
    for cbk, validator in make_schemes(ant):
        n_slots = getattr(cbk, "N4", 1)
        H = random_channel(rng, n_slots, ant.P)
        pmi = cbk.select(H, rank=rank)
        validator(cbk, pmi)
        W = cbk.precoder(pmi)
        check_precoder(W, rank, ant.P, n_slots)
        bits = cbk.overhead_bits(pmi)
        assert all(v >= 0 for v in bits.values()) and sum(bits.values()) > 0
        assert cbk.overhead_bits(pmi) == bits  # deterministic


@pytest.mark.parametrize("rank", [3, 4])
def test_high_ranks_r16_r17_r18(rank):
    ant = AntennaConfig.standard(4, 2)
    rng = np.random.default_rng(rank)
    schemes = [
        (R16Type2Codebook(ant, N3=N3, param_combination=5), validate_r16),
        (R17Type2Codebook(ant, N3=N3, param_combination=5), validate_r17),
        (R18Type2Codebook(ant, N3=N3, N4=2, param_combination=5), validate_r18),
    ]
    for cbk, validator in schemes:
        n_slots = getattr(cbk, "N4", 1)
        H = random_channel(rng, n_slots, ant.P, n_rx=4)
        pmi = cbk.select(H, rank=rank)
        validator(cbk, pmi)
        check_precoder(cbk.precoder(pmi), rank, ant.P, n_slots)


@pytest.mark.parametrize("rank", [3, 4, 5, 6, 7, 8])
def test_type1_high_rank_invariants(rank):
    ant = AntennaConfig.standard(4, 2)
    rng = np.random.default_rng(100 + rank)
    cbk = Type1Codebook(ant, N3=N3)
    H = random_channel(rng, 1, ant.P, n_rx=8)
    pmi = cbk.select(H, rank)
    validate_type1(cbk, pmi)
    check_precoder(cbk.precoder(pmi), rank, ant.P, 1)


def test_new_family_invariants():
    rng = np.random.default_rng(44)
    ant = AntennaConfig.standard(2, 1, Ng=2)
    mp = Type1MultiPanelCodebook(ant, N3=N3, mode=2)
    H = random_channel(rng, 1, ant.P, n_rx=4)
    pmi = mp.select(H, rank=4)
    validate_type1_multipanel(mp, pmi)
    check_precoder(mp.precoder(pmi), 4, ant.P, 1)

    ant = AntennaConfig.standard(4, 2)
    predicted = R18PredictedPortSelectionCodebook(ant, N3=N3, param_combination=5)
    H = random_channel(rng, 1, ant.P, n_rx=4)
    pmi = predicted.select(H, rank=4)
    validate_r17(predicted, pmi)
    check_precoder(predicted.precoder(pmi), 4, ant.P, 1)


@pytest.mark.parametrize("n1,n2", IDEMPOTENCE_CONFIGS, ids=str)
@pytest.mark.parametrize("rank", [1, 2])
def test_reselection_does_not_lose_fit(n1, n2, rank):
    """A codebook-representable target must be re-encoded at least as well
    as the original generic channel was: SGCS(W, W') >= SGCS(ideal, W) - eps.
    Layer weights break the singular-value degeneracy of the W^H channel."""
    ant = AntennaConfig.standard(n1, n2)
    rng = np.random.default_rng(17 * n1 + n2)
    D = np.diag([1.0, 0.6][:rank])
    for cbk, _ in make_schemes(ant):
        n_slots = getattr(cbk, "N4", 1)
        H = random_channel(rng, n_slots, ant.P)
        pmi = cbk.select(H, rank=rank)
        W = cbk.precoder(pmi)
        s1 = sgcs(eigen_precoder(H, rank=rank), W)
        H2 = D @ np.swapaxes(W, -1, -2).conj()  # channel made of W itself
        W2 = cbk.precoder(cbk.select(H2, rank=rank))
        s2 = sgcs(W, W2)
        assert s2 >= s1 - 0.05, f"{cbk.name}: refit SGCS {s2:.3f} < original {s1:.3f}"
        assert s2 >= (0.7 if rank == 1 else 0.4), f"{cbk.name}: refit SGCS {s2:.3f}"
