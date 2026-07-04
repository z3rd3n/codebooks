"""Multi-panel (Ng > 1) support in the synthetic ray channel.

Both channel classes used to hard-assume a single dual-polarized panel
(``P == 2*N1*N2``), so any Ng>1 ``AntennaConfig`` -- the ``type1-mp`` and
``refined-type1mp-r19`` catalog codebooks -- raised a broadcast ``ValueError``
deep inside ``generate()``. These tests pin the fixed behaviour: correct
output shape, and (for the deterministic ``SyntheticRayChannel``) exact PMI
recovery, which also confirms the port ordering matches what
``Type1MultiPanelCodebook``/``RefinedType1MultiPanelCodebook`` expect
(panel-major, then polarization-major).
"""

import numpy as np
import pytest

from nr_csi.baselines.ideal import eigen_precoder
from nr_csi.channel.synthetic import RandomRayChannel, Ray, SyntheticRayChannel
from nr_csi.codebooks.refined_type1mp_r19 import RefinedType1MultiPanelCodebook
from nr_csi.codebooks.type1_multipanel import Type1MultiPanelCodebook
from nr_csi.config import AntennaConfig
from nr_csi.metrics.similarity import sgcs


@pytest.mark.parametrize("ng", [2, 4])
def test_random_ray_channel_multipanel_shape(ng):
    """Previously: ValueError, operands could not be broadcast together."""
    a = AntennaConfig.standard(2, 1, Ng=ng)
    chan = RandomRayChannel(a, N3=4, n_rx=2, n_paths=3)
    H = chan.generate(n_slots=2, rng=np.random.default_rng(0))
    assert H.shape == (2, 4, 2, a.P)
    assert a.P == 2 * ng * a.N1 * a.N2
    assert np.all(np.isfinite(H))


def test_random_ray_channel_single_panel_unchanged():
    """Ng=1 behaviour (the pre-existing, already-tested path) must not move."""
    a = AntennaConfig.standard(4, 2)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    chan = RandomRayChannel(a, N3=4, n_rx=2, n_paths=3)
    H1 = chan.generate(n_slots=1, rng=rng1)
    H2 = chan.generate(n_slots=1, rng=rng2)
    assert np.allclose(H1, H2)
    assert H1.shape == (1, 4, 2, a.P)


def test_panel_phase_wrong_shape_raises():
    a = AntennaConfig.standard(2, 1, Ng=2)
    ray = Ray(gain=1.0, m1=0.0, panel_phase=np.array([0.0]))  # needs 2 entries
    with pytest.raises(ValueError, match="panel_phase"):
        SyntheticRayChannel(a, [ray], N3=1).generate()


def test_type1_multipanel_exact_recovery_on_grid():
    """A single on-grid ray gives a rank-1 physical channel, so only rank 1
    has a unique eigen-target to compare against; this pins the port layout
    (panel-major, then polarization-major) that ranks > 1 also rely on."""
    a = AntennaConfig.standard(2, 2, Ng=2)
    ray = Ray(gain=1.0, m1=0, m2=0, pol_phase=np.pi / 2, panel_phase=np.array([0.0, np.pi]))
    H = SyntheticRayChannel(a, [ray], N3=1).generate()
    assert H.shape == (1, 1, 1, a.P)
    cbk = Type1MultiPanelCodebook(a, N3=1, mode=1)
    pmi = cbk.select(H, rank=1)
    W = cbk.precoder(pmi)
    target = eigen_precoder(H[-1], rank=1)
    assert sgcs(target[0], W[0, 0]) > 0.999


def test_refined_type1_multipanel_exact_recovery_on_grid():
    a = AntennaConfig.standard(4, 3, Ng=2)  # 48 ports, Table 5.2.2.2.2a-1
    ray = Ray(gain=1.0, m1=2, m2=1, pol_phase=np.pi, panel_phase=np.array([0.0, np.pi / 2]))
    H = SyntheticRayChannel(a, [ray], N3=1).generate()
    assert H.shape == (1, 1, 1, a.P)
    cbk = RefinedType1MultiPanelCodebook(a, N3=1)
    pmi = cbk.select(H, rank=1)
    W = cbk.precoder(pmi)
    target = eigen_precoder(H[-1], rank=1)
    assert sgcs(target[0], W[0, 0]) > 0.999


def test_random_ray_channel_multipanel_runs_through_multipanel_codebooks():
    """The bug's exact failure mode: RandomRayChannel + a multi-panel
    codebook's select()/precoder() round trip, end to end."""
    a1 = AntennaConfig.standard(2, 2, Ng=2)  # R15 Table 5.2.2.2.2-1
    a2 = AntennaConfig.standard(4, 3, Ng=2)  # R19 Table 5.2.2.2.2a-1
    for a, cbk in (
        (a1, Type1MultiPanelCodebook(a1, N3=4, mode=1)),
        (a2, RefinedType1MultiPanelCodebook(a2, N3=4)),
    ):
        chan = RandomRayChannel(a, N3=4, n_rx=2, n_paths=3)
        H = chan.generate(n_slots=1, rng=np.random.default_rng(7))
        pmi = cbk.select(H, rank=2)
        W = cbk.precoder(pmi)
        assert np.all(np.isfinite(W))
        assert W.shape == (1, 4, a.P, 2)
