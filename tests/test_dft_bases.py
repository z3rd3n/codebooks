"""DFT basis anchors: orthogonality, grid partition (paper Fig. pic_6), Kronecker structure."""

import numpy as np
import pytest

from nr_csi.config import SUPPORTED_N1N2, AntennaConfig
from nr_csi.utils import dft


@pytest.fixture(params=[(4, 2), (2, 2), (4, 1), (8, 1), (4, 4)])
def cfg(request):
    N1, N2 = request.param
    return AntennaConfig.standard(N1, N2)


def test_supported_table_validation():
    AntennaConfig.standard(4, 2)  # ok
    with pytest.raises(ValueError):
        AntennaConfig(N1=4, N2=2, O1=4, O2=1)  # wrong oversampling for (4,2)
    with pytest.raises(ValueError):
        AntennaConfig(N1=5, N2=3)  # unsupported shape
    free = AntennaConfig(N1=16, N2=1, O1=4, O2=1, strict=False)
    assert free.P == 32


def test_all_supported_shapes_construct():
    for (N1, N2), (O1, O2) in SUPPORTED_N1N2.items():
        c = AntennaConfig.standard(N1, N2)
        assert (c.O1, c.O2) == (O1, O2)
        assert c.P == 2 * N1 * N2


def test_beam_unit_modulus_and_norm(cfg):
    v = dft.spatial_beam(cfg, 3 % (cfg.N1 * cfg.O1), 0)
    assert v.shape == (cfg.n_ports_per_pol,)
    assert np.allclose(np.abs(v), 1.0)
    assert np.isclose(np.linalg.norm(v) ** 2, cfg.n_ports_per_pol)


def test_kronecker_structure(cfg):
    """v_{m1,m2} = a_{m1} (x) u_{m2} with vertical index running fastest (eq. beamv)."""
    m1, m2 = 5 % (cfg.N1 * cfg.O1), cfg.N2 * cfg.O2 - 1
    v = dft.spatial_beam(cfg, m1, m2)
    for k1 in range(cfg.N1):
        for k2 in range(cfg.N2):
            expected = np.exp(2j * np.pi * (m1 * k1 / (cfg.O1 * cfg.N1) + m2 * k2 / (cfg.O2 * cfg.N2)))
            assert np.isclose(v[k1 * cfg.N2 + k2], expected)


def test_orthogonal_group_is_orthogonal(cfg):
    for q1, q2 in [(0, 0), (cfg.O1 - 1, cfg.O2 - 1), (1 % cfg.O1, 0)]:
        B = dft.orthogonal_group(cfg, q1, q2)
        G = B.conj() @ B.T
        assert np.allclose(G, cfg.n_ports_per_pol * np.eye(cfg.N1 * cfg.N2), atol=1e-9)


def test_groups_partition_full_grid(cfg):
    """The O1*O2 orthogonal groups tile all N1*O1 x N2*O2 oversampled beams (Fig. pic_6)."""
    seen = set()
    for q1 in range(cfg.O1):
        for q2 in range(cfg.O2):
            for n in range(cfg.N1 * cfg.N2):
                n1, n2 = n % cfg.N1, n // cfg.N1
                m1, m2 = dft.beam_index(cfg, q1, q2, n1, n2)
                seen.add((int(m1), int(m2)))
    G1, G2 = cfg.n_beams
    assert seen == {(m1, m2) for m1 in range(G1) for m2 in range(G2)}
    assert len(seen) == G1 * G2


def test_spatial_grid_matches_single_beam(cfg):
    grid = dft.spatial_grid(cfg)
    G1, G2 = cfg.n_beams
    assert grid.shape == (G1, G2, cfg.n_ports_per_pol)
    assert np.allclose(grid[G1 - 1, G2 - 1], dft.spatial_beam(cfg, G1 - 1, G2 - 1))


@pytest.mark.parametrize("N,basis", [(12, dft.freq_basis), (8, dft.time_basis)])
def test_freq_time_bases_are_dft(N, basis):
    F = basis(N, np.arange(N))  # (N, N) full basis
    assert np.allclose(F.conj() @ F.T, N * np.eye(N), atol=1e-9)
    n = 3
    assert np.allclose(basis(N, n), np.exp(2j * np.pi * n * np.arange(N) / N))
