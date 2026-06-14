"""TS 38.214 Type I single-panel ranks 3-8."""

import numpy as np
import pytest

from nr_csi.codebooks.serialize import pack, unpack
from nr_csi.codebooks.type1 import (
    Type1Codebook,
    Type1PMI,
    i13_offsets_rank34,
)
from nr_csi.config import SUPPORTED_N1N2, AntennaConfig


def steering(n, o, index):
    return np.exp(2j * np.pi * index * np.arange(n) / (o * n))


def beam(a, l, m):
    return np.kron(steering(a.N1, a.O1, l), steering(a.N2, a.O2, m))


def test_rank34_small_closed_form():
    a = AntennaConfig.standard(4, 1)
    cbk = Type1Codebook(a)
    phi = 1j
    for rank in (3, 4):
        pmi = Type1PMI(rank, 1, 3, 0, np.array([1]), i13=2)
        v1 = beam(a, 3, 0)
        k1, k2 = i13_offsets_rank34(a.N1, a.N2, a.O1, a.O2)[2]
        v2 = beam(a, 3 + k1, k2)
        columns = [
            np.r_[v1, phi * v1],
            np.r_[v2, phi * v2],
            np.r_[v1, -phi * v1],
        ]
        if rank == 4:
            columns.append(np.r_[v2, -phi * v2])
        expected = np.stack(columns, axis=1) / np.sqrt(rank * a.P)
        assert np.allclose(cbk.precoder(pmi)[0, 0], expected)


def test_rank34_large_half_array_closed_form():
    a = AntennaConfig.standard(4, 2)
    cbk = Type1Codebook(a)
    l, m, p, n = 2, 3, 3, 1
    half = np.kron(steering(a.N1 // 2, a.O1, l), steering(a.N2, a.O2, m))
    phi = np.exp(1j * np.pi * n / 2)
    theta = np.exp(1j * np.pi * p / 4)
    for rank, coefficients in [
        (
            3,
            [
                [1, 1, 1],
                [theta, -theta, theta],
                [phi, phi, -phi],
                [phi * theta, -phi * theta, -phi * theta],
            ],
        ),
        (
            4,
            [
                [1, 1, 1, 1],
                [theta, -theta, theta, -theta],
                [phi, phi, -phi, -phi],
                [phi * theta, -phi * theta, -phi * theta, phi * theta],
            ],
        ),
    ]:
        pmi = Type1PMI(rank, 1, l, m, np.array([n]), i13=p)
        expected = np.concatenate(
            [np.asarray(row) * half[:, None] for row in coefficients], axis=0
        ) / np.sqrt(rank * a.P)
        assert np.allclose(cbk.precoder(pmi)[0, 0], expected)


def test_rank5_to_8_closed_form_and_power():
    a = AntennaConfig.standard(4, 2)
    cbk = Type1Codebook(a)
    v1, v2, v3, v4 = [beam(a, *index) for index in [(1, 2), (5, 2), (1, 6), (5, 6)]]
    expected_blocks = {
        5: (
            [v1, v1, v2, v2, v4],
            [1j * v1, -1j * v1, v2, -v2, v4],
        ),
        6: (
            [v1, v1, v2, v2, v4, v4],
            [1j * v1, -1j * v1, 1j * v2, -1j * v2, v4, -v4],
        ),
        7: (
            [v1, v1, v2, v3, v3, v4, v4],
            [1j * v1, -1j * v1, 1j * v2, v3, -v3, v4, -v4],
        ),
        8: (
            [v1, v1, v2, v2, v3, v3, v4, v4],
            [1j * v1, -1j * v1, 1j * v2, -1j * v2, v3, -v3, v4, -v4],
        ),
    }
    for rank, (top, bottom) in expected_blocks.items():
        pmi = Type1PMI(rank, 1, 1, 2, np.array([1]))
        expected = np.stack(
            [np.r_[up, down] for up, down in zip(top, bottom)], axis=1
        ) / np.sqrt(rank * a.P)
        W = cbk.precoder(pmi)[0, 0]
        assert np.allclose(W, expected)
        assert np.allclose(W.conj().T @ W, np.eye(rank) / rank, atol=1e-12)


@pytest.mark.parametrize("n1,n2", sorted(SUPPORTED_N1N2))
def test_all_configurations_modes_and_ranks_are_semi_unitary(n1, n2):
    a = AntennaConfig.standard(n1, n2)
    for mode in (1, 2):
        cbk = Type1Codebook(a, mode=mode)
        for rank in range(1, min(8, a.P) + 1):
            pmi = Type1PMI(
                rank,
                mode,
                0,
                0,
                np.array([0]),
                0 if rank in (2, 3, 4) else None,
            )
            W = cbk.precoder(pmi)[0, 0]
            assert np.allclose(W.conj().T @ W, np.eye(rank) / rank, atol=1e-12)


@pytest.mark.parametrize("n1,n2", [(2, 1), (4, 1), (4, 2)])
def test_all_supported_ranks_round_trip(n1, n2):
    a = AntennaConfig.standard(n1, n2)
    cbk = Type1Codebook(a, N3=2)
    for rank in range(1, min(8, a.P) + 1):
        pmi = Type1PMI(
            rank,
            1,
            0,
            0,
            np.array([0, cbk._n_i2(rank) - 1]),
            0 if rank in (2, 3, 4) else None,
        )
        bits = pack(cbk, pmi)
        restored = unpack(cbk, bits, rank)
        assert len(bits) == cbk.total_overhead_bits(pmi)
        assert vars(restored).keys() == vars(pmi).keys()
        assert np.array_equal(restored.i2, pmi.i2)
        assert restored.i13 == pmi.i13
        assert np.allclose(cbk.precoder(restored), cbk.precoder(pmi))


def test_rank8_is_supported_for_eight_ports_and_i12_is_omitted():
    a = AntennaConfig.standard(4, 1)
    cbk = Type1Codebook(a)
    H = np.ones((1, 1, 8, a.P), dtype=complex)
    pmi = cbk.select(H, rank=8)
    assert "i12" not in cbk.overhead_bits(pmi)
    assert cbk.precoder(pmi).shape == (1, 1, 8, 8)
    assert cbk._n_i11(8) == a.n_beams[0] // 2


def test_rank78_vertical_index_uses_half_grid_for_n2_2():
    a = AntennaConfig.standard(4, 2)
    cbk = Type1Codebook(a)
    assert cbk._n_i12(7) == a.n_beams[1] // 2
    assert cbk._n_i12(8) == a.n_beams[1] // 2


def test_high_rank_restriction_and_invalid_rank():
    a = AntennaConfig.standard(4, 2)
    restriction = np.ones(8, dtype=bool)
    restriction[5] = False
    cbk = Type1Codebook(a, rank_restriction=restriction)
    H = np.ones((1, 1, 8, a.P), dtype=complex)
    with pytest.raises(ValueError, match="prohibited"):
        cbk.select(H, rank=6)
    with pytest.raises(ValueError, match="unsupported"):
        cbk.select(H, rank=9)
