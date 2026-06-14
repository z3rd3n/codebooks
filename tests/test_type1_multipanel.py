"""TS 38.214 Type I multi-panel codebook."""

import numpy as np
import pytest

from nr_csi.codebooks.serialize import pack, unpack
from nr_csi.codebooks.type1_multipanel import (
    Type1MPPMI,
    Type1MultiPanelCodebook,
    i13_offsets_multipanel_rank34,
)
from nr_csi.config import SUPPORTED_NG_N1N2, AntennaConfig
from nr_csi.eval import evaluate


def steering(n, o, index):
    return np.exp(2j * np.pi * index * np.arange(n) / (o * n))


def beam(a, l, m):
    return np.kron(steering(a.N1, a.O1, l), steering(a.N2, a.O2, m))


def test_configuration_table_and_port_counts():
    assert len(SUPPORTED_NG_N1N2) == 8
    for (ng, n1, n2), oversampling in SUPPORTED_NG_N1N2.items():
        a = AntennaConfig.standard(n1, n2, Ng=ng)
        assert (a.O1, a.O2) == oversampling
        assert a.P == 2 * ng * n1 * n2


def test_mode1_rank4_closed_form():
    a = AntennaConfig.standard(2, 2, Ng=2)
    cbk = Type1MultiPanelCodebook(a)
    pmi = Type1MPPMI(4, 1, 1, 2, (3,), np.array([1]), i13=1)
    k1, k2 = i13_offsets_multipanel_rank34(a.N1, a.N2, a.O1, a.O2)[1]
    v1, v2 = beam(a, 1, 2), beam(a, 1 + k1, 2 + k2)
    phi, panel = 1j, -1j

    def family(v, sign):
        return np.r_[v, sign * phi * v, panel * v, sign * phi * panel * v] / np.sqrt(a.P)

    expected = np.stack(
        [family(v1, 1), family(v2, 1), family(v1, -1), family(v2, -1)], axis=1
    ) / 2
    assert np.allclose(cbk.precoder(pmi)[0, 0], expected)


def test_mode2_rank2_closed_form():
    a = AntennaConfig.standard(2, 2, Ng=2)
    cbk = Type1MultiPanelCodebook(a, mode=2)
    pmi = Type1MPPMI(2, 2, 3, 1, (2, 1), np.array([[1, 0, 1]]), i13=2)
    from nr_csi.codebooks.type1 import i13_offsets

    k1, k2 = i13_offsets(a.N1, a.N2, a.O1, a.O2)[2]
    v1, v2 = beam(a, 3, 1), beam(a, 3 + k1, 1 + k2)
    phi = 1j
    phase1 = np.exp(1j * np.pi / 4) * np.exp(1j * np.pi) * np.exp(-1j * np.pi / 4)
    phase2 = (
        np.exp(1j * np.pi / 4)
        * np.exp(1j * np.pi / 2)
        * np.exp(-1j * np.pi / 4)
        * np.exp(1j * np.pi / 2)
    )
    family1 = np.r_[v1, phi * v1, phase1 * v1, phase2 * v1] / np.sqrt(a.P)
    family2 = np.r_[v2, -phi * v2, phase1 * v2, -phase2 * v2] / np.sqrt(a.P)
    expected = np.stack([family1, family2], axis=1) / np.sqrt(2)
    assert np.allclose(cbk.precoder(pmi)[0, 0], expected)


@pytest.mark.parametrize("key", sorted(SUPPORTED_NG_N1N2))
def test_all_table_rows_are_semi_unitary(key):
    ng, n1, n2 = key
    a = AntennaConfig.standard(n1, n2, Ng=ng)
    for mode in ([1, 2] if ng == 2 else [1]):
        cbk = Type1MultiPanelCodebook(a, mode=mode)
        for rank in range(1, 5):
            i14 = (0,) * cbk._i14_shape()[0]
            i2 = np.array([0]) if mode == 1 else np.array([[0, 0, 0]])
            pmi = Type1MPPMI(rank, mode, 0, 0, i14, i2, 0 if rank > 1 else None)
            W = cbk.precoder(pmi)[0, 0]
            assert np.allclose(W.conj().T @ W, np.eye(rank) / rank, atol=1e-12)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_serialization_round_trip(mode, rank):
    a = AntennaConfig.standard(2, 2, Ng=2)
    cbk = Type1MultiPanelCodebook(a, N3=2, mode=mode)
    i14 = (1,) if mode == 1 else (1, 2)
    i2 = np.array([1, 0]) if mode == 1 else np.array([[1, 0, 1], [0, 1, 0]])
    pmi = Type1MPPMI(rank, mode, 2, 3, i14, i2, 0 if rank > 1 else None)
    bits = pack(cbk, pmi)
    restored = unpack(cbk, bits, rank)
    assert len(bits) == cbk.total_overhead_bits(pmi)
    assert restored.i14 == pmi.i14
    assert np.array_equal(restored.i2, pmi.i2)
    assert np.allclose(cbk.precoder(restored), cbk.precoder(pmi))


def test_guards_and_ri_restriction():
    with pytest.raises(ValueError, match="Ng=2"):
        Type1MultiPanelCodebook(AntennaConfig.standard(2, 1, Ng=4), mode=2)
    a = AntennaConfig.standard(2, 1, Ng=2)
    cbk = Type1MultiPanelCodebook(a, rank_restriction=np.array([1, 1, 0, 1]))
    H = np.ones((1, 1, 4, a.P), dtype=complex)
    with pytest.raises(ValueError, match="prohibited"):
        cbk.select(H, rank=3)
    pmi = Type1MPPMI(1, 1, 0, 0, (0,), np.array([0]))
    assert "i12" not in cbk.overhead_bits(pmi)


def test_harness_integration():
    a = AntennaConfig.standard(2, 1, Ng=2)

    class RandomChannel:
        def generate(self, n_slots=1, rng=None):
            rng = rng or np.random.default_rng()
            shape = (n_slots, 2, 4, a.P)
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    result = evaluate(
        Type1MultiPanelCodebook(a, N3=2),
        RandomChannel(),
        rank=4,
        n_drops=2,
        snr_db=[10],
    )
    assert 0 <= result.subspace_sgcs <= 1
