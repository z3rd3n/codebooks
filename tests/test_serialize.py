"""Bit-level serialization round-trip (plan B2).

The two properties asserted for every codebook family, variant, and rank:

1. ``unpack(pack(pmi)) == pmi``  (field-exact reconstruction)
2. ``len(pack(pmi)) == total_overhead_bits(pmi)``  (the overhead formulas
   count exactly the bits that exist -- no phantom or missing elements)
"""

import numpy as np
import pytest

from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
    Type1MultiPanelCodebook,
)
from nr_csi.codebooks.serialize import pack, unpack
from nr_csi.config import AntennaConfig

ANT = AntennaConfig.standard(4, 2)  # P = 16


def random_channel(rng, n_slots, N3, n_rx=4, P=ANT.P):
    return rng.standard_normal((n_slots, N3, n_rx, P)) + 1j * rng.standard_normal(
        (n_slots, N3, n_rx, P)
    )


def pmi_fields_equal(a, b) -> bool:
    for name in vars(a):
        va, vb = getattr(a, name), getattr(b, name)
        if isinstance(va, np.ndarray):
            if not np.array_equal(va, np.asarray(vb)):
                return False
        elif isinstance(va, list):
            if list(va) != list(vb):
                return False
        elif va != vb:
            return False
    return True


SCHEMES = [
    ("type1-mode1", lambda: Type1Codebook(ANT, N3=4), tuple(range(1, 9))),
    ("type1-mode2", lambda: Type1Codebook(ANT, N3=4, mode=2), tuple(range(1, 9))),
    (
        "type1-mp-mode1",
        lambda: Type1MultiPanelCodebook(AntennaConfig.standard(2, 1, Ng=2), N3=4),
        (1, 2, 3, 4),
    ),
    ("r15", lambda: R15Type2Codebook(ANT, N3=4, L=3), (1, 2)),
    ("r15-sa", lambda: R15Type2Codebook(ANT, N3=3, L=4, subband_amplitude=True), (1, 2)),
    ("r15-qpsk", lambda: R15Type2Codebook(ANT, N3=4, L=2, n_psk=4), (1, 2)),
    ("r15-ps", lambda: R15Type2Codebook(ANT, N3=4, L=4, port_selection=True, d=2), (1, 2)),
    ("r16", lambda: R16Type2Codebook(ANT, N3=8, param_combination=2), (1, 2, 3)),
    ("r16-bign3", lambda: R16Type2Codebook(ANT, N3=24, param_combination=2), (1, 2)),
    ("r16-ps", lambda: R16Type2Codebook(ANT, N3=8, param_combination=4,
                                        port_selection=True, d=2), (1, 2)),
    ("r17-m1", lambda: R17Type2Codebook(ANT, N3=8, param_combination=2), (1, 2)),
    ("r17-m2", lambda: R17Type2Codebook(ANT, N3=8, param_combination=6,
                                        N_window=4), (1, 2, 4)),
    ("r18-n4-1", lambda: R18Type2Codebook(ANT, N3=8, N4=1, param_combination=3), (1, 2)),
    ("r18-n4-2", lambda: R18Type2Codebook(ANT, N3=8, N4=2, param_combination=3), (1, 2)),
    ("r18-n4-4", lambda: R18Type2Codebook(ANT, N3=8, N4=4, param_combination=5), (1, 2, 3)),
]


@pytest.mark.parametrize("name,make,ranks", SCHEMES, ids=[s[0] for s in SCHEMES])
def test_round_trip_and_bit_count(name, make, ranks):
    cbk = make()
    n_slots = getattr(cbk, "N4", 1)
    for rank in ranks:
        for seed in (0, 1, 2):
            rng = np.random.default_rng(100 * seed + rank)
            H = random_channel(rng, n_slots, cbk.N3, P=cbk.antenna.P)
            pmi = cbk.select(H, rank=rank)
            bits = pack(cbk, pmi)
            assert set(bits) <= {"0", "1"}
            assert len(bits) == cbk.total_overhead_bits(pmi), (
                f"{name} rank {rank}: bitstream {len(bits)} != "
                f"declared {cbk.total_overhead_bits(pmi)}"
            )
            pmi2 = unpack(cbk, bits, rank)
            assert pmi_fields_equal(pmi, pmi2), f"{name} rank {rank} seed {seed}"
            # the reconstructed PMI drives the same precoder
            assert np.allclose(cbk.precoder(pmi), cbk.precoder(pmi2))


def test_unpacked_pmi_packs_to_same_bits():
    cbk = R16Type2Codebook(ANT, N3=8, param_combination=2)
    H = random_channel(np.random.default_rng(5), 1, 8)
    pmi = cbk.select(H, rank=2)
    bits = pack(cbk, pmi)
    assert pack(cbk, unpack(cbk, bits, 2)) == bits


def test_truncated_stream_rejected():
    cbk = R15Type2Codebook(ANT, N3=4, L=2)
    H = random_channel(np.random.default_rng(6), 1, 4)
    pmi = cbk.select(H, rank=1)
    bits = pack(cbk, pmi)
    with pytest.raises(ValueError):
        unpack(cbk, bits[:-3], 1)
