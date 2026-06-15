"""Direct paper-equation reconstruction and selector-optimality tests."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
    _spatial,
)
from nr_csi.codebooks.etype2_r16 import select_taps
from nr_csi.codebooks.type1 import Type1PMI, i13_offsets
from nr_csi.codebooks.type2_r15 import R15Type2PMI
from nr_csi.config import SUPPORTED_N1N2, AntennaConfig
from tests.paper.paper_oracles import (
    direct_su_rate,
    paper_combo_index,
    r15_precoder,
    r16_precoder,
    r17_precoder,
    r18_precoder,
    spatial_beam,
    type1_precoder,
)


def complex_channel(seed: int, n_slots: int, N3: int, n_rx: int, P: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_slots, N3, n_rx, P)) + 1j * rng.standard_normal(
        (n_slots, N3, n_rx, P)
    )


TYPE1_CASES = [
    (shape, mode, rank)
    for shape in sorted(SUPPORTED_N1N2)
    for mode in (1, 2)
    for rank in (1, 2)
    if mode == 1 or shape[1] > 1
]


@pytest.mark.parametrize("shape,mode,rank", TYPE1_CASES)
def test_type1_reconstruction_matches_direct_table_equation(shape, mode, rank):
    antenna = AntennaConfig.standard(*shape)
    cbk = Type1Codebook(antenna, N3=3, mode=mode)
    G1, G2 = antenna.n_beams
    i11 = min(1, G1 // mode - 1)
    i12 = min(1, max(G2 // mode, 1) - 1)
    n_i2 = 4 if mode == 1 and rank == 1 else 2 if mode == 1 else 16 // rank
    pmi = Type1PMI(
        rank=rank,
        mode=mode,
        i11=i11,
        i12=i12,
        i2=np.arange(3) % n_i2,
        i13=(len(i13_offsets(antenna.N1, antenna.N2, antenna.O1, antenna.O2)) - 1)
        if rank == 2
        else None,
    )
    assert np.allclose(cbk.precoder(pmi), type1_precoder(antenna, pmi), atol=1e-12)


R15_CASES = [
    ((2, 1), 2, 1, False, False),
    ((2, 1), 2, 2, True, True),
    ((2, 2), 2, 1, False, False),
    ((2, 2), 3, 2, True, False),
    ((4, 1), 4, 1, False, True),
    ((4, 2), 2, 2, False, True),
    ((4, 2), 3, 1, True, False),
    ((4, 2), 4, 2, True, False),
    ((8, 1), 2, 1, False, False),
    ((8, 1), 4, 2, True, True),
    ((4, 4), 3, 1, True, False),
    ((16, 1), 4, 2, False, False),
]


def make_r15_pmi(cbk: R15Type2Codebook, rank: int) -> R15Type2PMI:
    L2 = 2 * cbk.L
    pmi = R15Type2PMI(rank=rank)
    if cbk.port_selection:
        pmi.i11_ps = 1
    else:
        pmi.q1 = min(1, cbk.antenna.O1 - 1)
        pmi.q2 = min(1, cbk.antenna.O2 - 1)
        pmi.i12 = paper_combo_index(tuple(range(cbk.L)), cbk.antenna.n_ports_per_pol)
    pmi.i13 = [(2 * layer) % L2 for layer in range(rank)]
    pmi.k1 = np.tile(np.arange(L2) % 7 + 1, (rank, 1))
    pmi.k2 = np.indices((rank, cbk.N3, L2)).sum(axis=0) % 2
    pmi.c = np.indices((rank, cbk.N3, L2)).sum(axis=0) % cbk.n_psk
    for layer, star in enumerate(pmi.i13):
        pmi.k1[layer, star] = 7
        pmi.k2[layer, :, star] = 1
        pmi.c[layer, :, star] = 0
    return pmi


@pytest.mark.parametrize("shape,L,rank,sa,port_selection", R15_CASES)
def test_r15_reconstruction_matches_direct_equation(shape, L, rank, sa, port_selection):
    antenna = AntennaConfig.standard(*shape)
    cbk = R15Type2Codebook(
        antenna,
        N3=3,
        L=L,
        subband_amplitude=sa,
        port_selection=port_selection,
        d=1,
    )
    pmi = make_r15_pmi(cbk, rank)
    assert np.allclose(cbk.precoder(pmi), r15_precoder(cbk, pmi), atol=1e-12)


R16_CASES = [
    ((2, 2), 4, 1, 1, False),
    ((2, 2), 8, 2, 2, True),
    ((4, 1), 12, 3, 1, False),
    ((4, 2), 8, 4, 2, False),
    ((4, 2), 12, 5, 3, False),
    ((4, 2), 18, 6, 4, True),
    ((8, 1), 19, 4, 1, False),
    ((8, 1), 20, 4, 2, True),
    ((4, 4), 24, 4, 1, False),
    ((8, 2), 36, 5, 2, False),
]


@pytest.mark.parametrize("shape,N3,combo,rank,port_selection", R16_CASES)
def test_r16_reconstruction_matches_direct_frequency_equation(
    shape, N3, combo, rank, port_selection
):
    antenna = AntennaConfig.standard(*shape)
    R = 2 if N3 > 19 else 1
    cbk = R16Type2Codebook(
        antenna,
        N3=N3,
        param_combination=combo,
        R=R,
        port_selection=port_selection,
    )
    H = complex_channel(1000 + N3 + rank, 1, N3, max(rank, 2), antenna.P)
    pmi = cbk.select(H, rank=rank)
    assert np.allclose(cbk.precoder(pmi), r16_precoder(cbk, pmi), atol=1e-12)


R17_CASES = [
    ((2, 2), 4, 1, 1, 2),
    ((2, 2), 8, 2, 2, 4),
    ((4, 2), 8, 5, 1, 2),
    ((4, 2), 12, 6, 2, 4),
    ((4, 2), 18, 7, 3, 4),
    ((8, 1), 8, 8, 4, 4),
]


@pytest.mark.parametrize("shape,N3,combo,rank,N_window", R17_CASES)
def test_r17_reconstruction_matches_direct_free_port_equation(
    shape, N3, combo, rank, N_window
):
    antenna = AntennaConfig.standard(*shape)
    cbk = R17Type2Codebook(
        antenna, N3=N3, param_combination=combo, N_window=N_window
    )
    H = complex_channel(2000 + N3 + rank, 1, N3, max(rank, 2), antenna.P)
    pmi = cbk.select(H, rank=rank)
    assert np.allclose(cbk.precoder(pmi), r17_precoder(cbk, pmi), atol=1e-12)


R18_CASES = [
    ((2, 2), 8, 1, 2, 1),
    ((2, 2), 8, 2, 2, 2),
    ((4, 1), 12, 4, 3, 1),
    ((4, 2), 8, 8, 4, 2),
    ((4, 2), 12, 4, 5, 3),
    ((4, 2), 18, 2, 6, 4),
    ((8, 1), 19, 4, 3, 1),
    ((8, 1), 20, 4, 3, 2),
    ((4, 4), 24, 8, 5, 1),
]


@pytest.mark.parametrize("shape,N3,N4,combo,rank", R18_CASES)
def test_r18_reconstruction_matches_direct_tucker_sum(shape, N3, N4, combo, rank):
    antenna = AntennaConfig.standard(*shape)
    R = 2 if N3 > 19 else 1
    cbk = R18Type2Codebook(
        antenna, N3=N3, N4=N4, param_combination=combo, R=R
    )
    H = complex_channel(3000 + N3 + N4 + rank, N4, N3, max(rank, 2), antenna.P)
    pmi = cbk.select(H, rank=rank)
    assert np.allclose(cbk.precoder(pmi), r18_precoder(cbk, pmi), atol=1e-12)


@pytest.mark.parametrize(
    "shape,L",
    [
        ((2, 1), 1),
        ((2, 2), 2),
        ((4, 1), 2),
        ((4, 2), 4),
        ((8, 1), 4),
        ((4, 4), 4),
        ((16, 1), 6),
    ],
)
def test_group_and_beam_selector_matches_brute_force_energy(shape, L):
    antenna = AntennaConfig.standard(*shape)
    rng = np.random.default_rng(4000 + L + antenna.P)
    targets = rng.standard_normal((5, antenna.P, 2)) + 1j * rng.standard_normal(
        (5, antenna.P, 2)
    )
    expected = None
    for q1 in range(antenna.O1):
        for q2 in range(antenna.O2):
            basis = np.stack(
                [
                    spatial_beam(
                        antenna,
                        antenna.O1 * (n % antenna.N1) + q1,
                        antenna.O2 * (n // antenna.N1) + q2,
                    )
                    for n in range(antenna.n_ports_per_pol)
                ]
            )
            split = np.stack(
                [targets[:, : antenna.P // 2], targets[:, antenna.P // 2 :]],
                axis=1,
            )
            energy = np.sum(
                np.abs(np.einsum("bp,tspv->btsv", basis.conj(), split)) ** 2,
                axis=(1, 2, 3),
            )
            selected = tuple(np.argsort(energy)[::-1][:L].tolist())
            score = float(np.sum(energy[list(selected)]))
            candidate = (score, q1, q2, selected)
            if expected is None or score > expected[0]:
                expected = candidate
    q1, q2, i12 = _spatial.select_group_and_beams(antenna, targets, L)
    actual_flat = tuple(paper_combo_from_index(i12, antenna.n_ports_per_pol, L))
    assert (q1, q2) == expected[1:3]
    assert actual_flat == tuple(sorted(expected[3]))


def paper_combo_from_index(index: int, n_total: int, k: int) -> list[int]:
    for values in itertools.combinations(range(n_total), k):
        if paper_combo_index(values, n_total) == index:
            return list(values)
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    "shape,L,d",
    [
        ((2, 1), 2, 1),
        ((2, 2), 2, 1),
        ((2, 2), 4, 2),
        ((4, 1), 4, 1),
        ((4, 2), 2, 2),
        ((4, 2), 4, 4),
        ((8, 1), 4, 2),
    ],
)
def test_port_window_selector_matches_brute_force(shape, L, d):
    antenna = AntennaConfig.standard(*shape)
    rng = np.random.default_rng(5000 + antenna.P + L + d)
    targets = rng.standard_normal((3, antenna.P, 2)) + 1j * rng.standard_normal(
        (3, antenna.P, 2)
    )
    half = antenna.P // 2
    port_energy = np.sum(np.abs(targets[:, :half]) ** 2, axis=(0, 2))
    port_energy += np.sum(np.abs(targets[:, half:]) ** 2, axis=(0, 2))
    scores = [
        sum(port_energy[(initial * d + i) % half] for i in range(L))
        for initial in range(int(np.ceil(half / d)))
    ]
    assert _spatial.select_ps_initial(antenna, targets, L, d) == int(np.argmax(scores))


TAP_SELECTION_CASES = [
    (8, 1, None),
    (8, 2, None),
    (8, 4, None),
    (19, 3, None),
    (20, 2, None),
    (20, 4, None),
    (36, 3, None),
    (36, 5, None),
    (36, 3, -2),
    (36, 5, -4),
]


@pytest.mark.parametrize("N3,Mv,m_initial", TAP_SELECTION_CASES)
def test_tap_selector_matches_exhaustive_constrained_optimum(N3, Mv, m_initial):
    rng = np.random.default_rng(6000 + N3 + Mv + (m_initial or 0))
    energy = rng.random(N3)
    energy[0] = 2.0
    actual = select_taps(energy, Mv, N3, m_initial)
    if N3 <= 19:
        candidates = itertools.combinations(range(1, N3), Mv - 1)
    else:
        initials = range(0, -2 * Mv, -1) if m_initial is None else [m_initial]
        candidate_sets = set()
        for initial in initials:
            window = [(initial + j) % N3 for j in range(2 * Mv)]
            candidate_sets.update(
                itertools.combinations([n for n in window if n != 0], Mv - 1)
            )
        candidates = candidate_sets
    expected = max(
        ([0, *rest] for rest in candidates),
        key=lambda taps: sum(energy[n] for n in taps),
    )
    assert set(actual) == set(expected)


@pytest.mark.parametrize("mode,rank", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_type1_selector_matches_independent_exhaustive_rate_search(mode, rank):
    antenna = AntennaConfig.standard(2, 2)
    cbk = Type1Codebook(antenna, N3=2, mode=mode, selection_snr_db=7.0)
    H = complex_channel(7000 + 10 * mode + rank, 1, 2, 2, antenna.P)
    selected = cbk.select(H, rank=rank)
    G1, G2 = antenna.n_beams
    wideband = (
        itertools.product(range(G1), range(G2))
        if mode == 1
        else itertools.product(range(G1 // 2), range(G2 // 2))
    )
    i13_values = range(len(i13_offsets(2, 2, 4, 4))) if rank == 2 else [None]
    best = None
    for i11, i12 in wideband:
        for i13 in i13_values:
            choices = []
            for t in range(2):
                per_subband = []
                for i2 in range(cbk._n_i2(rank)):
                    pmi = Type1PMI(
                        rank=rank,
                        mode=mode,
                        i11=i11,
                        i12=i12,
                        i2=np.array([i2]),
                        i13=i13,
                    )
                    W = type1_precoder(antenna, pmi)[0, 0]
                    rate = direct_su_rate(H[0, t : t + 1], W[None], cbk.selection_rho)
                    per_subband.append(rate)
                choices.append((max(per_subband), int(np.argmax(per_subband))))
            metric = sum(value for value, _ in choices)
            candidate = (metric, i11, i12, i13, [choice for _, choice in choices])
            if best is None or metric > best[0]:
                best = candidate
    assert (selected.i11, selected.i12, selected.i13, selected.i2.tolist()) == (
        best[1],
        best[2],
        best[3],
        best[4],
    )


@pytest.mark.parametrize("family", ["type1", "r15", "r16", "r17", "r18"])
def test_selection_is_invariant_to_global_channel_scale_and_phase(family):
    antenna = AntennaConfig.standard(4, 2)
    if family == "type1":
        cbk, slots, N3, rank = Type1Codebook(antenna, N3=3), 1, 3, 2
    elif family == "r15":
        cbk, slots, N3, rank = R15Type2Codebook(antenna, N3=3, L=4), 1, 3, 2
    elif family == "r16":
        cbk, slots, N3, rank = R16Type2Codebook(antenna, N3=8), 1, 8, 2
    elif family == "r17":
        cbk, slots, N3, rank = R17Type2Codebook(antenna, N3=8), 1, 8, 2
    else:
        cbk, slots, N3, rank = R18Type2Codebook(antenna, N3=8, N4=4), 4, 8, 2
    H = complex_channel(8000 + len(family), slots, N3, rank, antenna.P)
    W1 = cbk.precoder(cbk.select(H, rank=rank))
    W2 = cbk.precoder(cbk.select(3.7 * np.exp(0.63j) * H, rank=rank))
    assert np.allclose(W1, W2, atol=1e-11)
