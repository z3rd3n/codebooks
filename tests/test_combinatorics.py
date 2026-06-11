"""Round-trip anchors for the combinatorial codecs behind Algorithms 1-4."""

import itertools
from math import comb

import numpy as np
import pytest

from nr_csi.utils import combinatorics as cb


def test_index_codec_exhaustive_small():
    """The codec is a bijection onto [0, C(n,k)) matching the spec formula
    index = sum_i C(n_total-1-n_i, k-i); note this gives index 0 to the
    lexicographically *largest* combination ({n-k..n-1}), not the smallest."""
    for n_total, k in [(4, 2), (6, 3), (8, 4), (16, 4), (5, 1)]:
        seen = set()
        for combo in itertools.combinations(range(n_total), k):
            idx = cb.combo_to_index(list(combo), n_total)
            assert idx == sum(comb(n_total - 1 - n, k - i) for i, n in enumerate(combo))
            assert cb.index_to_combo(idx, n_total, k) == list(combo)
            seen.add(idx)
        assert seen == set(range(comb(n_total, k)))
        assert cb.combo_to_index(list(range(n_total - k, n_total)), n_total) == 0
        assert cb.combo_to_index(list(range(k)), n_total) == comb(n_total, k) - 1


def test_index_codec_random_large():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_total = int(rng.integers(8, 40))
        k = int(rng.integers(1, min(n_total, 8)))
        combo = sorted(rng.choice(n_total, size=k, replace=False).tolist())
        idx = cb.combo_to_index(combo, n_total)
        assert 0 <= idx < comb(n_total, k)
        assert cb.index_to_combo(idx, n_total, k) == combo


def test_index_codec_rejects_bad_input():
    with pytest.raises(ValueError):
        cb.combo_to_index([0, 0], 4)
    with pytest.raises(ValueError):
        cb.combo_to_index([0, 4], 4)
    with pytest.raises(ValueError):
        cb.index_to_combo(comb(6, 3), 6, 3)


def test_algorithm1_beam_combination_roundtrip():
    """Algorithm 1: i_{1,2} <-> {n1^(i), n2^(i)} with n = N1*n2 + n1."""
    N1, N2, L = 4, 2, 4
    rng = np.random.default_rng(1)
    for _ in range(50):
        flat = sorted(rng.choice(N1 * N2, size=L, replace=False).tolist())
        n1 = [n % N1 for n in flat]
        n2 = [n // N1 for n in flat]
        i12 = cb.encode_beam_combination(n1, n2, N1, N2)
        assert 0 <= i12 < comb(N1 * N2, L)
        d1, d2 = cb.decode_beam_combination(i12, N1, N2, L)
        assert (d1, d2) == (n1, n2)


def test_algorithm2_restriction_groups_roundtrip():
    """Algorithm 2: beta_1 <-> 4 group indices over O1*O2=16 (g = O1*r2 + r1)."""
    O1, O2 = 4, 4
    for g_combo in itertools.combinations(range(O1 * O2), 4):
        r1 = [g % O1 for g in g_combo]
        r2 = [g // O1 for g in g_combo]
        beta1 = cb.encode_restriction_groups(r1, r2, O1, O2)
        assert beta1 < 2**11  # B1 is an 11-bit sequence in the spec
        g, d1, d2 = cb.decode_restriction_groups(beta1, O1, O2)
        assert g == list(g_combo)
        assert (d1, d2) == (r1, r2)


def test_algorithm3_taps_small_n3():
    """N3 <= 19: M_v-1 taps encoded out of N3-1 (strongest tap 0 implicit)."""
    N3, Mv = 18, 5
    rng = np.random.default_rng(2)
    for _ in range(100):
        rest = sorted(rng.choice(np.arange(1, N3), size=Mv - 1, replace=False).tolist())
        taps = [0] + rest
        i16, i15 = cb.encode_taps(taps, N3, Mv)
        assert i15 is None
        assert 0 <= i16 < comb(N3 - 1, Mv - 1)
        assert cb.decode_taps(i16, N3, Mv, i15) == taps


def test_algorithm3_taps_two_level_n3_large():
    """N3 > 19: taps confined to a cyclic 2*Mv window indicated by i_{1,5}."""
    N3, Mv = 36, 5
    # Window wholly non-negative (M_initial = 0)
    taps = [0, 1, 4, 7, 9]
    i16, i15 = cb.encode_taps(taps, N3, Mv)
    assert i15 == 0
    assert cb.decode_taps(i16, N3, Mv, i15) == taps
    # Window wrapping below zero: taps near N3-1 represent negative delays
    taps = [0, 2, 33, 34, 35]  # = {-3,-2,-1,0,2} mod 36
    i16, i15 = cb.encode_taps(taps, N3, Mv)
    assert i15 != 0
    m_initial = i15 - 2 * Mv
    assert -2 * Mv + 1 <= m_initial < 0
    assert cb.decode_taps(i16, N3, Mv, i15) == taps
    # Spread that fits in no 2*Mv=10 cyclic window must be rejected
    with pytest.raises(ValueError):
        cb.encode_taps([0, 5, 12, 20, 28], N3, Mv)


def test_algorithm3_taps_two_level_random_roundtrip():
    N3, Mv = 36, 5
    rng = np.random.default_rng(3)
    count = 0
    while count < 100:
        m_init = int(rng.integers(-2 * Mv + 1, 1))
        window = [(m_init + j) % N3 for j in range(2 * Mv)]
        rest = rng.choice([w for w in window if w != 0], size=Mv - 1, replace=False)
        taps = sorted([0] + rest.tolist())
        i16, i15 = cb.encode_taps(taps, N3, Mv)
        assert cb.decode_taps(i16, N3, Mv, i15) == taps
        count += 1


def test_algorithm3_single_tap():
    assert cb.decode_taps(0, 18, 1) == [0]


def test_algorithm4_ports_roundtrip():
    """Algorithm 4 (errata-corrected): i_{1,2} <-> L free ports out of P/2."""
    P, L = 32, 8
    rng = np.random.default_rng(4)
    for _ in range(50):
        m = sorted(rng.choice(P // 2, size=L, replace=False).tolist())
        i12 = cb.encode_ports(m, P)
        assert 0 <= i12 < comb(P // 2, L)
        assert cb.decode_ports(i12, P, L) == m
