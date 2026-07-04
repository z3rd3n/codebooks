"""Deep seeded algorithmic sweeps across the paper's supported domains."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from nr_csi.channel import RandomRayChannel
from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
)
from nr_csi.codebooks.serialize import pack, unpack
from nr_csi.config import (
    R16_PARAM_COMBOS,
    R17_PARAM_COMBOS,
    R18_PARAM_COMBOS,
    SUPPORTED_N1N2,
    AntennaConfig,
)
from nr_csi.eval.f1 import run_f1_case
from nr_csi.metrics.similarity import sgcs
from nr_csi.utils import dft

pytestmark = pytest.mark.slow


def gaussian_channel(seed: int, slots: int, N3: int, n_rx: int, P: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((slots, N3, n_rx, P)) + 1j * rng.standard_normal(
        (slots, N3, n_rx, P)
    )


def pmi_equal(left, right) -> bool:
    for name, value in vars(left).items():
        other = getattr(right, name)
        if isinstance(value, np.ndarray):
            if not np.array_equal(value, other):
                return False
        elif value != other:
            return False
    return True


def assert_valid_sample(cbk, H: np.ndarray, rank: int) -> None:
    pmi = cbk.select(H, rank=rank)
    W = cbk.precoder(pmi)
    assert np.isfinite(W).all()
    norms = np.linalg.norm(W, axis=-2)
    assert np.allclose(norms, 1 / np.sqrt(rank), atol=1e-10)
    bits = pack(cbk, pmi)
    assert len(bits) == cbk.total_overhead_bits(pmi)
    restored = unpack(cbk, bits, rank)
    assert pmi_equal(pmi, restored)
    assert np.allclose(W, cbk.precoder(restored), atol=1e-12)
    bitmap = getattr(pmi, "i17", None)
    if bitmap is not None:
        per_layer = bitmap.reshape(rank, -1).sum(axis=1)
        assert np.all(per_layer >= 1)
        assert np.all(per_layer <= cbk.K0)
        assert int(bitmap.sum()) <= 2 * cbk.K0


@pytest.mark.parametrize("seed", range(32))
def test_all_seeded_gaussian_channels_across_release_families(seed):
    antenna = AntennaConfig.standard(4, 2)
    rank = 1 + seed % 2

    type1 = Type1Codebook(antenna, N3=4, mode=1 + seed % 2)
    assert_valid_sample(
        type1,
        gaussian_channel(10000 + seed, 1, 4, rank, antenna.P),
        rank,
    )

    r15 = R15Type2Codebook(
        antenna,
        N3=4,
        L=(2, 3, 4)[seed % 3],
        n_psk=(4, 8)[seed % 2],
        subband_amplitude=bool(seed % 2),
        port_selection=bool((seed // 2) % 2),
    )
    assert_valid_sample(
        r15,
        gaussian_channel(11000 + seed, 1, 4, rank, antenna.P),
        rank,
    )

    r16_combo = 1 + seed % 6
    r16 = R16Type2Codebook(
        antenna,
        N3=12,
        param_combination=r16_combo,
        port_selection=bool(seed % 2),
    )
    assert_valid_sample(
        r16,
        gaussian_channel(12000 + seed, 1, 12, rank, antenna.P),
        rank,
    )

    r17 = R17Type2Codebook(
        antenna,
        N3=8,
        param_combination=1 + seed % 8,
        N_window=(2, 4)[seed % 2],
    )
    assert_valid_sample(
        r17,
        gaussian_channel(13000 + seed, 1, 8, rank, antenna.P),
        rank,
    )

    N4 = (1, 2, 4, 8)[seed % 4]
    r18 = R18Type2Codebook(
        antenna,
        N3=8,
        N4=N4,
        param_combination=1 + seed % 7,
    )
    assert_valid_sample(
        r18,
        gaussian_channel(14000 + seed, N4, 8, rank, antenna.P),
        rank,
    )


@pytest.mark.parametrize("seed", range(32))
def test_all_seeded_sparse_ray_channels_across_release_families(seed):
    antenna = AntennaConfig.standard(4, 2)
    rank = 1 + seed % 2
    N4 = (1, 2, 4, 8)[seed % 4]
    source = RandomRayChannel(
        antenna,
        N3=8,
        n_rx=rank,
        n_paths=6,
        max_delay=4,
        max_doppler=max(N4 - 1, 0),
        doppler_period=N4,
    )
    H = source.generate(n_slots=N4, rng=np.random.default_rng(20000 + seed))

    assert_valid_sample(Type1Codebook(antenna, N3=8), H[-1:], rank)
    assert_valid_sample(
        R15Type2Codebook(
            antenna,
            N3=8,
            L=(2, 3, 4)[seed % 3],
            subband_amplitude=bool(seed % 2),
        ),
        H[-1:],
        rank,
    )
    assert_valid_sample(
        R16Type2Codebook(antenna, N3=8, param_combination=1 + seed % 6),
        H[-1:],
        rank,
    )
    assert_valid_sample(
        R17Type2Codebook(antenna, N3=8, param_combination=1 + seed % 8),
        H[-1:],
        rank,
    )
    assert_valid_sample(
        R18Type2Codebook(
            antenna, N3=8, N4=N4, param_combination=1 + seed % 7
        ),
        H,
        rank,
    )


ANTENNA_CASES = [
    (shape, mode, rank)
    for shape in sorted(SUPPORTED_N1N2)
    for mode in (1, 2)
    for rank in (1, 2)
    if mode == 1 or shape[1] > 1
]


@pytest.mark.parametrize("shape,mode,rank", ANTENNA_CASES)
def test_every_supported_antenna_row_in_type1_and_r15(shape, mode, rank):
    antenna = AntennaConfig.standard(*shape)
    H = gaussian_channel(30000 + antenna.P + 10 * mode + rank, 1, 3, rank, antenna.P)
    assert_valid_sample(Type1Codebook(antenna, N3=3, mode=mode), H, rank)
    for L in (2, 3, 4):
        if L <= antenna.n_ports_per_pol:
            assert_valid_sample(R15Type2Codebook(antenna, N3=3, L=L), H, rank)


R16_MATRIX = [
    (combo, rank, N3)
    for combo, row in R16_PARAM_COMBOS.items()
    for rank in range(1, 5)
    if rank <= 2 or row.p_v34 is not None
    for N3 in (19, 20)
]


@pytest.mark.parametrize("combo,rank,N3", R16_MATRIX)
def test_every_r16_parameter_row_rank_and_n3_boundary(combo, rank, N3):
    antenna = AntennaConfig.standard(4, 4)
    # 5.2.2.2.5: combinations 7/8 require ranks 3-4 disallowed.
    ri = [1, 1, 0, 0] if combo in (7, 8) else None
    cbk = R16Type2Codebook(antenna, N3=N3, param_combination=combo, ri_restriction=ri)
    H = gaussian_channel(40000 + 100 * combo + 10 * rank + N3, 1, N3, rank, antenna.P)
    assert_valid_sample(cbk, H, rank)


R17_MATRIX = list(itertools.product(sorted(R17_PARAM_COMBOS), range(1, 5)))


@pytest.mark.parametrize("combo,rank", R17_MATRIX)
def test_every_r17_parameter_row_and_rank(combo, rank):
    # 5.2.2.2.7 bars combinations 7/8 at P_CSI-RS = 32, so those rows are
    # exercised on the 16-port array instead.
    antenna = (
        AntennaConfig.standard(4, 2) if combo in (7, 8) else AntennaConfig.standard(4, 4)
    )
    cbk = R17Type2Codebook(antenna, N3=8, param_combination=combo)
    H = gaussian_channel(50000 + 100 * combo + rank, 1, 8, rank, antenna.P)
    assert_valid_sample(cbk, H, rank)


R18_MATRIX = [
    (combo, rank, N4, N3)
    for combo, row in R18_PARAM_COMBOS.items()
    for rank in range(1, 5)
    if rank <= 2 or row.p_v34 is not None
    for N4 in (1, 2, 4, 8)
    for N3 in (19, 20)
]


@pytest.mark.parametrize("combo,rank,N4,N3", R18_MATRIX)
def test_every_r18_parameter_row_rank_time_size_and_n3_boundary(
    combo, rank, N4, N3
):
    antenna = AntennaConfig.standard(4, 4)
    # 5.2.2.2.10: combinations 8/9 require ranks 3-4 disallowed.
    ri = [1, 1, 0, 0] if combo in (8, 9) else None
    cbk = R18Type2Codebook(
        antenna, N3=N3, N4=N4, param_combination=combo, ri_restriction=ri
    )
    H = gaussian_channel(
        60000 + 1000 * combo + 100 * rank + 10 * N4 + N3,
        N4,
        N3,
        rank,
        antenna.P,
    )
    assert_valid_sample(cbk, H, rank)


@pytest.mark.parametrize("family", ["type1", "r15", "r16", "r17", "r18"])
def test_receive_unitary_transform_does_not_change_selected_precoder(family):
    antenna = AntennaConfig.standard(4, 2)
    rank, N3, N4 = 2, 8, 4
    H = gaussian_channel(70000 + len(family), N4, N3, rank, antenna.P)
    rng = np.random.default_rng(71000 + len(family))
    unitary, _ = np.linalg.qr(
        rng.standard_normal((rank, rank)) + 1j * rng.standard_normal((rank, rank))
    )
    transformed = np.einsum("ab,stbp->stap", unitary, H)
    if family == "type1":
        cbk, H1, H2 = Type1Codebook(antenna, N3=N3), H[-1:], transformed[-1:]
    elif family == "r15":
        cbk = R15Type2Codebook(antenna, N3=N3, L=4)
        H1, H2 = H[-1:], transformed[-1:]
    elif family == "r16":
        cbk = R16Type2Codebook(antenna, N3=N3)
        H1, H2 = H[-1:], transformed[-1:]
    elif family == "r17":
        cbk = R17Type2Codebook(antenna, N3=N3)
        H1, H2 = H[-1:], transformed[-1:]
    else:
        cbk, H1, H2 = R18Type2Codebook(antenna, N3=N3, N4=N4), H, transformed
    W1 = cbk.precoder(cbk.select(H1, rank=rank))
    W2 = cbk.precoder(cbk.select(H2, rank=rank))
    assert np.allclose(W1, W2, atol=1e-10)


@pytest.mark.parametrize("combo,rank,N3", [(2, 1, 19), (2, 2, 20), (3, 1, 20), (3, 2, 19)])
def test_r18_n4_one_degenerates_to_r16_across_boundaries(combo, rank, N3):
    antenna = AntennaConfig.standard(4, 2)
    H = gaussian_channel(80000 + 100 * combo + 10 * rank + N3, 1, N3, rank, antenna.P)
    r16 = R16Type2Codebook(antenna, N3=N3, param_combination=combo)
    r18 = R18Type2Codebook(
        antenna, N3=N3, N4=1, param_combination=combo
    )
    W16 = r16.precoder(r16.select(H, rank=rank))
    W18 = r18.precoder(r18.select(H, rank=rank))
    assert np.allclose(W16, W18, atol=1e-12)


@pytest.mark.parametrize("seed", range(16))
def test_r17_m1_matches_r16_port_selection_on_flat_channels(seed):
    antenna = AntennaConfig.standard(2, 2)
    rng = np.random.default_rng(90000 + seed)
    h = rng.standard_normal(antenna.P) + 1j * rng.standard_normal(antenna.P)
    H = np.tile(h.conj()[None, None, None], (1, 4, 1, 1))
    r16 = R16Type2Codebook(
        antenna, N3=4, param_combination=4, port_selection=True
    )
    r17 = R17Type2Codebook(antenna, N3=4, param_combination=2)
    W16 = r16.precoder(r16.select(H, rank=1))
    W17 = r17.precoder(r17.select(H, rank=1))
    assert sgcs(W16, W17) > 1 - 1e-12


@pytest.mark.parametrize("q1", range(4))
@pytest.mark.parametrize("seed", range(8))
def test_regular_and_port_selection_match_through_unitary_peb(q1, seed):
    antenna = AntennaConfig.standard(8, 1)
    N3, L, half = 8, 4, antenna.P // 2
    rng = np.random.default_rng(100000 + 100 * q1 + seed)
    beams = dft.orthogonal_group(antenna, q1, 0)[:L]
    physical = np.zeros((N3, antenna.P), dtype=complex)
    for i, beam in enumerate(beams):
        gains = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        delay = np.exp(2j * np.pi * np.arange(N3) * i / N3)
        physical[:, :half] += gains[0] * np.outer(delay, beam)
        physical[:, half:] += gains[1] * np.outer(delay, beam)
    H = physical.conj()[None, :, None, :]
    F = dft.unitary_peb(antenna, q1, 0)
    H_eff = np.concatenate([H[..., :half] @ F, H[..., half:] @ F], axis=-1)
    regular = R16Type2Codebook(antenna, N3=N3, param_combination=4)
    port = R16Type2Codebook(
        antenna, N3=N3, param_combination=4, port_selection=True
    )
    W_regular = regular.precoder(regular.select(H, rank=1))
    W_port = port.precoder(port.select(H_eff, rank=1))
    mapped = np.concatenate(
        [
            np.einsum("ab,stbv->stav", F, W_port[..., :half, :]),
            np.einsum("ab,stbv->stav", F, W_port[..., half:, :]),
        ],
        axis=-2,
    )
    assert sgcs(W_regular, mapped) > 1 - 1e-12


@pytest.mark.parametrize("N,streams", [(4, 1), (16, 1), (16, 2)])
@pytest.mark.parametrize("n_paths", [2, 4, 8])
def test_f1_multi_seed_aggregate_ordering_and_monotonic_snr(N, streams, n_paths):
    snr = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
    curves = [
        run_f1_case(
            N,
            streams,
            snr,
            n_drops=100,
            n_paths=n_paths,
            seed=seed,
            t1_second_beam="free",
        )
        for seed in range(8)
    ]
    upper = np.mean([curve.upper_bound for curve in curves], axis=0)
    type1 = np.mean([curve.type1 for curve in curves], axis=0)
    type2 = np.mean([curve.type2 for curve in curves], axis=0)
    assert np.all(np.diff(upper) > 0)
    assert np.all(np.diff(type1) > 0)
    assert np.all(np.diff(type2) > 0)
    assert np.all(upper >= type2 - 1e-10)
    assert np.all(type2 >= type1 - 0.08)
