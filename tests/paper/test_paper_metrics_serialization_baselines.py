"""Paper-level wire-format, metric, baseline, and channel equation tests."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from nr_csi.baselines.ideal import (
    bd,
    eigen_precoder,
    ezf,
    gmd,
    harmonic_mean_allocation,
    mmse,
    mrt,
    rzf,
    svd_precoder,
    water_filling,
    wmmse,
    zf,
)
from nr_csi.channel import RandomRayChannel, Ray, SyntheticRayChannel
from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
)
from nr_csi.codebooks.compact import compact_r15, compact_r16, compact_r18, tucker_r18
from nr_csi.codebooks.etype2_r16 import R16Type2PMI
from nr_csi.codebooks.etype2_r18 import R18Type2PMI
from nr_csi.codebooks.fetype2_r17 import R17Type2PMI
from nr_csi.codebooks.serialize import BitReader, BitWriter, pack, unpack
from nr_csi.codebooks.type1 import Type1PMI
from nr_csi.codebooks.type2_r15 import R15Type2PMI
from nr_csi.config import AntennaConfig
from nr_csi.metrics.se import mu_rate, su_rate
from tests.paper.paper_oracles import direct_su_rate, spatial_beam


def random_complex(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("width", range(9))
def test_bit_writer_reader_exhaustive_unsigned_domain(width):
    for value in range(1 << width):
        writer = BitWriter()
        writer.write(value, width)
        expected = "" if width == 0 else format(value, f"0{width}b")
        assert writer.getvalue() == expected
        reader = BitReader(expected)
        assert reader.read(width) == value
        assert reader.done()


@pytest.mark.parametrize("width", range(8))
def test_bit_writer_rejects_values_outside_field(width):
    writer = BitWriter()
    with pytest.raises(ValueError):
        writer.write(1 << width, width)
    with pytest.raises(ValueError):
        writer.write(-1, width)


def test_type1_mode1_golden_bitstream():
    antenna = AntennaConfig.standard(2, 1)
    cbk = Type1Codebook(antenna, N3=2)
    pmi = Type1PMI(rank=1, mode=1, i11=5, i12=0, i2=np.array([1, 3]))
    assert pack(cbk, pmi) == "1010111"
    assert vars(unpack(cbk, "1010111", 1))["i11"] == 5


def test_type1_mode2_rank2_golden_bitstream():
    antenna = AntennaConfig.standard(2, 2)
    cbk = Type1Codebook(antenna, N3=2, mode=2)
    pmi = Type1PMI(
        rank=2, mode=2, i11=2, i12=1, i13=3, i2=np.array([7, 0])
    )
    assert pack(cbk, pmi) == "100111111000"


def test_r15_regular_golden_bitstream_with_omitted_zero_phase():
    antenna = AntennaConfig.standard(2, 1)
    cbk = R15Type2Codebook(antenna, N3=1, L=2, n_psk=8)
    pmi = R15Type2PMI(
        rank=1,
        q1=2,
        q2=0,
        i12=0,
        i13=[0],
        k1=np.array([[7, 1, 0, 2]]),
        k2=np.ones((1, 1, 4), dtype=int),
        c=np.array([[[0, 3, 0, 5]]]),
    )
    assert pack(cbk, pmi) == "1000001000010011101"


def test_r16_regular_golden_bitstream():
    antenna = AntennaConfig.standard(2, 1)
    cbk = R16Type2Codebook(antenna, N3=4, param_combination=1)
    pmi = R16Type2PMI(
        rank=1,
        q1=1,
        q2=0,
        i12=0,
        i16=[0],
        i17=np.array([[[True, True, False, False]]]),
        i18=[0],
        k1=np.array([[15, 5]]),
        k2=np.array([[[7, 4, 0, 0]]]),
        c=np.array([[[0, 6, 0, 0]]]),
    )
    # i11 "01", i12/i16 zero-width, i17 "1100", i18 "0" (rank 1: ceil(log2
    # K^NZ) = 1 bit, TS 38.212 Table 6.3.2.1.2-1A), k1 "0101", k2/c "1000110"
    assert pack(cbk, pmi) == "011100001011000110"


def test_r17_conditional_fields_golden_bitstream():
    antenna = AntennaConfig.standard(4, 2)
    cbk = R17Type2Codebook(antenna, N3=8, param_combination=5, N_window=4)
    bitmap = np.zeros((1, 2, 8), dtype=bool)
    bitmap[0, 0, [0, 3]] = True
    pmi = R17Type2PMI(
        rank=1,
        i12=69,
        i16=2,
        i17=bitmap,
        i18=[0],
        k1=np.array([[15, 9]]),
        k2=np.zeros((1, 2, 8), dtype=int),
        c=np.zeros((1, 2, 8), dtype=int),
    )
    pmi.k2[0, 0, 0] = 7
    pmi.k2[0, 0, 3] = 5
    pmi.c[0, 0, 3] = 11
    expected = "1000101" + "10" + "1001000000000000" + "0000" + "1001" + "1011011"
    assert pack(cbk, pmi) == expected


def test_r18_n4_two_zero_width_shift_golden_bitstream():
    antenna = AntennaConfig.standard(2, 1)
    cbk = R18Type2Codebook(antenna, N3=4, N4=2, param_combination=1)
    bitmap = np.zeros((1, 2, 1, 4), dtype=bool)
    bitmap[0, 0, 0, [0, 1]] = True
    pmi = R18Type2PMI(
        rank=1,
        q1=3,
        q2=0,
        i12=0,
        i16=[0],
        i110=[0],
        i17=bitmap,
        i18=[0],
        k1=np.array([[15, 4]]),
        k2=np.zeros((1, 2, 1, 4), dtype=int),
        c=np.zeros((1, 2, 1, 4), dtype=int),
    )
    pmi.k2[0, 0, 0, 0] = 7
    pmi.k2[0, 0, 0, 1] = 2
    pmi.c[0, 0, 0, 1] = 13
    # rank-1 i18 is ceil(log2 K^NZ) = 1 bit (TS 38.212 Table 6.3.2.1.2-1C)
    expected = "11" + "11000000" + "0" + "0100" + "0101101"
    assert pack(cbk, pmi) == expected
    assert len(expected) == cbk.total_overhead_bits(pmi)


@pytest.mark.parametrize("suffix", ["0", "1", "101"])
def test_unpack_rejects_trailing_bits_as_malformed_stream(suffix):
    antenna = AntennaConfig.standard(2, 1)
    cbk = Type1Codebook(antenna, N3=1)
    pmi = Type1PMI(rank=1, mode=1, i11=1, i12=0, i2=np.array([2]))
    with pytest.raises(ValueError):
        unpack(cbk, pack(cbk, pmi) + suffix, rank=1)


SU_RATE_CASES = [
    (leading, Nr, P, rank, rho)
    for leading in ((1,), (3,), (2, 4))
    for Nr, P, rank in ((1, 2, 1), (2, 4, 1), (2, 4, 2), (3, 6, 2))
    for rho in (0.0, 0.3, 10.0)
]


@pytest.mark.parametrize("leading,Nr,P,rank,rho", SU_RATE_CASES)
def test_su_rate_matches_receive_covariance_logdet(leading, Nr, P, rank, rho):
    rng = np.random.default_rng(sum(leading) + 10 * Nr + P + rank + int(rho * 10))
    H = random_complex(rng, (*leading, Nr, P))
    W = random_complex(rng, (*leading, P, rank))
    W /= np.linalg.norm(W, axis=(-2, -1), keepdims=True)
    assert np.isclose(su_rate(H, W, rho), direct_su_rate(H, W, rho), atol=1e-12)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("rho", [0.1, 1.0, 20.0])
def test_su_rate_is_invariant_to_right_unitary_stream_rotation(rank, rho):
    rng = np.random.default_rng(100 + rank + int(rho))
    H = random_complex(rng, (4, rank + 1, 6))
    W = random_complex(rng, (4, 6, rank))
    unitary, _ = np.linalg.qr(random_complex(rng, (rank, rank)))
    assert np.isclose(su_rate(H, W, rho), su_rate(H, W @ unitary, rho), atol=1e-11)


@pytest.mark.parametrize("K,Nr,P,rank", [(1, 1, 2, 1), (2, 1, 4, 1), (3, 2, 6, 1), (2, 2, 6, 2)])
def test_mu_rate_matches_direct_signal_interference_covariance(K, Nr, P, rank):
    rng = np.random.default_rng(200 + K + Nr + P + rank)
    H = random_complex(rng, (K, 3, Nr, P))
    W = random_complex(rng, (K, 3, P, rank))
    W /= np.linalg.norm(W, axis=(-2, -1), keepdims=True)
    rho = 2.5
    expected = []
    for user in range(K):
        per_slot = []
        for t in range(3):
            signal_matrix = H[user, t] @ W[user, t]
            signal = rho * signal_matrix @ signal_matrix.conj().T
            interference = np.eye(Nr, dtype=complex)
            for other in range(K):
                if other == user:
                    continue
                other_matrix = H[user, t] @ W[other, t]
                interference += rho * other_matrix @ other_matrix.conj().T
            per_slot.append(
                np.linalg.slogdet(np.eye(Nr) + np.linalg.solve(interference, signal))[1]
                / np.log(2)
            )
        expected.append(np.mean(per_slot))
    assert np.allclose(mu_rate(H, W, rho), expected, atol=1e-11)


@pytest.mark.parametrize("Nr,Nt,streams", [(1, 2, 1), (2, 4, 1), (2, 4, 2), (4, 6, 3)])
def test_svd_precoder_is_dominant_right_singular_subspace(Nr, Nt, streams):
    rng = np.random.default_rng(300 + Nr + Nt + streams)
    H = random_complex(rng, (Nr, Nt))
    W = svd_precoder(H, streams)
    _, _, Vh = np.linalg.svd(H, full_matrices=False)
    projector = W @ W.conj().T
    expected = Vh.conj().T[:, :streams] @ Vh[:streams]
    assert np.allclose(W.conj().T @ W, np.eye(streams), atol=1e-12)
    assert np.allclose(projector, expected, atol=1e-12)


@pytest.mark.parametrize("Nr,Nt", [(1, 2), (2, 4), (3, 5), (4, 8)])
def test_mrt_is_exact_channel_hermitian(Nr, Nt):
    rng = np.random.default_rng(400 + Nr + Nt)
    H = random_complex(rng, (Nr, Nt))
    assert np.array_equal(mrt(H), H.conj().T)


@pytest.mark.parametrize("users,ports", [(1, 2), (2, 4), (3, 6), (4, 8)])
def test_zf_is_right_inverse_for_full_row_rank(users, ports):
    rng = np.random.default_rng(500 + users + ports)
    H = random_complex(rng, (users, ports))
    assert np.allclose(H @ zf(H), np.eye(users), atol=1e-11)


@pytest.mark.parametrize("users,ports,xi", [(2, 4, 0.01), (2, 4, 1.0), (3, 6, 0.2), (4, 8, 3.0)])
def test_rzf_matches_regularized_normal_equation(users, ports, xi):
    rng = np.random.default_rng(600 + users + ports + int(10 * xi))
    H = random_complex(rng, (users, ports))
    expected = H.conj().T @ np.linalg.inv(H @ H.conj().T + xi * np.eye(users))
    assert np.allclose(rzf(H, xi), expected)


@pytest.mark.parametrize("snr", [0.1, 1.0, 10.0, 100.0])
def test_mmse_is_rzf_with_inverse_snr_regularization(snr):
    rng = np.random.default_rng(700 + int(snr))
    H = random_complex(rng, (3, 6))
    assert np.allclose(mmse(H, snr), rzf(H, 1 / snr))


@pytest.mark.parametrize("shape,streams", [((2, 4), 1), ((2, 4), 2), ((3, 5), 2), ((4, 6), 3)])
def test_gmd_reconstructs_rank_truncation_and_equalizes_diagonal(shape, streams):
    rng = np.random.default_rng(800 + sum(shape) + streams)
    H = random_complex(rng, shape)
    Q, R, P = gmd(H, streams)
    U, singular, Vh = np.linalg.svd(H, full_matrices=False)
    truncated = U[:, :streams] @ np.diag(singular[:streams]) @ Vh[:streams]
    geometric = np.prod(singular[:streams]) ** (1 / streams)
    assert np.allclose(Q @ R @ P.conj().T, truncated, atol=1e-10)
    assert np.allclose(np.diag(R), geometric, atol=1e-10)


@pytest.mark.parametrize(
    "users,Nr,Nt,streams,xi",
    [(2, 2, 6, 1, 0.2), (3, 2, 8, 1, 1.0), (2, 3, 8, 2, 0.5)],
)
def test_regularized_ezf_matches_stacked_eigendirection_equation(
    users, Nr, Nt, streams, xi
):
    rng = np.random.default_rng(900 + users + Nr + Nt + streams)
    H = random_complex(rng, (users, Nr, Nt))
    directions = []
    for channel in H:
        _, _, Vh = np.linalg.svd(channel, full_matrices=False)
        directions.append(Vh.conj().T[:, :streams])
    V = np.concatenate(directions, axis=1)
    expected = V @ np.linalg.inv(V.conj().T @ V + xi * np.eye(users * streams))
    assert np.allclose(ezf(H, streams, xi), expected)


@pytest.mark.parametrize(
    "users,Nr,Nt,streams",
    [(2, 1, 4, 1), (2, 2, 6, 1), (3, 1, 6, 1), (2, 2, 8, 2)],
)
def test_block_diagonalization_nulls_every_other_user(users, Nr, Nt, streams):
    rng = np.random.default_rng(1000 + users + Nr + Nt + streams)
    H = random_complex(rng, (users, Nr, Nt))
    W = bd(H, streams)
    for user, other in itertools.permutations(range(users), 2):
        block = W[:, other * streams : (other + 1) * streams]
        assert np.linalg.norm(H[user] @ block) < 1e-10


def weighted_sum_rate(H: np.ndarray, W: np.ndarray, priorities: np.ndarray) -> float:
    users = H.shape[0]
    channels = H[:, None]
    precoders = np.stack([W[:, k : k + 1] for k in range(users)])[:, None]
    return float(priorities @ mu_rate(channels, precoders, rho=1.0))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wmmse_more_iterations_do_not_reduce_weighted_sum_rate(seed):
    rng = np.random.default_rng(1100 + seed)
    H = random_complex(rng, (3, 1, 6))
    priorities = np.array([1.0, 2.0, 4.0])
    W5 = wmmse(H, snr=1.0, n_iter=5, priorities=priorities, rng=np.random.default_rng(seed))
    W20 = wmmse(H, snr=1.0, n_iter=20, priorities=priorities, rng=np.random.default_rng(seed))
    assert np.isclose(np.trace(W20 @ W20.conj().T).real, 1.0)
    assert weighted_sum_rate(H, W20, priorities) >= weighted_sum_rate(H, W5, priorities) - 1e-9


WATER_CASES = [
    (gains, total)
    for gains in (
        np.array([1.0]),
        np.array([1.0, 1.0]),
        np.array([0.2, 1.0, 5.0]),
        np.array([0.1, 0.3, 2.0, 8.0]),
    )
    for total in (0.1, 1.0, 10.0)
]


@pytest.mark.parametrize("gains,total", WATER_CASES)
def test_water_filling_satisfies_kkt_conditions(gains, total):
    power = water_filling(gains, total)
    assert np.all(power >= 0)
    assert np.isclose(power.sum(), total)
    active = power > 1e-12
    water_levels = power[active] + 1 / gains[active]
    assert np.allclose(water_levels, water_levels[0])
    assert np.all(1 / gains[~active] >= water_levels[0] - 1e-12)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 8])
def test_harmonic_allocation_equalizes_amplitude_weighted_power(count):
    gains = np.geomspace(0.1, 10.0, count)
    power = harmonic_mean_allocation(gains, 3.0)
    assert np.isclose(power.sum(), 3.0)
    assert np.allclose(power * np.sqrt(gains), power[0] * np.sqrt(gains[0]))


RAY_CASES = [
    (shape, delay, doppler, phase)
    for shape in ((2, 1), (2, 2), (4, 1), (4, 2), (8, 1))
    for delay, doppler, phase in ((0.0, 0.0, 0.0), (1.0, 1.0, np.pi / 2), (1.5, 2.0, -0.7))
]


@pytest.mark.parametrize("shape,delay,doppler,phase", RAY_CASES)
def test_single_synthetic_ray_matches_closed_form(shape, delay, doppler, phase):
    antenna = AntennaConfig.standard(*shape)
    N3, N4 = 8, 4
    a_rx = np.array([1 + 0.5j, -0.2 + 0.3j])
    ray = Ray(
        gain=0.7 - 0.4j,
        m1=1.5,
        m2=0.5 if antenna.N2 > 1 else 0.0,
        delay=delay,
        doppler=doppler,
        pol_phase=phase,
        a_rx=a_rx,
    )
    H = SyntheticRayChannel(
        antenna, [ray], N3=N3, n_rx=2, doppler_period=N4
    ).generate(n_slots=N4)
    beam = spatial_beam(antenna, ray.m1, ray.m2)
    dual = np.concatenate([beam, np.exp(1j * phase) * beam])
    expected = ray.gain * np.einsum(
        "s,t,r,p->strp",
        np.exp(2j * np.pi * np.arange(N4) * doppler / N4),
        np.exp(2j * np.pi * np.arange(N3) * delay / N3),
        a_rx,
        dual.conj(),
    )
    assert np.allclose(H, expected, atol=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_random_ray_channel_is_seed_reproducible_and_power_finite(seed):
    antenna = AntennaConfig.standard(4, 2)
    channel = RandomRayChannel(
        antenna,
        N3=8,
        n_rx=2,
        n_paths=4,
        max_delay=3,
        max_doppler=2,
        doppler_period=4,
    )
    first = channel.generate(n_slots=4, rng=np.random.default_rng(seed))
    second = channel.generate(n_slots=4, rng=np.random.default_rng(seed))
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.linalg.norm(first) > 0


@pytest.mark.parametrize("seed", range(5))
def test_compact_models_match_explicit_nested_sums(seed):
    rng = np.random.default_rng(1200 + seed)
    L, half, Mv, Q, N3, N4 = 3, 5, 2, 2, 6, 4
    B = random_complex(rng, (L, half))
    coefficients = random_complex(rng, (2 * L,))
    explicit_r15 = np.concatenate(
        [B.T @ coefficients[:L], B.T @ coefficients[L:]]
    )
    assert np.allclose(compact_r15(B, coefficients), explicit_r15)

    taps = [0, 3]
    Wc = random_complex(rng, (2 * L, Mv))
    explicit_r16 = np.zeros((2 * half, N3), dtype=complex)
    for t in range(N3):
        combined = sum(
            Wc[:, f] * np.exp(2j * np.pi * t * tap / N3)
            for f, tap in enumerate(taps)
        )
        explicit_r16[:, t] = np.concatenate(
            [B.T @ combined[:L], B.T @ combined[L:]]
        )
    assert np.allclose(compact_r16(B, Wc, taps, N3), explicit_r16)

    shifts = [0, 1]
    tensor = random_complex(rng, (2 * L, Mv, Q))
    explicit_r18 = np.zeros((N4, N3, 2 * half), dtype=complex)
    for slot in range(N4):
        for t in range(N3):
            combined = sum(
                tensor[:, f, q]
                * np.exp(2j * np.pi * t * taps[f] / N3)
                * np.exp(2j * np.pi * slot * shifts[q] / N4)
                for f in range(Mv)
                for q in range(Q)
            )
            explicit_r18[slot, t] = np.concatenate(
                [B.T @ combined[:L], B.T @ combined[L:]]
            )
    assert np.allclose(compact_r18(B, tensor, taps, shifts, N3, N4), explicit_r18)
    assert np.allclose(tucker_r18(B, tensor, taps, shifts, N3, N4), explicit_r18)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_eigen_precoder_is_normalized_dominant_subspace(rank):
    rng = np.random.default_rng(1300 + rank)
    H = random_complex(rng, (4, 5, 8))
    W = eigen_precoder(H, rank=rank)
    assert np.allclose(
        np.swapaxes(W, -2, -1).conj() @ W,
        np.eye(rank) / rank,
        atol=1e-12,
    )
