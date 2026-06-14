"""Independent equation-level oracles transcribed from ``paper/main.tex``.

These helpers deliberately avoid ``nr_csi`` DFT, combinatoric, compact-model,
and reconstruction helpers.  Tests may pass production PMI objects into the
oracles, but expected arrays are built directly from the paper equations.
"""

from __future__ import annotations

import itertools
import math
from math import comb

import numpy as np

R15_WB = np.array(
    [0.0, 1 / 8, 1 / np.sqrt(32), 1 / 4, 1 / np.sqrt(8), 1 / 2, 1 / np.sqrt(2), 1]
)
R15_SB = np.array([1 / np.sqrt(2), 1.0])
R16_REF = np.array([np.nan] + [2.0 ** (-(15 - k) / 4) for k in range(1, 16)])
R16_DIFF = np.array([2.0 ** (-(7 - k) / 2) for k in range(8)])


def paper_combo_index(indices: tuple[int, ...] | list[int], n_total: int) -> int:
    values = tuple(sorted(indices))
    return sum(comb(n_total - 1 - n, len(values) - i) for i, n in enumerate(values))


def paper_combo_decode(index: int, n_total: int, k: int) -> list[int]:
    for values in itertools.combinations(range(n_total), k):
        if paper_combo_index(values, n_total) == index:
            return list(values)
    raise ValueError(f"no combination for index={index}, n_total={n_total}, k={k}")


def paper_taps(i16: int, N3: int, Mv: int, i15: int | None) -> list[int]:
    if Mv == 1:
        return [0]
    if N3 <= 19:
        return [0] + [x + 1 for x in paper_combo_decode(i16, N3 - 1, Mv - 1)]
    if i15 is None:
        raise ValueError("i15 is required for N3 > 19")
    m_initial = i15 if i15 == 0 else i15 - 2 * Mv
    relative = [x + 1 for x in paper_combo_decode(i16, 2 * Mv - 1, Mv - 1)]
    mapped = [
        n if n <= m_initial + 2 * Mv - 1 else n + N3 - 2 * Mv for n in relative
    ]
    return sorted([0] + [n % N3 for n in mapped])


def steering(N: int, O: int, index: int) -> np.ndarray:
    return np.exp(2j * np.pi * index * np.arange(N) / (O * N))


def spatial_beam(antenna, m1: int, m2: int) -> np.ndarray:
    return np.kron(steering(antenna.N1, antenna.O1, m1), steering(antenna.N2, antenna.O2, m2))


def frequency_basis(N3: int, tap: int) -> np.ndarray:
    return np.exp(2j * np.pi * tap * np.arange(N3) / N3)


def time_basis(N4: int, shift: int) -> np.ndarray:
    return np.exp(2j * np.pi * shift * np.arange(N4) / N4)


def regular_basis(antenna, q1: int, q2: int, i12: int, L: int) -> np.ndarray:
    flat = paper_combo_decode(i12, antenna.N1 * antenna.N2, L)
    return np.stack(
        [
            spatial_beam(
                antenna,
                antenna.O1 * (n % antenna.N1) + q1,
                antenna.O2 * (n // antenna.N1) + q2,
            )
            for n in flat
        ]
    )


def ps_basis(antenna, initial: int, L: int, d: int) -> np.ndarray:
    basis = np.zeros((L, antenna.P // 2))
    for i in range(L):
        basis[i, (initial * d + i) % (antenna.P // 2)] = 1
    return basis


def dual_synthesis(basis: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    L = basis.shape[0]
    return np.concatenate(
        [basis.T @ coefficients[:L], basis.T @ coefficients[L:]], axis=0
    )


def type1_precoder(antenna, pmi) -> np.ndarray:
    offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
    out = np.zeros((1, len(pmi.i2), antenna.P, pmi.rank), dtype=complex)
    for t, i2_raw in enumerate(pmi.i2):
        i2 = int(i2_raw)
        if pmi.mode == 1:
            l, m, phase_index = pmi.i11, pmi.i12, i2
        else:
            n_phases = 4 if pmi.rank == 1 else 2
            dl, dm = offsets[i2 // n_phases]
            l, m, phase_index = 2 * pmi.i11 + dl, 2 * pmi.i12 + dm, i2 % n_phases
        v1 = spatial_beam(antenna, l, m)
        phase = np.exp(1j * np.pi * phase_index / 2)
        if pmi.rank == 1:
            out[0, t, :, 0] = np.concatenate([v1, phase * v1]) / np.sqrt(antenna.P)
            continue
        if antenna.N1 > antenna.N2 > 1:
            i13 = [(0, 0), (antenna.O1, 0), (0, antenna.O2), (2 * antenna.O1, 0)]
        elif antenna.N1 == antenna.N2:
            i13 = [(0, 0), (antenna.O1, 0), (0, antenna.O2), (antenna.O1, antenna.O2)]
        elif antenna.N1 > 2 and antenna.N2 == 1:
            i13 = [(0, 0), (antenna.O1, 0), (2 * antenna.O1, 0), (3 * antenna.O1, 0)]
        else:
            i13 = [(0, 0), (antenna.O1, 0)]
        dl, dm = i13[pmi.i13]
        v2 = spatial_beam(antenna, l + dl, m + dm)
        out[0, t] = np.stack(
            [np.concatenate([v1, phase * v1]), np.concatenate([v2, -phase * v2])],
            axis=1,
        ) / np.sqrt(2 * antenna.P)
    return out


def r15_precoder(cbk, pmi) -> np.ndarray:
    if cbk.port_selection:
        basis = ps_basis(cbk.antenna, pmi.i11_ps, cbk.L, cbk.d)
        beta_basis = 1.0
    else:
        basis = regular_basis(cbk.antenna, pmi.q1, pmi.q2, pmi.i12, cbk.L)
        beta_basis = cbk.antenna.n_ports_per_pol
    out = np.zeros((1, cbk.N3, cbk.antenna.P, pmi.rank), dtype=complex)
    for layer in range(pmi.rank):
        k1 = np.asarray(pmi.k1[layer])
        order = sorted(
            range(2 * cbk.L),
            key=lambda i: (i != pmi.i13[layer], -int(k1[i]), i),
        )
        nonzero = [i for i in order if k1[i] > 0]
        strong_count = min(len(nonzero), 4 if cbk.L in (2, 3) else 6)
        weak = set(nonzero[strong_count:])
        phase_sizes = np.array([4 if cbk.sa and i in weak else cbk.n_psk for i in range(2 * cbk.L)])
        for t in range(cbk.N3):
            p1 = R15_WB[k1]
            p2 = R15_SB[pmi.k2[layer, t]] if cbk.sa else np.ones(2 * cbk.L)
            phase = np.exp(2j * np.pi * pmi.c[layer, t] / phase_sizes)
            coefficients = p1 * p2 * phase
            denominator = np.sqrt(beta_basis * np.sum((p1 * p2) ** 2) * pmi.rank)
            out[0, t, :, layer] = dual_synthesis(basis, coefficients) / denominator
    return out


def r16_precoder(cbk, pmi) -> np.ndarray:
    if cbk.port_selection:
        basis = ps_basis(cbk.antenna, pmi.i11_ps, cbk.L, cbk.d)
        basis_scale = 1.0
    else:
        basis = regular_basis(cbk.antenna, pmi.q1, pmi.q2, pmi.i12, cbk.L)
        basis_scale = cbk.antenna.n_ports_per_pol
    Mv = cbk.Mv(pmi.rank)
    out = np.zeros((1, cbk.N3, cbk.antenna.P, pmi.rank), dtype=complex)
    polarization = np.repeat([0, 1], cbk.L)
    for layer in range(pmi.rank):
        taps = paper_taps(pmi.i16[layer], cbk.N3, Mv, pmi.i15)
        coefficients = (
            R16_DIFF[pmi.k2[layer]]
            * np.exp(2j * np.pi * pmi.c[layer] / 16)
            * pmi.i17[layer]
        ).T
        coefficients *= R16_REF[pmi.k1[layer]][polarization, None]
        for t in range(cbk.N3):
            delay = np.array([frequency_basis(cbk.N3, tap)[t] for tap in taps])
            combined = coefficients @ delay
            gamma = np.sum(np.abs(combined) ** 2)
            if gamma == 0:
                gamma = 1.0
            out[0, t, :, layer] = dual_synthesis(basis, combined) / np.sqrt(
                basis_scale * gamma * pmi.rank
            )
    return out


def r17_precoder(cbk, pmi) -> np.ndarray:
    ports = (
        list(range(cbk.L))
        if pmi.i12 is None
        else paper_combo_decode(pmi.i12, cbk.antenna.P // 2, cbk.L)
    )
    basis = np.zeros((cbk.L, cbk.antenna.P // 2))
    basis[np.arange(cbk.L), ports] = 1
    taps = [0] if cbk.M == 1 else ([0, 1] if min(cbk.N_window, cbk.N3) == 2 else [0, pmi.i16 + 1])
    out = np.zeros((1, cbk.N3, cbk.antenna.P, pmi.rank), dtype=complex)
    polarization = np.repeat([0, 1], cbk.L)
    for layer in range(pmi.rank):
        coefficients = (
            R16_DIFF[pmi.k2[layer]]
            * np.exp(2j * np.pi * pmi.c[layer] / 16)
            * pmi.i17[layer]
        ).T
        coefficients *= R16_REF[pmi.k1[layer]][polarization, None]
        for t in range(cbk.N3):
            delay = np.array([frequency_basis(cbk.N3, tap)[t] for tap in taps])
            combined = coefficients @ delay
            gamma = np.sum(np.abs(combined) ** 2)
            if gamma == 0:
                gamma = 1.0
            out[0, t, :, layer] = dual_synthesis(basis, combined) / np.sqrt(
                gamma * pmi.rank
            )
    return out


def r18_precoder(cbk, pmi) -> np.ndarray:
    basis = regular_basis(cbk.antenna, pmi.q1, pmi.q2, pmi.i12, cbk.L)
    Mv = cbk.Mv(pmi.rank)
    out = np.zeros((cbk.N4, cbk.N3, cbk.antenna.P, pmi.rank), dtype=complex)
    polarization = np.repeat([0, 1], cbk.L)
    for layer in range(pmi.rank):
        taps = paper_taps(pmi.i16[layer], cbk.N3, Mv, pmi.i15)
        shifts = [0] if cbk.Q == 1 else [0, pmi.i110[layer] + 1]
        coefficients = (
            R16_DIFF[pmi.k2[layer]]
            * np.exp(2j * np.pi * pmi.c[layer] / 16)
            * pmi.i17[layer]
        ).transpose(2, 1, 0)
        coefficients *= R16_REF[pmi.k1[layer]][polarization, None, None]
        for slot in range(cbk.N4):
            doppler = np.array([time_basis(cbk.N4, shift)[slot] for shift in shifts])
            for t in range(cbk.N3):
                delay = np.array([frequency_basis(cbk.N3, tap)[t] for tap in taps])
                combined = np.einsum("ifq,f,q->i", coefficients, delay, doppler)
                gamma = np.sum(np.abs(combined) ** 2)
                if gamma == 0:
                    gamma = 1.0
                out[slot, t, :, layer] = dual_synthesis(basis, combined) / np.sqrt(
                    cbk.antenna.n_ports_per_pol * gamma * pmi.rank
                )
    return out


def direct_su_rate(H: np.ndarray, W: np.ndarray, rho: float) -> float:
    values = []
    for h, w in zip(H.reshape(-1, *H.shape[-2:]), W.reshape(-1, *W.shape[-2:])):
        effective = h @ w
        covariance = np.eye(h.shape[-2]) + rho * effective @ effective.conj().T
        values.append(np.linalg.slogdet(covariance)[1] / math.log(2))
    return float(np.mean(values))
