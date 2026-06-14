"""Runtime PMI validation, shared by ``precoder()`` (gNB side) and tests.

Each helper checks a reported PMI against its codebook configuration:
field presence, array shapes, index ranges, and the strongest-coefficient
conventions (k1 = 15 / k2 = 7 / c = 0 at the unreported strongest entry).
They raise ``ValueError`` with a descriptive message -- a gNB must reject a
malformed report rather than reconstruct garbage.

The K0 coefficient budgets are selection-side constraints and deliberately
not enforced here (a reconstructing gNB can apply any bitmap it is given).
"""

from __future__ import annotations

import math
from math import comb

import numpy as np


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"malformed PMI: {msg}")


def _check_array(arr, name: str, shape: tuple[int, ...], lo: int, hi: int) -> None:
    _check(arr is not None, f"{name} missing")
    a = np.asarray(arr)
    _check(a.shape == shape, f"{name} shape {a.shape} != {shape}")
    if a.dtype != bool:
        _check(
            bool(np.all((a >= lo) & (a <= hi))),
            f"{name} values outside [{lo}, {hi}]",
        )


def validate_type1(cbk, pmi) -> None:
    _check(1 <= pmi.rank <= min(8, cbk.antenna.P),
           f"rank {pmi.rank} unsupported for P={cbk.antenna.P}")
    _check(pmi.mode == cbk.mode, f"mode {pmi.mode} != configured {cbk.mode}")
    n11 = cbk._n_i11(pmi.rank)
    n12 = cbk._n_i12(pmi.rank)
    _check(0 <= pmi.i11 < n11, f"i11={pmi.i11} not in [0, {n11})")
    _check(0 <= pmi.i12 < n12, f"i12={pmi.i12} not in [0, {n12})")
    n_i2 = cbk._n_i2(pmi.rank)
    _check_array(pmi.i2, "i2", (cbk.N3,), 0, n_i2 - 1)
    if pmi.rank in (2, 3, 4):
        n_off = cbk._n_i13(pmi.rank)
        _check(pmi.i13 is not None and 0 <= pmi.i13 < n_off,
               f"i13={pmi.i13} not in [0, {n_off})")
    else:
        _check(pmi.i13 is None, f"i13 must not be reported for rank {pmi.rank}")


def validate_type1_multipanel(cbk, pmi) -> None:
    G1, G2 = cbk.antenna.n_beams
    _check(1 <= pmi.rank <= 4, f"rank {pmi.rank} not in 1..4")
    _check(pmi.mode == cbk.mode, f"mode {pmi.mode} != configured {cbk.mode}")
    _check(0 <= pmi.i11 < G1, f"i11={pmi.i11} not in [0, {G1})")
    _check(0 <= pmi.i12 < G2, f"i12={pmi.i12} not in [0, {G2})")
    _check(isinstance(pmi.i14, tuple), "i14 must be a tuple")
    _check(len(pmi.i14) == cbk._i14_shape()[0],
           f"i14 length {len(pmi.i14)} != {cbk._i14_shape()[0]}")
    _check(all(0 <= int(p) < 4 for p in pmi.i14), "i14 values outside [0, 3]")
    if pmi.rank > 1:
        n_off = cbk._n_i13(pmi.rank)
        _check(pmi.i13 is not None and 0 <= pmi.i13 < n_off,
               f"i13={pmi.i13} not in [0, {n_off})")
    else:
        _check(pmi.i13 is None, "i13 must not be reported for rank 1")
    if cbk.mode == 1:
        _check_array(pmi.i2, "i2", (cbk.N3,), 0, 3 if pmi.rank == 1 else 1)
    else:
        _check_array(pmi.i2, "i2", (cbk.N3, 3), 0, 3)
        i2 = np.asarray(pmi.i2)
        _check(bool(np.all(i2[:, 1:] <= 1)), "i2 panel phase values outside [0, 1]")
        if pmi.rank > 1:
            _check(bool(np.all(i2[:, 0] <= 1)), "i2 polarization phase outside [0, 1]")


def _validate_spatial(cbk, pmi) -> None:
    a = cbk.antenna
    if getattr(cbk, "port_selection", False):
        n_init = math.ceil(a.P / (2 * cbk.d))
        _check(pmi.i11_ps is not None and 0 <= pmi.i11_ps < n_init,
               f"i11={pmi.i11_ps} not in [0, {n_init})")
    else:
        _check(pmi.q1 is not None and 0 <= pmi.q1 < a.O1, f"q1={pmi.q1} not in [0, {a.O1})")
        _check(pmi.q2 is not None and 0 <= pmi.q2 < a.O2, f"q2={pmi.q2} not in [0, {a.O2})")
        n_comb = comb(a.N1 * a.N2, cbk.L)
        _check(pmi.i12 is not None and 0 <= pmi.i12 < n_comb,
               f"i12={pmi.i12} not in [0, C({a.N1 * a.N2},{cbk.L}))")


def validate_r15(cbk, pmi) -> None:
    L2 = 2 * cbk.L
    _check(pmi.rank in (1, 2), f"rank {pmi.rank} not in 1..2")
    _validate_spatial(cbk, pmi)
    _check(pmi.i13 is not None and len(pmi.i13) == pmi.rank, "i13 must have one entry per layer")
    _check_array(pmi.k1, "k1", (pmi.rank, L2), 0, 7)
    _check_array(pmi.k2, "k2", (pmi.rank, cbk.N3, L2), 0, 1)
    _check_array(pmi.c, "c", (pmi.rank, cbk.N3, L2), 0, cbk.n_psk - 1)
    for li in range(pmi.rank):
        i_star = pmi.i13[li]
        _check(0 <= i_star < L2, f"i13[{li}]={i_star} not in [0, {L2})")
        _check(pmi.k1[li, i_star] == 7, f"layer {li}: strongest coefficient must have k1=7")
        _check(bool(np.all(pmi.c[li, :, i_star] == 0)),
               f"layer {li}: strongest coefficient must have c=0")


def _validate_taps(cbk, pmi, Mv: int) -> None:
    _check(len(pmi.i16) == pmi.rank, "i16 must have one entry per layer")
    if cbk.N3 > 19:
        _check(pmi.i15 is not None and 0 <= pmi.i15 < 2 * Mv,
               f"i15={pmi.i15} not in [0, {2 * Mv})")
        hi = comb(2 * Mv - 1, Mv - 1)
    else:
        _check(pmi.i15 is None, "i15 must not be reported for N3 <= 19")
        hi = comb(cbk.N3 - 1, Mv - 1)
    for li, i16 in enumerate(pmi.i16):
        _check(0 <= i16 < hi, f"i16[{li}]={i16} not in [0, {hi})")


def _validate_coefficients(pmi, name_shape, n_psk: int) -> None:
    v = pmi.rank
    _check_array(pmi.i17, "i17", (v, *name_shape), 0, 1)
    _check(np.asarray(pmi.i17).dtype == bool, "i17 must be a boolean bitmap")
    _check_array(pmi.k1, "k1", (v, 2), 1, 15)
    _check_array(pmi.k2, "k2", (v, *name_shape), 0, 7)
    _check_array(pmi.c, "c", (v, *name_shape), 0, n_psk - 1)
    _check(len(pmi.i18) == v, "i18 must have one entry per layer")


def validate_r16(cbk, pmi) -> None:
    from .etype2_r16 import decode_i18

    _check(1 <= pmi.rank <= 4, f"rank {pmi.rank} not in 1..4")
    Mv, L2 = cbk.Mv(pmi.rank), 2 * cbk.L
    _validate_spatial(cbk, pmi)
    _validate_taps(cbk, pmi, Mv)
    _validate_coefficients(pmi, (Mv, L2), cbk.N_PSK)
    for li in range(pmi.rank):
        n_i18 = int(pmi.i17[li, 0].sum()) if pmi.rank == 1 else L2
        _check(0 <= pmi.i18[li] < n_i18, f"i18[{li}]={pmi.i18[li]} not in [0, {n_i18})")
        i_star = decode_i18(pmi.i18[li], pmi.i17[li], pmi.rank)
        _check(bool(pmi.i17[li, 0, i_star]),
               f"layer {li}: strongest coefficient absent from bitmap")
        _check(pmi.k1[li, i_star // cbk.L] == 15,
               f"layer {li}: strongest polarization must have k1=15")
        _check(pmi.k2[li, 0, i_star] == 7 and pmi.c[li, 0, i_star] == 0,
               f"layer {li}: strongest coefficient must have k2=7, c=0")


def validate_r17(cbk, pmi) -> None:
    _check(1 <= pmi.rank <= 4, f"rank {pmi.rank} not in 1..4")
    M, K1 = cbk.M, cbk.K1
    if cbk.alpha < 1:
        n_comb = comb(cbk.antenna.P // 2, cbk.L)
        _check(pmi.i12 is not None and 0 <= pmi.i12 < n_comb,
               f"i12={pmi.i12} not in [0, C({cbk.antenna.P // 2},{cbk.L}))")
    else:
        _check(pmi.i12 is None, "i12 must not be reported when alpha = 1")
    if M == 2 and min(cbk.N_window, cbk.N3) > 2:
        _check(pmi.i16 is not None and 0 <= pmi.i16 < cbk.N_window - 1,
               f"i16={pmi.i16} not in [0, {cbk.N_window - 1})")
    else:
        _check(pmi.i16 is None, "i16 must not be reported when M=1 or N=2")
    _validate_coefficients(pmi, (M, K1), cbk.N_PSK)
    for li in range(pmi.rank):
        _check(0 <= pmi.i18[li] < K1 * M, f"i18[{li}]={pmi.i18[li]} not in [0, {K1 * M})")
        f_star, i_star = divmod(pmi.i18[li], K1)
        _check(bool(pmi.i17[li, f_star, i_star]),
               f"layer {li}: strongest coefficient absent from bitmap")
        _check(pmi.k1[li, i_star // cbk.L] == 15,
               f"layer {li}: strongest polarization must have k1=15")
        _check(pmi.k2[li, f_star, i_star] == 7 and pmi.c[li, f_star, i_star] == 0,
               f"layer {li}: strongest coefficient must have k2=7, c=0")


def validate_r18(cbk, pmi) -> None:
    from .etype2_r18 import decode_i18

    _check(1 <= pmi.rank <= 4, f"rank {pmi.rank} not in 1..4")
    Mv, L2, Q = cbk.Mv(pmi.rank), 2 * cbk.L, cbk.Q
    _validate_spatial(cbk, pmi)
    _validate_taps(cbk, pmi, Mv)
    if Q == 2:
        _check(len(pmi.i110) == pmi.rank, "i110 must have one entry per layer")
        for li, off in enumerate(pmi.i110):
            _check(0 <= off < cbk.N4 - 1, f"i110[{li}]={off} not in [0, {cbk.N4 - 1})")
    else:
        _check(len(pmi.i110) == 0, "i110 must not be reported when N4 = 1")
    _validate_coefficients(pmi, (Q, Mv, L2), cbk.N_PSK)
    for li in range(pmi.rank):
        n_i18 = int(pmi.i17[li, :, 0, :].sum()) if pmi.rank == 1 else L2 * Q
        _check(0 <= pmi.i18[li] < n_i18, f"i18[{li}]={pmi.i18[li]} not in [0, {n_i18})")
        i_star, tau_star = decode_i18(pmi.i18[li], pmi.i17[li], pmi.rank, cbk.L)
        _check(bool(pmi.i17[li, tau_star, 0, i_star]),
               f"layer {li}: strongest coefficient absent from bitmap")
        _check(pmi.k1[li, i_star // cbk.L] == 15,
               f"layer {li}: strongest polarization must have k1=15")
        _check(pmi.k2[li, tau_star, 0, i_star] == 7 and pmi.c[li, tau_star, 0, i_star] == 0,
               f"layer {li}: strongest coefficient must have k2=7, c=0")
