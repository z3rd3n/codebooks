"""Bit-level PMI serialization (plan B2).

``pack(cbk, pmi)`` produces the actual feedback bitstream ('0'/'1' string)
and ``unpack(cbk, bits, rank)`` reconstructs the PMI.  Field widths are
derived from the same configuration logic as ``overhead_bits``, so the
round-trip property

    unpack(pack(pmi)) == pmi   and   len(pack(pmi)) == total_overhead_bits(pmi)

forces the bit accounting to be honest: any element the formulas over- or
under-count breaks one of the two equalities.

Reporting-reduction rules are honored exactly: the strongest coefficient's
k1/k2/c are never serialized (fixed to 15/7/0 on unpack), zero-amplitude R15
coefficients carry no phase bits, and conditional elements (i12, i15, i16,
i110, i22) appear only in the configurations that report them.
"""

from __future__ import annotations

import math
from math import comb

import numpy as np


class BitWriter:
    def __init__(self) -> None:
        self.bits: list[str] = []

    def write(self, value: int, width: int) -> None:
        value = int(value)
        if width < 0 or value < 0 or value >= (1 << width):
            if not (width == 0 and value == 0):
                raise ValueError(f"value {value} does not fit in {width} bits")
        if width:
            self.bits.append(format(value, f"0{width}b"))

    def getvalue(self) -> str:
        return "".join(self.bits)


class BitReader:
    def __init__(self, bits: str) -> None:
        self.bits = bits
        self.pos = 0

    def read(self, width: int) -> int:
        if width == 0:
            return 0
        if self.pos + width > len(self.bits):
            raise ValueError("bitstream exhausted")
        out = int(self.bits[self.pos : self.pos + width], 2)
        self.pos += width
        return out

    def done(self) -> bool:
        return self.pos == len(self.bits)


def _w(n: int) -> int:
    """ceil(log2(n)) with the spec's 'at least one codepoint' convention."""
    return math.ceil(math.log2(n)) if n > 1 else 0


def _require_done(reader: BitReader) -> None:
    """Reject malformed external input that carries trailing, unconsumed bits.

    Wire-format validation must raise ``ValueError`` (not ``assert``, which is
    stripped under ``python -O`` and reports a bare ``AssertionError``)."""
    if not reader.done():
        remaining = len(reader.bits) - reader.pos
        raise ValueError(f"bitstream has {remaining} trailing bit(s)")


# ---------------------------------------------------------------------------
# R15 Type I
# ---------------------------------------------------------------------------

def pack_type1(cbk, pmi) -> str:
    w = BitWriter()
    w.write(pmi.i11, _w(cbk._n_i11(pmi.rank)))
    if cbk.antenna.N2 > 1:
        w.write(pmi.i12, _w(cbk._n_i12(pmi.rank)))
    if pmi.rank in (2, 3, 4):
        w.write(pmi.i13, _w(cbk._n_i13(pmi.rank)))
    for t in range(cbk.N3):
        w.write(int(pmi.i2[t]), _w(cbk._n_i2(pmi.rank)))
    return w.getvalue()


def unpack_type1(cbk, bits: str, rank: int):
    from .type1 import Type1PMI

    r = BitReader(bits)
    i11 = r.read(_w(cbk._n_i11(rank)))
    i12 = r.read(_w(cbk._n_i12(rank))) if cbk.antenna.N2 > 1 else 0
    i13 = r.read(_w(cbk._n_i13(rank))) if rank in (2, 3, 4) else None
    i2 = np.array([r.read(_w(cbk._n_i2(rank))) for _ in range(cbk.N3)])
    _require_done(r)
    return Type1PMI(rank=rank, mode=cbk.mode, i11=i11, i12=i12, i2=i2, i13=i13)


def pack_type1_multipanel(cbk, pmi) -> str:
    G1, G2 = cbk.antenna.n_beams
    w = BitWriter()
    w.write(pmi.i11, _w(G1))
    if cbk.antenna.N2 > 1:
        w.write(pmi.i12, _w(G2))
    if pmi.rank > 1:
        w.write(pmi.i13, _w(cbk._n_i13(pmi.rank)))
    for phase in pmi.i14:
        w.write(phase, 2)
    if cbk.mode == 1:
        width = 2 if pmi.rank == 1 else 1
        for value in pmi.i2:
            w.write(value, width)
    else:
        widths = (2 if pmi.rank == 1 else 1, 1, 1)
        for state in pmi.i2:
            for value, width in zip(state, widths):
                w.write(value, width)
    return w.getvalue()


def unpack_type1_multipanel(cbk, bits: str, rank: int):
    from .type1_multipanel import Type1MPPMI

    G1, G2 = cbk.antenna.n_beams
    r = BitReader(bits)
    i11 = r.read(_w(G1))
    i12 = r.read(_w(G2)) if cbk.antenna.N2 > 1 else 0
    i13 = r.read(_w(cbk._n_i13(rank))) if rank > 1 else None
    i14 = tuple(r.read(2) for _ in range(cbk._i14_shape()[0]))
    if cbk.mode == 1:
        width = 2 if rank == 1 else 1
        i2 = np.array([r.read(width) for _ in range(cbk.N3)])
    else:
        widths = (2 if rank == 1 else 1, 1, 1)
        i2 = np.array([[r.read(width) for width in widths] for _ in range(cbk.N3)])
    _require_done(r)
    return Type1MPPMI(rank, cbk.mode, i11, i12, i14, i2, i13)


# ---------------------------------------------------------------------------
# R15 Type II
# ---------------------------------------------------------------------------

def _r15_widths(cbk):
    a = cbk.antenna
    if cbk.port_selection:
        spatial = [("i11", _w(math.ceil(a.P / (2 * cbk.d))))]
    else:
        spatial = [("i11", _w(a.O1 * a.O2)), ("i12", _w(comb(a.N1 * a.N2, cbk.L)))]
    return spatial


def pack_r15(cbk, pmi) -> str:
    a = cbk.antenna
    L2 = 2 * cbk.L
    w = BitWriter()
    if cbk.port_selection:
        w.write(pmi.i11_ps, _w(math.ceil(a.P / (2 * cbk.d))))
    else:
        w.write(pmi.q2 * a.O1 + pmi.q1, _w(a.O1 * a.O2))
        w.write(pmi.i12, _w(comb(a.N1 * a.N2, cbk.L)))
    for li in range(pmi.rank):
        w.write(pmi.i13[li], _w(L2))
    for li in range(pmi.rank):
        for i in range(L2):  # i14: 3-bit wideband amplitudes, strongest skipped
            if i != pmi.i13[li]:
                w.write(int(pmi.k1[li, i]), 3)
    for li in range(pmi.rank):
        i_star = pmi.i13[li]
        strong, weak, zero = cbk._partition(pmi.k1[li], i_star)
        sizes = cbk._phase_alphabets(pmi.k1[li], i_star)
        for t in range(cbk.N3):
            for i in range(L2):  # i21: phases (zero-amp and strongest skipped)
                if i == i_star or i in zero:
                    continue
                w.write(int(pmi.c[li, t, i]), round(math.log2(sizes[i])))
            if cbk.sa:
                for i in strong:  # i22: 1-bit subband amplitudes
                    w.write(int(pmi.k2[li, t, i]), 1)
    return w.getvalue()


def unpack_r15(cbk, bits: str, rank: int):
    from .type2_r15 import R15Type2PMI

    a = cbk.antenna
    L2 = 2 * cbk.L
    r = BitReader(bits)
    pmi = R15Type2PMI(rank=rank)
    if cbk.port_selection:
        pmi.i11_ps = r.read(_w(math.ceil(a.P / (2 * cbk.d))))
    else:
        q = r.read(_w(a.O1 * a.O2))
        pmi.q1, pmi.q2 = q % a.O1, q // a.O1
        pmi.i12 = r.read(_w(comb(a.N1 * a.N2, cbk.L)))
    pmi.i13 = [r.read(_w(L2)) for _ in range(rank)]
    pmi.k1 = np.zeros((rank, L2), dtype=int)
    pmi.k2 = np.ones((rank, cbk.N3, L2), dtype=int)
    pmi.c = np.zeros((rank, cbk.N3, L2), dtype=int)
    for li in range(rank):
        for i in range(L2):
            pmi.k1[li, i] = 7 if i == pmi.i13[li] else r.read(3)
    for li in range(rank):
        i_star = pmi.i13[li]
        strong, weak, zero = cbk._partition(pmi.k1[li], i_star)
        sizes = cbk._phase_alphabets(pmi.k1[li], i_star)
        for t in range(cbk.N3):
            for i in range(L2):
                if i == i_star or i in zero:
                    continue
                pmi.c[li, t, i] = r.read(round(math.log2(sizes[i])))
            if cbk.sa:
                for i in strong:
                    pmi.k2[li, t, i] = r.read(1)
    _require_done(r)
    return pmi


# ---------------------------------------------------------------------------
# R16 / R17 / R18 shared coefficient block (i23 / i24 / i25)
# ---------------------------------------------------------------------------

def _pack_coefficients(w, pmi, star_flat, n_psk):
    """k1 (other polarization), then k2/c for nonzero bitmap entries except
    the strongest; ``star_flat[li]`` is the strongest entry's flat index into
    the layer's bitmap array."""
    for li in range(pmi.rank):
        bm = pmi.i17[li].reshape(-1)
        k2 = pmi.k2[li].reshape(-1)
        c = pmi.c[li].reshape(-1)
        p_star = 0 if pmi.k1[li, 0] == 15 else 1
        w.write(int(pmi.k1[li, 1 - p_star]), 4)  # i23
        for j in range(bm.size):  # i24 + i25
            if bm[j] and j != star_flat[li]:
                w.write(int(k2[j]), 3)
                w.write(int(c[j]), _w(n_psk))


def _unpack_coefficients(r, pmi, star_flat, p_stars, n_psk, shape):
    rank = pmi.rank
    pmi.k1 = np.ones((rank, 2), dtype=int)
    pmi.k2 = np.zeros((rank, *shape), dtype=int)
    pmi.c = np.zeros((rank, *shape), dtype=int)
    for li in range(rank):
        pmi.k1[li, p_stars[li]] = 15
        pmi.k1[li, 1 - p_stars[li]] = r.read(4)
        bm = pmi.i17[li].reshape(-1)
        k2 = pmi.k2[li].reshape(-1)
        c = pmi.c[li].reshape(-1)
        for j in range(bm.size):
            if bm[j] and j != star_flat[li]:
                k2[j] = r.read(3)
                c[j] = r.read(_w(n_psk))
        k2[star_flat[li]] = 7
        c[star_flat[li]] = 0
        pmi.k2[li] = k2.reshape(shape)
        pmi.c[li] = c.reshape(shape)


def _tap_width(cbk, Mv: int) -> int:
    if cbk.N3 > 19:
        return _w(comb(2 * Mv - 1, Mv - 1))
    return _w(comb(cbk.N3 - 1, Mv - 1))


# ---------------------------------------------------------------------------
# R16 eType II
# ---------------------------------------------------------------------------

def pack_r16(cbk, pmi) -> str:
    from .etype2_r16 import decode_i18

    a = cbk.antenna
    Mv, L2 = cbk.Mv(pmi.rank), 2 * cbk.L
    w = BitWriter()
    if cbk.port_selection:
        w.write(pmi.i11_ps, _w(math.ceil(a.P / (2 * cbk.d))))
    else:
        w.write(pmi.q2 * a.O1 + pmi.q1, _w(a.O1 * a.O2))
        w.write(pmi.i12, _w(comb(a.N1 * a.N2, cbk.L)))
    if cbk.N3 > 19:
        w.write(pmi.i15, _w(2 * Mv))
    for li in range(pmi.rank):
        w.write(pmi.i16[li], _tap_width(cbk, Mv))
    for li in range(pmi.rank):  # i17 bitmap, one bit per (f, i) in array order
        for b in pmi.i17[li].reshape(-1):
            w.write(int(b), 1)
    for li in range(pmi.rank):
        w.write(pmi.i18[li], _w(L2))
    star_flat = []
    for li in range(pmi.rank):
        i_star = decode_i18(pmi.i18[li], pmi.i17[li], pmi.rank)
        star_flat.append(0 * L2 + i_star)  # entry (f=0, i_star) in (Mv, 2L)
    _pack_coefficients(w, pmi, star_flat, cbk.N_PSK)
    return w.getvalue()


def unpack_r16(cbk, bits: str, rank: int):
    from .etype2_r16 import R16Type2PMI, decode_i18

    a = cbk.antenna
    Mv, L2 = cbk.Mv(rank), 2 * cbk.L
    r = BitReader(bits)
    pmi = R16Type2PMI(rank=rank)
    if cbk.port_selection:
        pmi.i11_ps = r.read(_w(math.ceil(a.P / (2 * cbk.d))))
    else:
        q = r.read(_w(a.O1 * a.O2))
        pmi.q1, pmi.q2 = q % a.O1, q // a.O1
        pmi.i12 = r.read(_w(comb(a.N1 * a.N2, cbk.L)))
    if cbk.N3 > 19:
        pmi.i15 = r.read(_w(2 * Mv))
    pmi.i16 = [r.read(_tap_width(cbk, Mv)) for _ in range(rank)]
    pmi.i17 = np.zeros((rank, Mv, L2), dtype=bool)
    for li in range(rank):
        flat = np.array([r.read(1) for _ in range(Mv * L2)], dtype=bool)
        pmi.i17[li] = flat.reshape(Mv, L2)
    pmi.i18 = [r.read(_w(L2)) for _ in range(rank)]
    star_flat, p_stars = [], []
    for li in range(rank):
        i_star = decode_i18(pmi.i18[li], pmi.i17[li], rank)
        star_flat.append(i_star)
        p_stars.append(i_star // cbk.L)
    _unpack_coefficients(r, pmi, star_flat, p_stars, cbk.N_PSK, (Mv, L2))
    _require_done(r)
    return pmi


# ---------------------------------------------------------------------------
# R17 FeType II
# ---------------------------------------------------------------------------

def pack_r17(cbk, pmi) -> str:
    w = BitWriter()
    if cbk.alpha < 1:
        w.write(pmi.i12, _w(comb(cbk.antenna.P // 2, cbk.L)))
    if cbk.M == 2 and min(cbk.N_window, cbk.N3) > 2:
        w.write(pmi.i16, _w(cbk.N_window - 1))
    for li in range(pmi.rank):
        for b in pmi.i17[li].reshape(-1):
            w.write(int(b), 1)
    for li in range(pmi.rank):
        w.write(pmi.i18[li], _w(cbk.K1 * cbk.M))
    star_flat = []
    for li in range(pmi.rank):
        f_star, i_star = divmod(pmi.i18[li], cbk.K1)
        star_flat.append(f_star * cbk.K1 + i_star)
    _pack_coefficients(w, pmi, star_flat, cbk.N_PSK)
    return w.getvalue()


def unpack_r17(cbk, bits: str, rank: int):
    from .fetype2_r17 import R17Type2PMI

    r = BitReader(bits)
    pmi = R17Type2PMI(rank=rank)
    if cbk.alpha < 1:
        pmi.i12 = r.read(_w(comb(cbk.antenna.P // 2, cbk.L)))
    if cbk.M == 2 and min(cbk.N_window, cbk.N3) > 2:
        pmi.i16 = r.read(_w(cbk.N_window - 1))
    pmi.i17 = np.zeros((rank, cbk.M, cbk.K1), dtype=bool)
    for li in range(rank):
        flat = np.array([r.read(1) for _ in range(cbk.M * cbk.K1)], dtype=bool)
        pmi.i17[li] = flat.reshape(cbk.M, cbk.K1)
    pmi.i18 = [r.read(_w(cbk.K1 * cbk.M)) for _ in range(rank)]
    star_flat, p_stars = [], []
    for li in range(rank):
        f_star, i_star = divmod(pmi.i18[li], cbk.K1)
        star_flat.append(f_star * cbk.K1 + i_star)
        p_stars.append(i_star // cbk.L)
    _unpack_coefficients(r, pmi, star_flat, p_stars, cbk.N_PSK, (cbk.M, cbk.K1))
    _require_done(r)
    return pmi


# ---------------------------------------------------------------------------
# R18 eType II Doppler
# ---------------------------------------------------------------------------

def pack_r18(cbk, pmi) -> str:
    from .etype2_r18 import decode_i18

    a = cbk.antenna
    Mv, L2, Q = cbk.Mv(pmi.rank), 2 * cbk.L, cbk.Q
    w = BitWriter()
    w.write(pmi.q2 * a.O1 + pmi.q1, _w(a.O1 * a.O2))
    w.write(pmi.i12, _w(comb(a.N1 * a.N2, cbk.L)))
    if cbk.N3 > 19:
        w.write(pmi.i15, _w(2 * Mv))
    for li in range(pmi.rank):
        w.write(pmi.i16[li], _tap_width(cbk, Mv))
    for li in range(pmi.rank):
        for b in pmi.i17[li].reshape(-1):
            w.write(int(b), 1)
    for li in range(pmi.rank):
        w.write(pmi.i18[li], _w(L2 * Q))
    if cbk.N4 > 1:
        for li in range(pmi.rank):
            w.write(pmi.i110[li], _w(cbk.N4 - 1))
    star_flat = []
    for li in range(pmi.rank):
        i_star, tau_star = decode_i18(pmi.i18[li], pmi.i17[li], pmi.rank, cbk.L)
        star_flat.append((tau_star * Mv + 0) * L2 + i_star)  # (Q, Mv, 2L) order
    _pack_coefficients(w, pmi, star_flat, cbk.N_PSK)
    return w.getvalue()


def unpack_r18(cbk, bits: str, rank: int):
    from .etype2_r18 import R18Type2PMI, decode_i18

    a = cbk.antenna
    Mv, L2, Q = cbk.Mv(rank), 2 * cbk.L, cbk.Q
    r = BitReader(bits)
    pmi = R18Type2PMI(rank=rank)
    q = r.read(_w(a.O1 * a.O2))
    pmi.q1, pmi.q2 = q % a.O1, q // a.O1
    pmi.i12 = r.read(_w(comb(a.N1 * a.N2, cbk.L)))
    if cbk.N3 > 19:
        pmi.i15 = r.read(_w(2 * Mv))
    pmi.i16 = [r.read(_tap_width(cbk, Mv)) for _ in range(rank)]
    pmi.i17 = np.zeros((rank, Q, Mv, L2), dtype=bool)
    for li in range(rank):
        flat = np.array([r.read(1) for _ in range(Q * Mv * L2)], dtype=bool)
        pmi.i17[li] = flat.reshape(Q, Mv, L2)
    pmi.i18 = [r.read(_w(L2 * Q)) for _ in range(rank)]
    if cbk.N4 > 1:
        pmi.i110 = [r.read(_w(cbk.N4 - 1)) for _ in range(rank)]
    star_flat, p_stars = [], []
    for li in range(rank):
        i_star, tau_star = decode_i18(pmi.i18[li], pmi.i17[li], rank, cbk.L)
        star_flat.append((tau_star * Mv + 0) * L2 + i_star)
        p_stars.append(i_star // cbk.L)
    _unpack_coefficients(r, pmi, star_flat, p_stars, cbk.N_PSK, (Q, Mv, L2))
    _require_done(r)
    return pmi


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def pack(cbk, pmi) -> str:
    from .etype2_r16 import R16Type2Codebook
    from .etype2_r18 import R18Type2Codebook
    from .fetype2_r17 import R17Type2Codebook
    from .type1 import Type1Codebook
    from .type1_multipanel import Type1MultiPanelCodebook
    from .type2_r15 import R15Type2Codebook

    if isinstance(cbk, Type1Codebook):
        return pack_type1(cbk, pmi)
    if isinstance(cbk, Type1MultiPanelCodebook):
        return pack_type1_multipanel(cbk, pmi)
    if isinstance(cbk, R15Type2Codebook):
        return pack_r15(cbk, pmi)
    if isinstance(cbk, R18Type2Codebook):
        return pack_r18(cbk, pmi)
    if isinstance(cbk, R16Type2Codebook):
        return pack_r16(cbk, pmi)
    if isinstance(cbk, R17Type2Codebook):
        return pack_r17(cbk, pmi)
    raise TypeError(f"no serializer for {type(cbk).__name__}")


def unpack(cbk, bits: str, rank: int):
    from .etype2_r16 import R16Type2Codebook
    from .etype2_r18 import R18Type2Codebook
    from .fetype2_r17 import R17Type2Codebook
    from .type1 import Type1Codebook
    from .type1_multipanel import Type1MultiPanelCodebook
    from .type2_r15 import R15Type2Codebook

    if isinstance(cbk, Type1Codebook):
        return unpack_type1(cbk, bits, rank)
    if isinstance(cbk, Type1MultiPanelCodebook):
        return unpack_type1_multipanel(cbk, bits, rank)
    if isinstance(cbk, R15Type2Codebook):
        return unpack_r15(cbk, bits, rank)
    if isinstance(cbk, R18Type2Codebook):
        return unpack_r18(cbk, bits, rank)
    if isinstance(cbk, R16Type2Codebook):
        return unpack_r16(cbk, bits, rank)
    if isinstance(cbk, R17Type2Codebook):
        return unpack_r17(cbk, bits, rank)
    raise TypeError(f"no serializer for {type(cbk).__name__}")
