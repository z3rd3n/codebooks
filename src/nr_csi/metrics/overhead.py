"""Feedback bit-overhead formulas (paper Tables bit1/bit2).

``r15_bits``/``r16_bits``/``r18_bits`` are pure transcriptions of the
per-information-element formulas, parameterized so the paper's Fig. f2
comparison can be evaluated for any configuration.  ``f2_comparison``
aggregates them under an *equal time coverage* convention:

* R15/R16 have no temporal compression, so covering N4 future slot
  intervals requires N4 reports;
* R18 covers all N4 intervals with a single predicted-PMI report.

Note (erratum, see README): the absolute bar heights of the paper's Fig. f2
cannot be reproduced from the paper's own Table bit1/bit2 formulas with the
stated parameters; the qualitative claims (R15 >> R16 > R18, equal growth
rate in L) do hold and are asserted by the test suite.
"""

from __future__ import annotations

import math
from math import comb

from ..config import AntennaConfig


def r15_bits(
    antenna: AntennaConfig,
    L: int,
    v: int = 2,
    N3: int = 18,
    n_psk: int = 4,
    K2: int = 6,
    Ml: int | None = None,
    subband_amplitude: bool = True,
) -> dict[str, int]:
    """R15 Type II regular codebook overhead (Tables bit1/bit2, R15 column).

    Ml: number of non-zero wideband amplitudes per layer (defaults to 2L).
    i2 elements are per subband and accumulated over the N3 subbands.
    """
    a = antenna
    Ml = 2 * L if Ml is None else Ml
    m = min(Ml, K2)
    bits = {
        "i11": math.ceil(math.log2(a.O1 * a.O2)),
        "i12": math.ceil(math.log2(comb(a.N1 * a.N2, L))),
        "i13": v * math.ceil(math.log2(2 * L)),
        "i14": v * 3 * (2 * L - 1),
    }
    log_psk = round(math.log2(n_psk))
    # phases: the m strongest coefficients at n_psk resolution (strongest not
    # reported), the Ml - m weakest non-zero ones at QPSK
    bits["i21"] = v * N3 * ((m - 1) * log_psk + 2 * (Ml - m))
    if subband_amplitude:
        bits["i22"] = v * N3 * (m - 1)
    return bits


def r16_bits(
    antenna: AntennaConfig,
    L: int,
    v: int = 2,
    N3: int = 18,
    Mv: int = 5,
    K_nz: int = 20,
) -> dict[str, int]:
    """R16 Enhanced Type II regular codebook overhead (one PMI report)."""
    a = antenna
    bits = {
        "i11": math.ceil(math.log2(a.O1 * a.O2)),
        "i12": math.ceil(math.log2(comb(a.N1 * a.N2, L))),
        "i17": v * 2 * L * Mv,
        "i18": v * math.ceil(math.log2(2 * L)),
        "i23": 4 * v,
        "i24": 3 * (K_nz - v),
        "i25": 4 * (K_nz - v),
    }
    if N3 > 19:
        bits["i15"] = math.ceil(math.log2(2 * Mv))
        bits["i16"] = v * math.ceil(math.log2(comb(2 * Mv - 1, Mv - 1)))
    else:
        bits["i16"] = v * math.ceil(math.log2(comb(N3 - 1, Mv - 1)))
    return bits


def r18_bits(
    antenna: AntennaConfig,
    L: int,
    v: int = 2,
    N3: int = 18,
    Mv: int = 5,
    Q: int = 2,
    N4: int = 4,
    K_nz: int = 20,
) -> dict[str, int]:
    """R18 Doppler codebook overhead (one report covers N4 intervals)."""
    bits = r16_bits(antenna, L, v, N3, Mv, K_nz)
    bits["i17"] = v * 2 * L * Mv * Q
    bits["i18"] = v * math.ceil(math.log2(2 * L * Q))
    if N4 > 1:
        bits["i110"] = v * math.ceil(math.log2(N4 - 1))
    return bits


def f2_comparison(
    antenna: AntennaConfig,
    Ls: tuple[int, ...] = (1, 2, 3, 4),
    v: int = 2,
    N3: int = 18,
    N4: int = 4,
    Mv: int = 5,
    Q: int = 2,
    n_psk: int = 4,
    K2: int = 6,
    K_nz: int = 20,
) -> dict[str, list[int]]:
    """Total feedback bits to cover N4 slot intervals, per codebook and L."""
    out: dict[str, list[int]] = {"R15 Regular": [], "R16 Regular": [], "R18 Regular": []}
    for L in Ls:
        out["R15 Regular"].append(
            N4 * sum(r15_bits(antenna, L, v, N3, n_psk, K2).values())
        )
        out["R16 Regular"].append(N4 * sum(r16_bits(antenna, L, v, N3, Mv, K_nz).values()))
        out["R18 Regular"].append(
            sum(r18_bits(antenna, L, v, N3, Mv, Q, N4, K_nz).values())
        )
    return out
