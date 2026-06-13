"""Configuration objects shared by all codebooks.

The tables transcribed here come from the tutorial paper (paper/main.tex),
which mirrors TS 38.214:

* ``SUPPORTED_N1N2``      -- Table "Supported Configurations of (N1,N2) and (O1,O2)"
* ``R16_PARAM_COMBOS``    -- Table "Parameter Configurations for L, beta, p_v"
                             (paramCombination-r16)
* ``R17_PARAM_COMBOS``    -- Table "Parameter Configurations for alpha, M, beta"
                             (paramCombination-r17)
* ``R18_PARAM_COMBOS``    -- Table "Codebook Parameter Configurations for L, beta, p_v"
                             (paramCombination-Doppler-r18)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

# (N1, N2) -> (O1, O2), keyed exactly as in the paper's Table tabNO.
SUPPORTED_N1N2: dict[tuple[int, int], tuple[int, int]] = {
    (2, 1): (4, 1),
    (2, 2): (4, 4),
    (4, 1): (4, 1),
    (3, 2): (4, 4),
    (6, 1): (4, 1),
    (4, 2): (4, 4),
    (8, 1): (4, 1),
    (4, 3): (4, 4),
    (6, 2): (4, 4),
    (12, 1): (4, 1),
    (4, 4): (4, 4),
    (8, 2): (4, 4),
    (16, 1): (4, 1),
}


@dataclass(frozen=True)
class AntennaConfig:
    """Logical antenna array at the gNB (dual-polarized UPA).

    N1/N2: number of ports per polarization in the horizontal/vertical
    dimension. O1/O2: oversampling factors. P_CSI-RS = 2*N1*N2.

    ``strict=True`` enforces the (N1,N2)->(O1,O2) pairs supported by the
    standard; ``strict=False`` allows experimental geometries (e.g. the
    single-polarization study in the paper's Fig. f1).
    """

    N1: int
    N2: int
    O1: int = 4
    O2: int = 1
    strict: bool = True

    def __post_init__(self) -> None:
        if self.strict:
            expected = SUPPORTED_N1N2.get((self.N1, self.N2))
            if expected is None:
                raise ValueError(
                    f"(N1,N2)=({self.N1},{self.N2}) is not a supported configuration; "
                    f"use strict=False for experimental geometries"
                )
            if (self.O1, self.O2) != expected:
                raise ValueError(
                    f"(O1,O2)=({self.O1},{self.O2}) must be {expected} for "
                    f"(N1,N2)=({self.N1},{self.N2})"
                )

    @classmethod
    def standard(cls, N1: int, N2: int) -> "AntennaConfig":
        O1, O2 = SUPPORTED_N1N2[(N1, N2)]
        return cls(N1=N1, N2=N2, O1=O1, O2=O2)

    @property
    def n_ports_per_pol(self) -> int:
        return self.N1 * self.N2

    @property
    def P(self) -> int:
        """Number of CSI-RS ports, P_CSI-RS = 2*N1*N2 (dual polarization)."""
        return 2 * self.N1 * self.N2

    @property
    def n_beams(self) -> tuple[int, int]:
        """Size of the oversampled DFT beam grid (N1*O1, N2*O2)."""
        return self.N1 * self.O1, self.N2 * self.O2


@dataclass(frozen=True)
class SubbandConfig:
    """Frequency-domain PMI reporting granularity.

    n_subbands: number of configured CQI subbands (csi-ReportingBand).
    R: number of PMI precoding matrices per CQI subband (1 or 2).
    N3 = n_subbands * R is the total number of PMI frequency units.
    """

    n_subbands: int
    R: int = 1

    def __post_init__(self) -> None:
        if self.R not in (1, 2):
            raise ValueError("R (numberOfPMI-SubbandsPerCQI-Subband) must be 1 or 2")
        # csi-ReportingBand ranges over {3,...,18} in the paper; we allow small
        # positive values for unit tests but keep the spec bounds on both ends.
        if self.n_subbands < 1:
            raise ValueError("n_subbands must be positive")
        if self.n_subbands > 19:
            raise ValueError("at most 19 CQI subbands are configurable")

    @property
    def N3(self) -> int:
        return self.n_subbands * self.R


def m_v(p_v: Fraction | float, N3: int, R: int) -> int:
    """Number of selected delay taps: M_v = ceil(p_v * N3 / R)  (paper eq. c66)."""
    if R not in (1, 2):
        raise ValueError("R must be 1 or 2")
    return math.ceil(Fraction(p_v) * N3 / R)


@dataclass(frozen=True)
class R16ParamCombo:
    """paramCombination-r16 -> (L, p_v for v in {1,2}, p_v for v in {3,4}, beta)."""

    index: int
    L: int
    p_v12: Fraction
    p_v34: Fraction | None  # None: rank 3-4 not supported for this combination
    beta: Fraction

    def p_v(self, v: int) -> Fraction:
        if v in (1, 2):
            return self.p_v12
        if v in (3, 4):
            if self.p_v34 is None:
                raise ValueError(f"rank {v} unsupported for paramCombination-r16={self.index}")
            return self.p_v34
        raise ValueError("rank must be 1..4")


R16_PARAM_COMBOS: dict[int, R16ParamCombo] = {
    c.index: c
    for c in [
        R16ParamCombo(1, 2, Fraction(1, 4), Fraction(1, 8), Fraction(1, 4)),
        R16ParamCombo(2, 2, Fraction(1, 4), Fraction(1, 8), Fraction(1, 2)),
        R16ParamCombo(3, 4, Fraction(1, 4), Fraction(1, 8), Fraction(1, 4)),
        R16ParamCombo(4, 4, Fraction(1, 4), Fraction(1, 8), Fraction(1, 2)),
        R16ParamCombo(5, 4, Fraction(1, 4), Fraction(1, 4), Fraction(3, 4)),
        R16ParamCombo(6, 4, Fraction(1, 2), Fraction(1, 4), Fraction(1, 2)),
        R16ParamCombo(7, 6, Fraction(1, 4), None, Fraction(1, 2)),
        R16ParamCombo(8, 6, Fraction(1, 4), None, Fraction(3, 4)),
    ]
}


@dataclass(frozen=True)
class R17ParamCombo:
    """paramCombination-r17 -> (M, alpha, beta)."""

    index: int
    M: int
    alpha: Fraction
    beta: Fraction


R17_PARAM_COMBOS: dict[int, R17ParamCombo] = {
    c.index: c
    for c in [
        R17ParamCombo(1, 1, Fraction(3, 4), Fraction(1, 2)),
        R17ParamCombo(2, 1, Fraction(1, 1), Fraction(1, 2)),
        R17ParamCombo(3, 1, Fraction(1, 1), Fraction(3, 4)),
        R17ParamCombo(4, 1, Fraction(1, 1), Fraction(1, 1)),
        R17ParamCombo(5, 2, Fraction(1, 2), Fraction(1, 2)),
        R17ParamCombo(6, 2, Fraction(3, 4), Fraction(1, 2)),
        R17ParamCombo(7, 2, Fraction(1, 1), Fraction(1, 2)),
        R17ParamCombo(8, 2, Fraction(1, 1), Fraction(3, 4)),
    ]
}


@dataclass(frozen=True)
class R18ParamCombo:
    """paramCombination-Doppler-r18 -> (L, p_v for v in {1,2}, p_v for v in {3,4}, beta)."""

    index: int
    L: int
    p_v12: Fraction
    p_v34: Fraction | None
    beta: Fraction

    def p_v(self, v: int) -> Fraction:
        if v in (1, 2):
            return self.p_v12
        if v in (3, 4):
            if self.p_v34 is None:
                raise ValueError(
                    f"rank {v} unsupported for paramCombination-Doppler-r18={self.index}"
                )
            return self.p_v34
        raise ValueError("rank must be 1..4")


R18_PARAM_COMBOS: dict[int, R18ParamCombo] = {
    c.index: c
    for c in [
        R18ParamCombo(1, 2, Fraction(1, 8), Fraction(1, 16), Fraction(1, 4)),
        R18ParamCombo(2, 2, Fraction(1, 4), Fraction(1, 8), Fraction(1, 2)),
        R18ParamCombo(3, 4, Fraction(1, 4), Fraction(1, 8), Fraction(1, 4)),
        R18ParamCombo(4, 4, Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)),
        R18ParamCombo(5, 4, Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)),
        R18ParamCombo(6, 4, Fraction(1, 4), Fraction(1, 4), Fraction(3, 4)),
        R18ParamCombo(7, 4, Fraction(1, 2), Fraction(1, 4), Fraction(1, 2)),
        R18ParamCombo(8, 6, Fraction(1, 4), None, Fraction(1, 2)),
        R18ParamCombo(9, 6, Fraction(1, 4), None, Fraction(3, 4)),
    ]
}

# Configurable subband sizes (RBs) per BWP size (Table tabCSS), used by the
# Sionna adapter to map an OFDM resource grid onto PMI subbands.
SUBBAND_SIZES: list[tuple[range, tuple[int, int]]] = [
    (range(24, 73), (4, 8)),
    (range(73, 145), (8, 16)),
    (range(145, 276), (16, 32)),
]


def subband_size_options(n_rb: int) -> tuple[int, int]:
    """Allowed subband sizes (RBs) for a BWP of ``n_rb`` RBs (Table tabCSS)."""
    for rng, sizes in SUBBAND_SIZES:
        if n_rb in rng:
            return sizes
    raise ValueError(f"BWP size {n_rb} RB outside the 24..275 range of Table tabCSS")


def n3_for_bwp(n_rb: int, subband_size: int, R: int = 1) -> int:
    """N3 = (number of configured subbands) * R for a BWP of ``n_rb`` RBs.

    The paper's worked example: 273 RB with subband size 16 -> 18 subbands.
    """
    options = subband_size_options(n_rb)
    if subband_size not in options:
        raise ValueError(
            f"subband size {subband_size} not allowed for {n_rb} RB; "
            f"Table tabCSS permits {options}"
        )
    if R not in (1, 2):
        raise ValueError("R must be 1 or 2")
    return math.ceil(n_rb / subband_size) * R
