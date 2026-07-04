"""Release-19 "Refined" Type II codebooks for large arrays (48/64/128 ports).

TS 38.214 v19.3.0 introduces refined variants of the enhanced Type II family
for arrays aggregated from K CSI-RS resources:

* 5.2.2.2.5a  'eTypeII-r19'               -> refines 5.2.2.2.5  (R16 eType II)
* 5.2.2.2.9a  'typeII-FePortSelection-r19'-> refines 5.2.2.2.7  (R17 feType II PS)
* 5.2.2.2.11a 'typeII-Doppler-r19'        -> refines 5.2.2.2.10 (R18 predicted)

Each clause states explicitly "The codebook is defined as in Clause 5.2.2.2.x",
i.e. the precoder reconstruction is *identical* to the corresponding R16/R17/R18
codebook.  The only differences are:

* the larger (N1, N2) geometries of Table 5.2.2.2.1a-1 (48/64/128 ports,
  O1 = O2 = 4) -- handled by ``AntennaConfig`` via ``SUPPORTED_N1N2_R19``;
* paramCombination restrictions (the L=6 / high-overhead rows are barred when
  rank > 2 is permitted or R = 2);
* a Release-19 RI restriction.

These classes therefore subclass the existing implementations and add only the
configuration guards.  The Release-19 *Type I* refined codebooks (5.2.2.2.1a /
5.2.2.2.2a) are a structurally new design (modeA/modeB, independently selected
high-rank beams) and are intentionally not covered here.
"""

from __future__ import annotations

import numpy as np

from ..config import SUPPORTED_N1N2_R19, AntennaConfig
from .etype2_r16 import R16Type2Codebook
from .etype2_r18 import R18Type2Codebook
from .fetype2_r17 import R17Type2Codebook


def _require_r19_array(antenna: AntennaConfig, allowed_ports: tuple[int, ...]) -> None:
    """The refined codebooks are only defined for the Table 5.2.2.2.1a-1
    geometries restricted to ``allowed_ports`` CSI-RS ports."""
    if (antenna.N1, antenna.N2) not in SUPPORTED_N1N2_R19:
        raise ValueError(
            f"(N1,N2)=({antenna.N1},{antenna.N2}) is not a Release-19 large-array "
            f"configuration (Table 5.2.2.2.1a-1: {sorted(SUPPORTED_N1N2_R19)})"
        )
    if antenna.P not in allowed_ports:
        raise ValueError(
            f"P_CSI-RS={antenna.P} unsupported; this codebook covers {allowed_ports} ports"
        )


def _as_ri_restriction(ri_restriction: np.ndarray | None) -> np.ndarray:
    if ri_restriction is None:
        ri_restriction = np.ones(4, dtype=bool)
    r = np.asarray(ri_restriction, dtype=bool)
    if r.shape != (4,):
        raise ValueError("RI restriction r = [r0..r3] must have 4 bits")
    return r


def _rank_gt2_allowed(r: np.ndarray) -> bool:
    """True if the RI restriction permits any rank > 2 (r_i = 1 for some i > 1)."""
    return bool(r[2] or r[3])


class RefinedEType2Codebook(R16Type2Codebook):
    """TS 38.214 5.2.2.2.5a 'eTypeII-r19' -- R16 eType II on 48/64/128 ports."""

    SUPPORTED_PORTS = (48, 64, 128)

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        param_combination: int = 4,
        R: int = 1,
        ri_restriction: np.ndarray | None = None,
    ) -> None:
        _require_r19_array(antenna, self.SUPPORTED_PORTS)
        r = _as_ri_restriction(ri_restriction)
        # paramCombination 7, 8 (the high-overhead L=6 rows) are barred when a
        # rank > 2 is permitted or when R = 2.
        if param_combination in (7, 8) and (R == 2 or _rank_gt2_allowed(r)):
            raise ValueError(
                "paramCombination-r19 7/8 require R=1 and ranks 3,4 disallowed "
                "(typeII-RI-Restriction-r19)"
            )
        super().__init__(
            antenna, N3, param_combination=param_combination, R=R, ri_restriction=r
        )
        self.name = "R19 Refined eType II"

    def select(self, H: np.ndarray, rank: int = 1):
        if 1 <= rank <= 4 and not self.ri_restriction[rank - 1]:
            raise ValueError(
                f"rank {rank} prohibited by typeII-RI-Restriction-r19 (r{rank - 1}=0)"
            )
        return super().select(H, rank)


class RefinedFeType2PortSelectionCodebook(R17Type2Codebook):
    """TS 38.214 5.2.2.2.9a 'typeII-FePortSelection-r19' -- R17 feType II PS on
    48/64 ports."""

    SUPPORTED_PORTS = (48, 64)

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        param_combination: int = 7,
        N_window: int = 4,
        ri_restriction: np.ndarray | None = None,
    ) -> None:
        _require_r19_array(antenna, self.SUPPORTED_PORTS)
        # The UE is not expected to be configured with paramCombination-r19 = 8.
        if param_combination == 8:
            raise ValueError(
                "paramCombination-r19=8 is not supported for the refined feType II "
                "port-selection codebook (5.2.2.2.9a)"
            )
        super().__init__(
            antenna,
            N3,
            param_combination=param_combination,
            N_window=N_window,
            ri_restriction=_as_ri_restriction(ri_restriction),
        )
        self.name = "R19 Refined feType II PS"

    def select(self, H: np.ndarray, rank: int = 1):
        if 1 <= rank <= 4 and not self.ri_restriction[rank - 1]:
            raise ValueError(
                f"rank {rank} prohibited by typeII-PortSelectionRI-Restriction-r19 "
                f"(r{rank - 1}=0)"
            )
        return super().select(H, rank)


class RefinedPredictedEType2Codebook(R18Type2Codebook):
    """TS 38.214 5.2.2.2.11a 'typeII-Doppler-r19' -- R18 predicted eType II on
    48/64/128 ports."""

    SUPPORTED_PORTS = (48, 64, 128)

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        N4: int = 4,
        param_combination: int = 3,
        R: int = 1,
        ri_restriction: np.ndarray | None = None,
    ) -> None:
        _require_r19_array(antenna, self.SUPPORTED_PORTS)
        r = _as_ri_restriction(ri_restriction)
        # paramCombination-Doppler 8, 9 (L=6) barred when rank > 2 permitted or R = 2.
        if param_combination in (8, 9) and (R == 2 or _rank_gt2_allowed(r)):
            raise ValueError(
                "paramCombination-Doppler-r19 8/9 require R=1 and ranks 3,4 disallowed "
                "(typeII-Doppler-RI-Restriction-r19)"
            )
        super().__init__(
            antenna,
            N3,
            N4=N4,
            param_combination=param_combination,
            R=R,
            ri_restriction=r,
        )
        self.name = "R19 Refined eType II predicted"
