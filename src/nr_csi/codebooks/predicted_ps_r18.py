"""R18 further-enhanced Type II PS predicted-PMI codebook (N4 = 1)."""

from __future__ import annotations

import numpy as np

from ..config import AntennaConfig
from .fetype2_r17 import R17Type2Codebook


class R18PredictedPortSelectionCodebook(R17Type2Codebook):
    """TS 38.214 section 5.2.2.2.11, reconstructed using the R17 table."""

    name = "R18 FeType II PS predicted PMI"
    N4 = 1

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        param_combination: int = 7,
        N_window: int = 4,
        R: int = 1,
        rank_restriction: np.ndarray | None = None,
    ) -> None:
        super().__init__(antenna, N3, param_combination, N_window)
        if R not in (1, 2):
            raise ValueError("R must be 1 or 2")
        if self.M == 1 and R != 1:
            raise ValueError("R=2 is not configured when M=1")
        self.R = R
        if rank_restriction is None:
            rank_restriction = np.ones(4, dtype=bool)
        self.rank_restriction = np.asarray(rank_restriction, dtype=bool)
        if self.rank_restriction.shape != (4,):
            raise ValueError("predicted-PS RI restriction must have 4 bits")

    def select(self, H: np.ndarray, rank: int = 1):
        if not 1 <= rank <= 4:
            raise ValueError("R18 predicted-PS supports ranks 1-4")
        if not self.rank_restriction[rank - 1]:
            raise ValueError(f"rank {rank} prohibited by predicted-PS RI restriction")
        return super().select(H, rank)
