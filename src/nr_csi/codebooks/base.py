"""Common interface implemented by every codebook (and by future ML schemes).

Conventions
-----------
* Channel tensors are ``H[slot, t, rx, port]`` with ``t = 0..N3-1`` the PMI
  frequency unit ("subband" for R=1) and ``slot`` the time interval axis
  (length 1 for codebooks without Doppler compression).
* ``precoder`` returns ``W[interval, t, port, layer]``; every layer column has
  unit norm and the rank-v matrix carries the spec's 1/sqrt(v) scaling, so
  tr(W^H W) = 1 per (interval, t).
* ``select`` is the UE side (channel -> PMI), ``precoder`` is the gNB side
  (PMI -> precoding matrices, pure 38.214 reconstruction).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class CodebookScheme(ABC):
    """A configured CSI feedback scheme. ML schemes implement the same API."""

    name: str = "scheme"

    @abstractmethod
    def select(self, H: np.ndarray, rank: int = 1) -> Any:
        """UE side: pick the PMI for channel ``H[slot, t, rx, port]``."""

    @abstractmethod
    def precoder(self, pmi: Any) -> np.ndarray:
        """gNB side: reconstruct ``W[interval, t, port, layer]`` from a PMI."""

    @abstractmethod
    def overhead_bits(self, pmi: Any) -> dict[str, int]:
        """Feedback cost in bits, per PMI information element."""

    def total_overhead_bits(self, pmi: Any) -> int:
        return sum(self.overhead_bits(pmi).values())


def normalize_columns(W: np.ndarray) -> np.ndarray:
    """Unit-normalize the last-axis-but-one columns of W[..., port, layer]."""
    norms = np.linalg.norm(W, axis=-2, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return W / norms
