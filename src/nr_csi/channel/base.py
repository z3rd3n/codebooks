"""Channel source interface.

A channel source produces frequency-domain MIMO channels on the PMI
reporting grid:  H[slot, t, rx, port]  with t = 0..N3-1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ChannelSource(ABC):
    n_rx: int
    n_ports: int
    N3: int

    @abstractmethod
    def generate(self, n_slots: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return H with shape (n_slots, N3, n_rx, n_ports)."""
