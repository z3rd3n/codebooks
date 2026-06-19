"""Leakage-safe train/val/test splits.

Independent single-slot CDL drops are i.i.d., so a deterministic *random* split
by index is statistically sound (the default).  Multi-slot trajectories are
temporally correlated -- a random split then leaks near-duplicate neighbours
across the train/test boundary and inflates the reported score -- so for those
use ``method='block'``, which assigns contiguous index blocks to each split.

Splits are derived deterministically from ``(n, fractions, seed, method)`` and
stored as parameters in the manifest rather than as giant index lists, so a
loader can reproduce them exactly.
"""

from __future__ import annotations

import numpy as np

SPLIT_NAMES = ("train", "val", "test")


def split_indices(
    n: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    method: str = "random",
) -> dict[str, np.ndarray]:
    """Return disjoint ``{'train','val','test'}`` index arrays over ``range(n)``.

    Sizes are ``round(frac * n)`` with any rounding remainder absorbed by train.
    ``method='random'`` shuffles deterministically with ``seed``; ``'block'``
    keeps contiguous order (train | val | test)."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions {fractions} must sum to 1.0")
    if method not in ("random", "block"):
        raise ValueError("method must be 'random' or 'block'")

    n_val = int(round(fractions[1] * n))
    n_test = int(round(fractions[2] * n))
    n_train = n - n_val - n_test
    if n_train < 0:
        raise ValueError(f"fractions allocate more than n={n} samples")

    idx = np.arange(n)
    if method == "random":
        np.random.default_rng(seed).shuffle(idx)
    return {
        "train": np.sort(idx[:n_train]),
        "val": np.sort(idx[n_train:n_train + n_val]),
        "test": np.sort(idx[n_train + n_val:]),
    }
