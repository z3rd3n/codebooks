"""CDL channel dataset generation for AI/ML CSI feedback compression.

Produces sharded HDF5 datasets of raw 3GPP TR 38.901 CDL channels (the ground
truth a learned CSI-feedback encoder/decoder is trained on), with the
preprocessing, I/O, and split helpers needed to consume them.  See
``scripts/dataset/`` for the CLI entry points and the project plan for the
design rationale.

Note: ``generate_dataset`` needs the optional ``sionna`` (TensorFlow) extra; the
preprocessing / I/O / split helpers are pure NumPy + optional ``h5py``.
"""

from __future__ import annotations

from .config import (
    ALL_PROFILES,
    NLOS_PROFILES,
    DatasetConfig,
    config_tag,
    parse_antenna,
    samples_per_config,
)
from .generator import generate_dataset
from .io import (
    iter_samples,
    load_config_array,
    read_manifest,
    read_shard,
    write_manifest,
    write_shard,
)
from .preprocess import (
    apply_awgn,
    from_angular_delay,
    power_normalize,
    spatial_dft,
    stack_real_imag,
    to_angular_delay,
    to_delay,
)
from .splits import split_indices

__all__ = [
    "ALL_PROFILES",
    "NLOS_PROFILES",
    "DatasetConfig",
    "config_tag",
    "parse_antenna",
    "samples_per_config",
    "generate_dataset",
    "load_config_array",
    "iter_samples",
    "read_manifest",
    "read_shard",
    "write_manifest",
    "write_shard",
    "apply_awgn",
    "from_angular_delay",
    "power_normalize",
    "spatial_dft",
    "stack_real_imag",
    "to_angular_delay",
    "to_delay",
    "split_indices",
]
