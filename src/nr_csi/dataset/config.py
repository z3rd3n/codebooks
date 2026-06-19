"""Configuration for CDL training-dataset generation.

A :class:`DatasetConfig` fully describes a dataset: which antenna geometries to
cover, which CDL profiles, the per-sample randomization ranges (delay spread, UE
speed, intended SNR), the OFDM grid, and how many samples / how to shard.  It is
JSON-serializable (see :meth:`to_dict`) so the exact recipe lands in the
``manifest.json`` next to the data.

The locked design choices for this project (raw-H ground truth, multi-config
generalization set, NLOS profiles A/B/C, ~100-200k samples) live in
:meth:`DatasetConfig.default_nlos_generalization`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AntennaConfig

# CDL profiles we treat as NLOS (rich angular spread).  D/E are near-LoS and
# excluded by default, but the generator accepts any subset.
NLOS_PROFILES: tuple[str, ...] = ("A", "B", "C")
ALL_PROFILES: tuple[str, ...] = ("A", "B", "C", "D", "E")


def parse_antenna(spec: str) -> AntennaConfig:
    """Parse a CLI ``"N1xN2"`` token (e.g. ``"16x1"``) into an ``AntennaConfig``.

    Uses ``AntennaConfig.standard`` so (O1,O2) follow TS 38.214 (incl. the R19
    large arrays); raises a clear error for an unsupported geometry.
    """
    try:
        n1_s, n2_s = spec.lower().split("x")
        n1, n2 = int(n1_s), int(n2_s)
    except ValueError as exc:  # pragma: no cover - CLI guard
        raise ValueError(f"antenna spec {spec!r} must look like '16x1'") from exc
    return AntennaConfig.standard(n1, n2)


def config_tag(ant: AntennaConfig) -> str:
    """Stable per-config directory/key name, e.g. ``'4x2_P16'``."""
    return f"{ant.N1}x{ant.N2}_P{ant.P}"


@dataclass
class DatasetConfig:
    """Everything the generator needs to produce a reproducible CDL dataset."""

    antennas: list[AntennaConfig]
    profiles: tuple[str, ...] = NLOS_PROFILES
    profile_weights: tuple[float, ...] | None = None  # None -> uniform
    delay_spread_ns: tuple[float, float] = (30.0, 300.0)
    ue_speed_kmh: tuple[float, float] = (3.0, 30.0)
    snr_db: tuple[float, float] = (0.0, 30.0)  # recorded only; AWGN is train-time aug
    n_rx: int = 2
    carrier_frequency: float = 3.5e9
    subcarrier_spacing: float = 30e3
    fft_size: int = 512
    n_freq: int = 256  # frequency bins stored per sample (= adapter N3)
    n_samples: int = 100_000
    batch_size: int = 256  # Sionna drops per generate_batch call
    shard_size: int = 5_000
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_method: str = "random"  # 'random' (i.i.d. drops) or 'block'
    seed: int = 0
    out_dir: str = "data/cdl_dataset"

    def __post_init__(self) -> None:
        if not self.antennas:
            raise ValueError("at least one antenna config is required")
        bad = [p for p in self.profiles if p not in ALL_PROFILES]
        if bad:
            raise ValueError(f"unknown CDL profiles {bad}; choose from {ALL_PROFILES}")
        if self.profile_weights is not None and len(self.profile_weights) != len(self.profiles):
            raise ValueError("profile_weights must match the number of profiles")
        if self.n_freq > self.fft_size:
            raise ValueError(f"n_freq={self.n_freq} cannot exceed fft_size={self.fft_size}")
        if abs(sum(self.splits) - 1.0) > 1e-6:
            raise ValueError(f"splits {self.splits} must sum to 1.0")
        if self.split_method not in ("random", "block"):
            raise ValueError("split_method must be 'random' or 'block'")
        if min(self.delay_spread_ns) <= 0 or self.delay_spread_ns[0] > self.delay_spread_ns[1]:
            raise ValueError("delay_spread_ns must be a positive (lo, hi) range")

    @classmethod
    def default_nlos_generalization(cls, **overrides) -> "DatasetConfig":
        """The locked recipe: 16/32/64-port arrays, CDL-A/B/C, randomized
        delay spread / speed / SNR.  ``overrides`` patch any field."""
        antennas = [
            AntennaConfig.standard(4, 2),   # 16 ports
            AntennaConfig.standard(16, 1),  # 32 ports
            AntennaConfig.standard(16, 2),  # 64 ports (R19 large array)
        ]
        params: dict = dict(antennas=antennas, profiles=NLOS_PROFILES)
        params.update(overrides)
        return cls(**params)

    def to_dict(self) -> dict:
        """JSON-serializable view (for the manifest)."""
        return {
            "antennas": [
                {"N1": a.N1, "N2": a.N2, "O1": a.O1, "O2": a.O2, "Ng": a.Ng, "P": a.P,
                 "tag": config_tag(a)}
                for a in self.antennas
            ],
            "profiles": list(self.profiles),
            "profile_weights": list(self.profile_weights) if self.profile_weights else None,
            "delay_spread_ns": list(self.delay_spread_ns),
            "ue_speed_kmh": list(self.ue_speed_kmh),
            "snr_db": list(self.snr_db),
            "n_rx": self.n_rx,
            "carrier_frequency": self.carrier_frequency,
            "subcarrier_spacing": self.subcarrier_spacing,
            "fft_size": self.fft_size,
            "n_freq": self.n_freq,
            "n_samples": self.n_samples,
            "batch_size": self.batch_size,
            "shard_size": self.shard_size,
            "splits": list(self.splits),
            "split_method": self.split_method,
            "seed": self.seed,
        }


def samples_per_config(n_samples: int, n_configs: int) -> list[int]:
    """Split ``n_samples`` as evenly as possible across configs (remainder first)."""
    base, rem = divmod(n_samples, n_configs)
    return [base + (1 if i < rem else 0) for i in range(n_configs)]
