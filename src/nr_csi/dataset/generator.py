"""Core CDL dataset generator.

For each antenna config it generates the config's share of samples in GPU
batches (one :meth:`SionnaCDLChannel.generate_batch` call per batch).  Each batch
draws a single ``(profile, delay_spread, ue_speed)`` triple from the configured
ranges -- fixed within the batch, varied across batches -- so the dataset spans
the diversity ranges while keeping per-sample provenance exact and the expensive
Sionna channel object reused across a whole batch.  Channels are stored clean and
energy-normalized; AWGN is a train-time augmentation (see :mod:`.preprocess`).

Reproducibility: NumPy draws the per-batch parameters from a seed derived from
``cfg.seed`` and the config index; the TF/Sionna RNG is seeded per batch the same
way ``scripts/compare/sionna_cdl_compare.py`` seeds its frozen bank.
"""

from __future__ import annotations

import pathlib

import numpy as np

from ..channel.sionna_adapter import SionnaCDLChannel
from . import io
from .config import DatasetConfig, config_tag, samples_per_config
from .preprocess import power_normalize


def seed_sionna(seed: int) -> None:
    """Seed the TF/Sionna RNG (mirrors the frozen-bank helper in compare/)."""
    import tensorflow as tf

    tf.random.set_seed(seed)
    try:
        from sionna.phy import config as sionna_config

        sionna_config.seed = seed
    except Exception:  # pragma: no cover - older/newer Sionna layouts
        pass


def _progress(iterable, total, enable, desc):
    if not enable:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        return iterable


def generate_dataset(cfg: DatasetConfig, progress: bool = True) -> dict:
    """Generate the full dataset described by ``cfg`` and write it to disk.

    Returns the written manifest dict.  Requires the optional ``sionna`` extra
    (TensorFlow) and ``h5py``."""
    out_dir = pathlib.Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = (np.asarray(cfg.profile_weights, float) / sum(cfg.profile_weights)
               if cfg.profile_weights else None)
    per_cfg = samples_per_config(cfg.n_samples, len(cfg.antennas))

    config_entries: list[dict] = []
    for ci, (ant, n_cfg) in enumerate(zip(cfg.antennas, per_cfg)):
        tag = config_tag(ant)
        cdir = out_dir / tag
        param_rng = np.random.default_rng([cfg.seed, ci])
        attrs = {
            "config_id": ci, "tag": tag,
            "N1": ant.N1, "N2": ant.N2, "O1": ant.O1, "O2": ant.O2,
            "Ng": ant.Ng, "P": ant.P, "n_rx": cfg.n_rx,
            "carrier_frequency": cfg.carrier_frequency,
            "subcarrier_spacing": cfg.subcarrier_spacing,
            "fft_size": cfg.fft_size, "n_freq": cfg.n_freq, "seed": cfg.seed,
        }

        buf_H: list[np.ndarray] = []
        buf = {"norm": [], "cdl_model": [], "delay_spread_ns": [], "ue_speed_kmh": []}
        buffered = 0
        shard_idx = 0
        shards: list[str] = []

        def flush() -> None:
            nonlocal buffered, shard_idx
            if buffered == 0:
                return
            path = cdir / f"shard_{shard_idx:05d}.h5"
            io.write_shard(
                path, np.concatenate(buf_H),
                {"norm": np.concatenate(buf["norm"]),
                 "cdl_model": np.concatenate(buf["cdl_model"]),
                 "delay_spread_ns": np.concatenate(buf["delay_spread_ns"]),
                 "ue_speed_kmh": np.concatenate(buf["ue_speed_kmh"])},
                attrs,
            )
            shards.append(str(path.relative_to(out_dir)))
            buf_H.clear()
            for v in buf.values():
                v.clear()
            buffered = 0
            shard_idx += 1

        produced = 0
        batch_no = 0
        n_batches = int(np.ceil(n_cfg / cfg.batch_size))
        bar = _progress(range(n_batches), n_batches, progress, f"{tag}")
        for _ in bar:
            if produced >= n_cfg:
                break
            n_this = min(cfg.batch_size, n_cfg - produced)
            model = str(param_rng.choice(list(cfg.profiles), p=weights))
            ds_ns = float(param_rng.uniform(*cfg.delay_spread_ns))
            speed = float(param_rng.uniform(*cfg.ue_speed_kmh))
            seed_sionna(cfg.seed + ci * 1_000_000 + batch_no)
            chan = SionnaCDLChannel(
                ant, N3=cfg.n_freq, model=model,
                carrier_frequency=cfg.carrier_frequency,
                delay_spread=ds_ns * 1e-9, n_rx=cfg.n_rx,
                ue_speed_kmh=speed, fft_size=cfg.fft_size,
                subcarrier_spacing=cfg.subcarrier_spacing,
            )
            Hb = chan.generate_batch(n_this, n_slots=1)  # (n,1,n_freq,rx,port)
            Hb = Hb[:, 0].transpose(0, 2, 3, 1)          # (n, rx, port, n_freq)
            Hn, norm = power_normalize(Hb)               # per-sample (rx,port,freq)
            buf_H.append(Hn.astype(np.complex64))
            buf["norm"].append(norm.reshape(n_this).astype(np.float32))
            buf["cdl_model"].append(np.array([model] * n_this, dtype="S1"))
            buf["delay_spread_ns"].append(np.full(n_this, ds_ns, dtype=np.float32))
            buf["ue_speed_kmh"].append(np.full(n_this, speed, dtype=np.float32))
            buffered += n_this
            produced += n_this
            batch_no += 1
            if buffered >= cfg.shard_size:
                flush()
        flush()

        config_entries.append({
            "config_id": ci, "tag": tag,
            "N1": ant.N1, "N2": ant.N2, "O1": ant.O1, "O2": ant.O2,
            "Ng": ant.Ng, "P": ant.P,
            "n_samples": produced, "shards": shards,
        })

    return io.write_manifest(out_dir, cfg, config_entries)
