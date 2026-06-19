"""Generate a CDL channel dataset for AI/ML CSI-feedback compression.

Writes sharded HDF5 + a ``manifest.json`` under ``--out`` (see
``nr_csi.dataset`` and the project plan).  Channels are stored clean and
energy-normalized; SNR is a train-time augmentation, recorded only.

Requires the optional extras::

    pip install -e ".[sionna,dataset]"

Examples::

    # tiny CPU smoke test (one 16-port config, a few hundred samples)
    python scripts/dataset/generate_cdl_dataset.py \
        --n-samples 200 --configs 4x2 --profiles A,B,C --out data/cdl_smoke

    # the locked NLOS generalization set (~100k, GPU)
    python scripts/dataset/generate_cdl_dataset.py \
        --configs 4x2,16x1,16x2 --n-samples 100000 --out data/cdl_nlos
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF before any import

import argparse  # noqa: E402

from nr_csi.dataset import DatasetConfig, generate_dataset, parse_antenna  # noqa: E402


def _range(s: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in s.split(","))
    return (lo, hi)


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--configs", default="4x2,16x1,16x2",
                   help="comma list of N1xN2 arrays (default the 16/32/64-port set)")
    p.add_argument("--profiles", default="A,B,C", help="comma list of CDL profiles")
    p.add_argument("--n-samples", type=int, default=100_000)
    p.add_argument("--n-rx", type=int, default=2)
    p.add_argument("--freq-res", type=int, default=256, dest="n_freq",
                   help="frequency bins stored per sample (adapter N3)")
    p.add_argument("--fft-size", type=int, default=512)
    p.add_argument("--subcarrier-spacing", type=float, default=30e3)
    p.add_argument("--carrier-frequency", type=float, default=3.5e9)
    p.add_argument("--batch-size", type=int, default=256, help="Sionna drops per call")
    p.add_argument("--shard-size", type=int, default=5_000)
    p.add_argument("--ds-range", type=_range, default=(30.0, 300.0),
                   help="delay-spread range in ns, 'lo,hi'")
    p.add_argument("--speed-range", type=_range, default=(3.0, 30.0),
                   help="UE speed range in km/h, 'lo,hi'")
    p.add_argument("--snr-range", type=_range, default=(0.0, 30.0),
                   help="intended SNR range in dB (recorded only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/cdl_dataset")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = cli()
    antennas = [parse_antenna(s) for s in args.configs.split(",")]
    cfg = DatasetConfig(
        antennas=antennas,
        profiles=tuple(args.profiles.split(",")),
        delay_spread_ns=args.ds_range,
        ue_speed_kmh=args.speed_range,
        snr_db=args.snr_range,
        n_rx=args.n_rx,
        carrier_frequency=args.carrier_frequency,
        subcarrier_spacing=args.subcarrier_spacing,
        fft_size=args.fft_size,
        n_freq=args.n_freq,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        seed=args.seed,
        out_dir=args.out,
    )
    print(f"Generating {cfg.n_samples} samples across "
          f"{[a.P for a in antennas]} ports, profiles {cfg.profiles} -> {args.out}")
    manifest = generate_dataset(cfg, progress=not args.no_progress)
    print(f"\nDone: {manifest['total_samples']} samples in "
          f"{len(manifest['configs'])} configs.")
    for e in manifest["configs"]:
        print(f"  {e['tag']:>12}: {e['n_samples']:>7} samples, {len(e['shards'])} shards")
    print(f"Manifest: {args.out}/manifest.json")


if __name__ == "__main__":
    main()
