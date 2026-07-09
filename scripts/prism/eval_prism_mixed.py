"""Evaluate PRISM and the 3GPP codebooks on a MIXED-profile frozen bank.

A real cell serves line-of-sight and non-line-of-sight users side by side, so
the deployment-level comparison is on a channel population that mixes the CDL
profiles.  This script builds a frozen mixed bank -- ``drops_per_profile``
Sionna drops from each of CDL-A/B/C/D/E, generated exactly like the
per-profile banks (same seed, same parameters) and interleaved
deterministically -- then scores the full codebook roster and the PRISM
variants on it.

Writes ``results/prism/frontier_mixed.json`` with the same row schema as the
per-profile files.

    .venv/bin/python scripts/prism/eval_prism_mixed.py --out results/prism
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nr_csi.config import AntennaConfig  # noqa: E402
from nr_csi.prism import PrismCodec  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_glimpse", REPO / "scripts" / "ml" / "eval_glimpse.py")
_eg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eg)

_pspec = importlib.util.spec_from_file_location(
    "eval_prism", REPO / "scripts" / "prism" / "eval_prism.py")
_ep = importlib.util.module_from_spec(_pspec)
_pspec.loader.exec_module(_ep)


def make_mixed_bank(ant, n3, profiles, drops_per_profile, seed):
    """Frozen mixed bank: per-profile banks built exactly like the
    single-profile evals, then interleaved A,B,C,D,E,A,B,... so every scheme
    sees the same deterministic sequence."""
    banks = [_eg.make_bank(ant, n3, m, drops_per_profile, seed).bank
             for m in profiles]
    interleaved = np.stack(banks, axis=1)  # (drops, profiles, ...)
    mixed = interleaved.reshape(-1, *banks[0].shape[1:])
    return _eg.FrozenBank(mixed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="4x4x8")
    ap.add_argument("--profiles", default="A,B,C,D,E")
    ap.add_argument("--drops-per-profile", type=int, default=40)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--models", type=pathlib.Path, default=pathlib.Path("models"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/prism"))
    args = ap.parse_args()

    n1, n2, n3 = (int(x) for x in args.geometry.split("x"))
    ant = AntennaConfig.standard(n1, n2)
    profiles = args.profiles.split(",")
    n_drops = args.drops_per_profile * len(profiles)
    print(f"=== mixed bank: {args.drops_per_profile} drops x {profiles} "
          f"= {n_drops} drops ===")
    bank = make_mixed_bank(ant, n3, profiles, args.drops_per_profile, args.seed)

    codecs = {
        "PRISM K4": PrismCodec.load(args.models / "prism_p32_K4"),
        "PRISM K1 (pooled)": PrismCodec.load(args.models / "prism_p32_K1"),
        "PRISM K2": PrismCodec.load(args.models / "prism_p32_K2"),
        "PRISM K8": PrismCodec.load(args.models / "prism_p32_K8"),
        "PRISM K4 (no-E)": PrismCodec.load(args.models / "prism_p32_K4_noE"),
    }
    roster = _eg.codebook_roster(ant, n3) + _ep.prism_roster(ant, n3, codecs)
    rows = _eg.run_roster(roster, bank, args.rank, n_drops)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = dict(
        geometry=dict(N1=n1, N2=n2, P=ant.P, N3=n3),
        cdl_model="mixed(" + ",".join(profiles) + ")",
        drops=n_drops, drops_per_profile=args.drops_per_profile,
        rank=args.rank, seed=args.seed, measurement_snr_db=None,
        model=str(args.models), points=rows,
    )
    out = args.out / "frontier_mixed.json"
    out.write_text(json.dumps(payload, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
