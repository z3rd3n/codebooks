"""Evaluate PRISM against the stored codebook + GLIMPSE frontiers.

Reuses ``eval_glimpse.py``'s frozen-bank construction verbatim (same Sionna
seed, same parameters) so PRISM rows are directly comparable to the stored
``results/ml/frontier_cdl*.json`` rows -- and *proves* it by re-scoring one
stored GLIMPSE (learned) row per bank and asserting the SGCS matches the JSON
to 1e-9 before evaluating anything new.

Writes ``results/prism/frontier_cdl{M}.json``: the stored rows merged with

* ``PRISM K4``          -- the headline mixture, LS decode, (m, B) grid;
* ``PRISM K1 (pooled)`` -- single KLT on the A-E mix (the averaging-penalty
  ablation, i.e. "broad GLIMPSE");
* ``PRISM K2`` / ``PRISM K8`` -- mixture-size sweep (primary CDL only);
* ``PRISM K4 (no-E)``   -- fitted without profile E (OOD honesty, E bank only).

    .venv/bin/python scripts/prism/eval_prism.py --out results/prism
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
from nr_csi.prism import PrismCodec, PrismScheme  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_glimpse", REPO / "scripts" / "ml" / "eval_glimpse.py")
_eg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eg)

GRID = _eg.GLIMPSE_GRID


def verify_bank(bank, stored: dict, model_path: str, n_drops: int) -> None:
    """Re-score one stored GLIMPSE (learned) row on the rebuilt bank and
    assert it reproduces the JSON -- the frozen-drop determinism check."""
    from nr_csi.ml.decoder import KerasDecoder

    ant = AntennaConfig.standard(stored["geometry"]["N1"], stored["geometry"]["N2"])
    n3 = stored["geometry"]["N3"]
    ref = next(r for r in stored["points"]
               if r["family"] == "GLIMPSE (learned)" and r["config"] == "m16-B3")
    codec, decs = _eg.load_codec_and_decoders(
        ant, n3, pathlib.Path(model_path), 64, 0, ("learned",))
    roster = _eg.glimpse_roster(ant, n3, codec, decs, grid=[(16, 3)])
    row = _eg.run_roster(roster, bank, stored["rank"], n_drops)[0]
    if abs(row["sgcs"] - ref["sgcs"]) > 1e-9:
        raise AssertionError(
            f"bank determinism check FAILED: re-scored {row['sgcs']!r} vs "
            f"stored {ref['sgcs']!r}")
    print(f"  bank verified: GLIMPSE m16-B3 sgcs {row['sgcs']:.6f} == stored")


def prism_roster(ant, n3, codecs: dict[str, PrismCodec], grid=GRID):
    roster = []
    for fam, codec in codecs.items():
        for m, b in grid:
            if m > codec.m_max:
                continue
            roster.append((PrismScheme(ant, n3, codec, m=m, bits=b),
                           "antenna", fam, f"m{m}-B{b}"))
    return roster


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stored", type=pathlib.Path, default=pathlib.Path("results/ml"))
    ap.add_argument("--glimpse-model", default="models/glimpse_p32")
    ap.add_argument("--models", type=pathlib.Path, default=pathlib.Path("models"))
    ap.add_argument("--cdl", default="A,B,C,D,E")
    ap.add_argument("--primary", default="C", help="bank that also gets the K sweep")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/prism"))
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for cdl_model in args.cdl.split(","):
        stored = json.loads(
            (args.stored / f"frontier_cdl{cdl_model}.json").read_text())
        g = stored["geometry"]
        ant = AntennaConfig.standard(g["N1"], g["N2"])
        n3, n_drops, rank = g["N3"], stored["drops"], stored["rank"]
        print(f"=== CDL-{cdl_model}: {n_drops} drops (matching stored file) ===")
        bank = _eg.make_bank(ant, n3, cdl_model, n_drops, stored["seed"])
        if not args.skip_verify:
            verify_bank(bank, stored, args.glimpse_model, n_drops)

        codecs = {
            "PRISM K4": PrismCodec.load(args.models / "prism_p32_K4"),
            "PRISM K1 (pooled)": PrismCodec.load(args.models / "prism_p32_K1"),
        }
        if cdl_model == args.primary:
            codecs["PRISM K2"] = PrismCodec.load(args.models / "prism_p32_K2")
            codecs["PRISM K8"] = PrismCodec.load(args.models / "prism_p32_K8")
        if cdl_model == "E":
            codecs["PRISM K4 (no-E)"] = PrismCodec.load(
                args.models / "prism_p32_K4_noE")

        rows = _eg.run_roster(prism_roster(ant, n3, codecs), bank, rank, n_drops)
        merged = dict(stored)
        merged["points"] = stored["points"] + rows
        merged["prism_models"] = {k: str(args.models) for k in codecs}
        out = args.out / f"frontier_cdl{cdl_model}.json"
        out.write_text(json.dumps(merged, indent=1))
        print("wrote", out)


if __name__ == "__main__":
    main()
