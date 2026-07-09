"""Fit the PRISM mixture codecs from angle-delay eigen targets.

Training rows: the A/B/C GLIMPSE target file (``data/ml/targets_p32.npz``,
train split) plus the D/E dataset's targets computed here WITH per-sample
profile labels (so the no-E ablation can exclude E).  Fits:

* ``prism_p32_K{k}`` for K in the sweep (K=1 is the pooled "broad GLIMPSE"
  ablation -- one basis fitted on the A-E mix);
* ``prism_p32_K4_noE`` -- the OOD-honesty variant fitted on A/B/C/D only,
  evaluated on the unseen profile E.

Prints per-regime capture diagnostics: what fraction of each profile group's
eigenvector energy the mixture's best basis captures at m_ref, versus the
single-basis codecs (GLIMPSE's A/B/C KLT and the pooled K=1).

    .venv/bin/python scripts/prism/fit_prism.py --out models
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nr_csi.config import AntennaConfig  # noqa: E402
from nr_csi.dataset import io  # noqa: E402
from nr_csi.dataset.splits import split_indices  # noqa: E402
from nr_csi.ml.projection import GlimpseCodec  # noqa: E402
from nr_csi.prism.mixture import PrismCodec, captured_energy, fit_mixture  # noqa: E402

# reuse the pinned target-extraction helpers from the GLIMPSE script
_spec = importlib.util.spec_from_file_location(
    "prepare_targets", REPO / "scripts" / "ml" / "prepare_targets.py")
_pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pt)


def labelled_targets(dataset_dir: pathlib.Path, n3: int):
    """Angle-delay eigen vectors + profile labels for every stored channel,
    train split only (same split convention as prepare_targets)."""
    manifest = io.read_manifest(dataset_dir)
    entry = manifest["configs"][0]
    ant = AntennaConfig.standard(entry["N1"], entry["N2"])
    codec = GlimpseCodec(ant, n3)
    gs, labels = [], []
    for shard_rel in entry["shards"]:
        shard = io.read_shard(dataset_dir / shard_rel)
        H = _pt.reduce_freq(shard["H"], n3)
        V = _pt.batched_aligned_targets(H, rank=1)
        gs.append(_pt.batched_to_vec(codec, V)[:, 0].astype(np.complex64))
        labels.append(np.asarray(
            [m.decode() if isinstance(m, bytes) else str(m)
             for m in shard["cdl_model"]]))
    g = np.concatenate(gs)
    labels = np.concatenate(labels)
    cfg = manifest["config"]
    idx = split_indices(len(g), tuple(cfg["splits"]), cfg["seed"],
                        cfg["split_method"])
    return g[idx["train"]], labels[idx["train"]]


def capture_report(tag, bases, groups, m_ref):
    for name, gg in groups.items():
        e = np.max(np.stack([captured_energy(A, gg, m_ref) for A in bases], 1), 1)
        print(f"  {tag:>14} capture@{m_ref} on {name}: {float(np.mean(e)):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abc-targets", type=pathlib.Path,
                    default=pathlib.Path("data/ml/targets_p32.npz"))
    ap.add_argument("--de-dataset", type=pathlib.Path,
                    default=pathlib.Path("data/cdl_p32_de"))
    ap.add_argument("--geometry", default="4x4x8", help="N1xN2xN3")
    ap.add_argument("--m-max", type=int, default=64)
    ap.add_argument("--m-ref", type=int, default=16)
    ap.add_argument("--k-sweep", default="1,2,4,8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("models"))
    args = ap.parse_args()

    n1, n2, n3 = (int(x) for x in args.geometry.split("x"))
    ant = AntennaConfig.standard(n1, n2)

    with np.load(args.abc_targets) as z:
        g_abc = z["train/g_clean_l1"]
    g_de, lab_de = labelled_targets(args.de_dataset, n3)
    print(f"rows: A/B/C {len(g_abc):,}  D {np.sum(lab_de == 'D'):,}  "
          f"E {np.sum(lab_de == 'E'):,}")

    g_all = np.concatenate([g_abc, g_de])
    groups = {"A/B/C": g_abc, "D": g_de[lab_de == "D"], "E": g_de[lab_de == "E"]}

    # reference: GLIMPSE's original A/B/C-only basis on each group
    glimpse = GlimpseCodec.load("models/glimpse_p32_codec")
    capture_report("GLIMPSE(ABC)", [glimpse.A], groups, args.m_ref)

    for k in (int(s) for s in args.k_sweep.split(",")):
        bases, sigmas, assign, obj = fit_mixture(
            g_all, k, m_max=args.m_max, m_ref=args.m_ref, seed=args.seed)
        codec = PrismCodec(ant, n3, bases=tuple(bases), sigmas=tuple(sigmas))
        path = args.out / f"prism_p32_K{k}"
        codec.save(path)
        print(f"K={k}: objective {obj:.4f} -> {path}.npz")
        capture_report(f"PRISM K={k}", bases, groups, args.m_ref)
        if k > 1:  # cluster composition: does it discover the LoS/NLoS split?
            src = np.array(["ABC"] * len(g_abc) + list(lab_de))
            for c in range(k):
                members = src[assign == c]
                frac = {p: float(np.mean(members == p)) for p in ("ABC", "D", "E")}
                print(f"    cluster {c}: n={len(members):,} "
                      + " ".join(f"{p}={v:.2f}" for p, v in frac.items()))

    # OOD-honesty variant: never sees profile E
    g_noe = np.concatenate([g_abc, g_de[lab_de == "D"]])
    bases, sigmas, _, obj = fit_mixture(
        g_noe, 4, m_max=args.m_max, m_ref=args.m_ref, seed=args.seed)
    PrismCodec(ant, n3, bases=tuple(bases), sigmas=tuple(sigmas)).save(
        args.out / "prism_p32_K4_noE")
    print(f"K=4 noE: objective {obj:.4f} -> {args.out}/prism_p32_K4_noE.npz")
    capture_report("PRISM noE", bases, groups, args.m_ref)


if __name__ == "__main__":
    main()
