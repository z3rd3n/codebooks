"""Precompute GLIMPSE training targets from a raw-H CDL dataset.

For every stored channel: reduce to the ``N3`` PMI frequency units (the same
contiguous averaging as the Sionna adapter), compute rank-2 phase-aligned
eigen targets -- exactly what :func:`nr_csi.codebooks._spatial.
aligned_eigen_targets` produces, vectorized over samples -- and store their
angle-delay vectors ``g`` for:

* layer 1, clean channel (also *the* rank-1 target);
* layer 1 from the channel + measurement AWGN at 20 / 10 dB (inputs for
  noise-robust training; the training target stays the clean vector);
* layer 2, clean (rank-2 support).

Output: ``<out>.npz`` with train/val splits taken from the dataset manifest.

    .venv/bin/python scripts/ml/prepare_targets.py data/cdl_p32 \
        --n3 8 --out data/ml/targets_p32
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nr_csi.baselines.ideal import eigen_precoder  # noqa: E402
from nr_csi.codebooks._spatial import aligned_eigen_targets  # noqa: E402
from nr_csi.config import AntennaConfig  # noqa: E402
from nr_csi.dataset import io  # noqa: E402
from nr_csi.dataset.preprocess import apply_awgn  # noqa: E402
from nr_csi.dataset.splits import split_indices  # noqa: E402
from nr_csi.ml.projection import GlimpseCodec  # noqa: E402


def batched_aligned_targets(H: np.ndarray, rank: int) -> np.ndarray:
    """Vectorized ``aligned_eigen_targets``: ``H[n, N3, Nr, P] -> V[n, N3, P, v]``."""
    V = eigen_precoder(H, rank=rank) * np.sqrt(rank)  # unit columns
    for t in range(1, V.shape[1]):
        inner = np.sum(V[:, t - 1].conj() * V[:, t], axis=1, keepdims=True)  # (n, 1, v)
        V[:, t] *= np.exp(-1j * np.angle(np.where(inner == 0, 1.0, inner)))
    return V


def batched_to_vec(codec: GlimpseCodec, V: np.ndarray) -> np.ndarray:
    """Vectorized ``codec.targets_to_vec``: ``V[n, N3, P, v] -> g[n, v, D]``."""
    from nr_csi.dataset.preprocess import spatial_dft, to_delay

    X = V.transpose(0, 3, 2, 1)  # (n, v, P, N3)
    G = to_delay(spatial_dft(X, codec.antenna))
    return G.reshape(V.shape[0], V.shape[3], codec.D)


def reduce_freq(H: np.ndarray, N3: int) -> np.ndarray:
    """``(n, rx, P, n_freq) -> (n, N3, rx, P)`` by contiguous averaging
    (identical to the Sionna adapter's PMI-unit slicing)."""
    n, rx, p, nf = H.shape
    g = nf // N3
    Hr = H[..., : g * N3].reshape(n, rx, p, N3, g).mean(-1)
    return Hr.transpose(0, 3, 1, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=pathlib.Path)
    ap.add_argument("--config", default=None, help="config tag (default: first)")
    ap.add_argument("--n3", type=int, default=8)
    ap.add_argument("--noise-snrs", default="20,10", help="measurement-noise variants (dB)")
    ap.add_argument("--seed", type=int, default=99, help="AWGN augmentation seed")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/ml/targets"))
    args = ap.parse_args()

    manifest = io.read_manifest(args.dataset_dir)
    entry = (next(e for e in manifest["configs"] if e["tag"] == args.config)
             if args.config else manifest["configs"][0])
    ant = AntennaConfig.standard(entry["N1"], entry["N2"])
    codec = GlimpseCodec(ant, args.n3)
    snrs = [float(s) for s in args.noise_snrs.split(",")]
    rng = np.random.default_rng(args.seed)

    keys = ["g_clean_l1", "g_clean_l2"] + [f"g_n{int(s)}_l1" for s in snrs]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    checked = False
    for shard_rel in entry["shards"]:
        shard = io.read_shard(args.dataset_dir / shard_rel)
        H = reduce_freq(shard["H"], args.n3)  # (n, N3, rx, P)
        V = batched_aligned_targets(H, rank=2)
        if not checked:  # pin the vectorized path to the reference function
            V_ref = aligned_eigen_targets(H[0], 2)
            np.testing.assert_allclose(V[0], V_ref, atol=1e-10)
            g_ref = codec.targets_to_vec(V_ref)
            np.testing.assert_allclose(batched_to_vec(codec, V[:1])[0], g_ref, atol=1e-10)
            checked = True
        g = batched_to_vec(codec, V)  # (n, 2, D)
        parts["g_clean_l1"].append(g[:, 0].astype(np.complex64))
        parts["g_clean_l2"].append(g[:, 1].astype(np.complex64))
        for s in snrs:
            Hn = apply_awgn(H, s, rng)
            gn = batched_to_vec(codec, batched_aligned_targets(Hn, rank=1))
            parts[f"g_n{int(s)}_l1"].append(gn[:, 0].astype(np.complex64))
        print(f"  {shard_rel}: {H.shape[0]} samples")

    full = {k: np.concatenate(v) for k, v in parts.items()}
    n = full["g_clean_l1"].shape[0]
    cfg = manifest["config"]
    idx = split_indices(n, tuple(cfg["splits"]), cfg["seed"], cfg["split_method"])
    out = {}
    for split in ("train", "val"):
        for k, v in full.items():
            out[f"{split}/{k}"] = v[idx[split]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out.with_suffix(".npz"), **out)
    meta = dict(N1=ant.N1, N2=ant.N2, P=ant.P, N3=args.n3, D=codec.D,
                n_train=len(idx["train"]), n_val=len(idx["val"]),
                noise_snrs=snrs, dataset=str(args.dataset_dir), tag=entry["tag"])
    print("wrote", args.out.with_suffix(".npz"), meta)


if __name__ == "__main__":
    main()
