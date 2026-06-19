"""Reference: how a trained CSI-feedback model plugs into the framework, and a
sanity check that a generated dataset is consumable end-to-end.

``RawHReferenceScheme`` implements the ``CodebookScheme`` interface the eventual
ML model will implement -- ``select`` (UE encoder) / ``precoder`` (gNB decoder) /
``overhead_bits``.  Here it is an *oracle* (no compression: it reconstructs the
precoder directly from the channel), marking exactly where a learned
encoder/decoder pair will slot in.  The driver loads one config from a dataset,
replays its channels through ``evaluate`` against a baseline R16 codebook, and
prints SE / SGCS / bits -- finite numbers prove the data is framework-ready.

    python scripts/dataset/ml_scheme_reference.py data/cdl_smoke --n3 8
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

import numpy as np

from nr_csi.baselines.ideal import eigen_precoder
from nr_csi.channel.base import ChannelSource
from nr_csi.codebooks import R16Type2Codebook
from nr_csi.codebooks.base import CodebookScheme
from nr_csi.config import AntennaConfig
from nr_csi.dataset import io
from nr_csi.eval import evaluate


class RawHReferenceScheme(CodebookScheme):
    """Oracle stand-in for a learned raw-H encoder/decoder (no compression).

    A real model replaces ``select`` with a learned quantizing encoder and
    ``precoder`` with the trained decoder; ``overhead_bits`` then returns the
    quantized latent size instead of 0."""

    name = "raw-H oracle (reference)"

    def select(self, H: np.ndarray, rank: int = 1) -> Any:
        # H[slot, N3, rx, port]; the "report" is the (slot-averaged) channel.
        return {"H": H.mean(axis=0), "rank": rank}

    def precoder(self, pmi: Any) -> np.ndarray:
        W = eigen_precoder(pmi["H"], rank=pmi["rank"])  # (N3, P, rank)
        return W[None]  # (interval=1, N3, P, rank)

    def overhead_bits(self, pmi: Any) -> dict[str, int]:
        return {"latent": 0}  # oracle: a trained model reports its latent here


class FrozenH(ChannelSource):
    """Replay stored channels (reduced to ``N3`` subbands) as the framework's
    H[slot, N3, rx, port]."""

    def __init__(self, H_stack: np.ndarray) -> None:  # (n, N3, rx, port)
        self.bank = H_stack
        self.i = 0
        self.N3 = H_stack.shape[1]
        self.n_rx = H_stack.shape[2]
        self.n_ports = H_stack.shape[3]

    def generate(self, n_slots: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        H = self.bank[self.i % len(self.bank)]
        self.i += 1
        return np.repeat(H[None], n_slots, axis=0)  # (n_slots, N3, rx, port)


def reduce_freq(H: np.ndarray, N3: int) -> np.ndarray:
    """(n, rx, port, n_freq) -> (n, N3, rx, port) by averaging freq bins into
    N3 PMI subbands (the same reduction the Sionna adapter applies)."""
    n, rx, port, nf = H.shape
    g = nf // N3
    Hr = H[..., :g * N3].reshape(n, rx, port, N3, g).mean(-1)
    return Hr.transpose(0, 3, 1, 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_dir", type=pathlib.Path)
    p.add_argument("--config", default=None, help="config tag (default: first)")
    p.add_argument("--n3", type=int, default=8, help="PMI subbands for the baseline")
    p.add_argument("--n-drops", type=int, default=64)
    p.add_argument("--rank", type=int, default=1)
    args = p.parse_args()

    manifest = io.read_manifest(args.dataset_dir)
    entry = (next(e for e in manifest["configs"] if e["tag"] == args.config)
             if args.config else manifest["configs"][0])
    ant = AntennaConfig.standard(entry["N1"], entry["N2"])

    data = io.load_config_array(args.dataset_dir, entry["tag"])
    n = min(args.n_drops, data["H"].shape[0])
    H = reduce_freq(data["H"][:n], args.n3)  # (n, N3, rx, port)

    schemes = [
        R16Type2Codebook(ant, N3=args.n3, param_combination=4),
        RawHReferenceScheme(),
    ]
    print(f"Config {entry['tag']} (P={ant.P}), N3={args.n3}, {n} drops, rank {args.rank}")
    for scheme in schemes:
        res = evaluate(scheme, FrozenH(H), snr_db=[0.0, 10.0, 20.0],
                       rank=args.rank, n_drops=n)
        se = ", ".join(f"{s:.2f}" for s in res.se)
        print(f"  {scheme.name:>26}: SE@[0,10,20]=[{se}]  "
              f"SGCS={res.sgcs:.3f}  bits={res.overhead_bits:.0f}")
    print("Finite SE/SGCS above => the dataset is framework-consumable.")


if __name__ == "__main__":
    main()
