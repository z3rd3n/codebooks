"""Evaluate GLIMPSE against the 3GPP codebook families on frozen CDL banks.

Every scheme -- all configurations of Type I / R15 Type II / R16 eType II /
R17 FeType II (via the DFT PEB, its deployment model) and GLIMPSE at a grid
of (m, B) payload points with the learned / OMP / least-squares decoders --
is scored on the *same* pre-generated Sionna CDL drops through
``nr_csi.eval.evaluate``: identical channels, metrics, and bit accounting.

Writes one JSON per CDL model with per-scheme rows (bits, SGCS +- SEM,
SE@10 dB, subspace SGCS).  Optional extras: measurement-noise sweep and
rank-2.

    .venv/bin/python scripts/ml/eval_glimpse.py --model models/glimpse_p32 \
        --drops 200 --cdl C --out results/ml
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nr_csi.channel.base import ChannelSource  # noqa: E402
from nr_csi.codebooks import (  # noqa: E402
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    Type1Codebook,
)
from nr_csi.config import AntennaConfig  # noqa: E402
from nr_csi.eval import evaluate  # noqa: E402
from nr_csi.figtools.figlib import BeamDomainChannel  # noqa: E402
from nr_csi.ml import GlimpseScheme, LeastSquaresDecoder, OMPDecoder  # noqa: E402
from nr_csi.ml.projection import GlimpseCodec  # noqa: E402

SNRS = [0.0, 10.0, 20.0]
GLIMPSE_GRID = [(4, 3), (6, 3), (8, 2), (8, 3), (8, 4), (10, 3), (12, 3), (12, 4),
                (16, 2), (16, 3), (16, 4), (20, 3), (24, 2), (24, 3), (24, 4),
                (32, 3), (32, 4), (40, 3), (48, 3), (56, 3), (64, 3)]


class FrozenBank(ChannelSource):
    """Replayable bank of pre-generated CDL drops (paired comparisons)."""

    def __init__(self, H: np.ndarray) -> None:  # (n, slot, N3, rx, P)
        self.bank = H
        self.i = 0
        self.N3 = H.shape[2]
        self.n_rx = H.shape[3]
        self.n_ports = H.shape[4]

    def generate(self, n_slots: int = 1, rng=None) -> np.ndarray:
        H = self.bank[self.i % len(self.bank)]
        self.i += 1
        return H[:n_slots]

    def reset(self) -> None:
        self.i = 0


class ResetBeamDomain(BeamDomainChannel):
    def reset(self) -> None:
        if hasattr(self.inner, "reset"):
            self.inner.reset()


def make_bank(ant, n3, cdl_model, n_drops, seed, delay_spread=100e-9,
              speed=3.0, n_rx=2) -> FrozenBank:
    import tensorflow as tf

    from nr_csi.channel.sionna_adapter import SionnaCDLChannel

    tf.random.set_seed(seed)
    try:
        from sionna.phy import config as sionna_config

        sionna_config.seed = seed
    except Exception:
        pass
    chan = SionnaCDLChannel(ant, N3=n3, model=cdl_model, n_rx=n_rx,
                            ue_speed_kmh=speed, delay_spread=delay_spread,
                            fft_size=256)
    return FrozenBank(chan.generate_batch(n_drops, n_slots=1))


def codebook_roster(ant, n3) -> list[tuple]:
    """(scheme, domain, family, config) -- the fig_02 sweep."""
    import itertools

    pts = []

    def add(make, domain, family, label):
        try:
            pts.append((make(), domain, family, label))
        except ValueError:
            pass

    for mode in (1, 2):
        add(lambda m=mode: Type1Codebook(ant, N3=n3, mode=m), "antenna",
            "R15 Type I", f"mode{mode}")
    for L, n_psk, sa in itertools.product((2, 3, 4), (4, 8), (False, True)):
        add(lambda L=L, n=n_psk, s=sa: R15Type2Codebook(ant, N3=n3, L=L, n_psk=n,
                                                        subband_amplitude=s),
            "antenna", "R15 Type II", f"L{L}-{n_psk}PSK{'-SA' if sa else ''}")
    # paramCombination-r16 7/8 (L=6) additionally require ranks 3-4 disallowed
    # (typeII-RI-Restriction-r16 r2=r3=0); at rank-1-only evaluation this is
    # legal and the roster should not silently skip them via the ValueError
    # guard that fires under the default (all-ranks-allowed) restriction.
    ri_rank12_only = np.array([True, True, False, False])
    for pc in range(1, 9):
        add(lambda p=pc: R16Type2Codebook(
                ant, N3=n3, param_combination=p,
                ri_restriction=ri_rank12_only if p in (7, 8) else None),
            "antenna", "R16 eType II", f"pc{pc}")
    for pc in range(1, 9):
        add(lambda p=pc: R17Type2Codebook(ant, N3=n3, param_combination=p),
            "beam", "R17 FeType II PS", f"pc{pc}")
    return pts


def load_codec_and_decoders(ant, n3, model_path, m_max, seed, decoders):
    """The published KLT codec + the requested decoders (shared across rows)."""
    codec_path = pathlib.Path(f"{model_path}_codec")
    codec = (GlimpseCodec.load(codec_path) if codec_path.with_suffix(".npz").exists()
             else GlimpseCodec(ant, n3, m_max=m_max, seed=seed))
    decs = {}
    if "learned" in decoders:
        from nr_csi.ml.decoder import KerasDecoder

        decs["learned"] = KerasDecoder.load(model_path)
    if "omp" in decoders:
        decs["omp"] = OMPDecoder(codec)
    if "ls" in decoders:
        decs["ls"] = LeastSquaresDecoder(codec)
    return codec, decs


def glimpse_roster(ant, n3, codec, decs, grid=GLIMPSE_GRID) -> list[tuple]:
    # One codec (the published KLT constant) drives the UE projection for every
    # decoder, so all GLIMPSE rows share the same report; water-filled by default.
    roster = []
    for name, dec in decs.items():
        fam = {"learned": "GLIMPSE (learned)", "omp": "GLIMPSE (OMP)",
               "ls": "GLIMPSE (LS)"}[name]
        for m, b in grid:
            if m > codec.m_max:
                continue
            roster.append((GlimpseScheme(ant, n3, dec, m=m, bits=b, codec=codec),
                           "antenna", fam, f"m{m}-B{b}"))
    return roster


def ablation_roster(ant, n3, klt_codec, learned, m_max, seed,
                    grid=GLIMPSE_GRID) -> list[tuple]:
    """Ablations, all holding one factor fixed at a time:

    * **quantizer**: KLT + learned decoder with *uniform* (not water-filled)
      bit allocation -- isolates the value of reverse water-filling;
    * **basis**: *random* projection + OMP/LS -- isolates the value of the
      fitted KLT basis (no learned network, so no second training run).
    """
    roster = []
    # (1) uniform-allocation ablation on the headline KLT + learned decoder
    if learned is not None:
        for m, b in grid:
            if m > klt_codec.m_max:
                continue
            roster.append((GlimpseScheme(ant, n3, learned, m=m, bits=b, codec=klt_codec,
                                         allocation="uniform"),
                           "antenna", "GLIMPSE (learned, uniform-B)", f"m{m}-B{b}"))
    # (2) random-basis ablation with the free classical decoders
    rand = GlimpseCodec(ant, n3, m_max=m_max, seed=seed)  # distribution-blind
    for dec_name, dec in (("omp", OMPDecoder(rand)), ("ls", LeastSquaresDecoder(rand))):
        for m, b in grid:
            if m > rand.m_max:
                continue
            roster.append((GlimpseScheme(ant, n3, dec, m=m, bits=b, codec=rand),
                           "antenna", f"GLIMPSE-random ({dec_name.upper()})", f"m{m}-B{b}"))
    return roster


def run_roster(roster, bank, rank, n_drops, measurement_snr_db=None) -> list[dict]:
    rows = []
    for scheme, domain, family, label in roster:
        bank.reset()
        chan = ResetBeamDomain(bank, scheme.antenna) if domain == "beam" else bank
        res = evaluate(scheme, chan, snr_db=SNRS, rank=rank, n_drops=n_drops,
                       rng=np.random.default_rng(0),
                       measurement_snr_db=measurement_snr_db)
        sem = float(np.std(res.per_drop_sgcs) / np.sqrt(len(res.per_drop_sgcs)))
        rows.append(dict(
            family=family, config=label, bits=res.overhead_bits, sgcs=res.sgcs,
            sgcs_sem=sem, subspace_sgcs=res.subspace_sgcs,
            se=dict(zip([str(s) for s in SNRS], res.se)),
            se_ub=dict(zip([str(s) for s in SNRS], res.se_upper_bound)),
        ))
        print(f"{family:<18} {label:<12} bits={res.overhead_bits:6.0f} "
              f"sgcs={res.sgcs:.3f}+-{sem:.3f} se10={res.se[1]:.2f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=pathlib.Path, default=pathlib.Path("models/glimpse_p32"))
    ap.add_argument("--geometry", default="4x4x8", help="N1xN2xN3")
    ap.add_argument("--cdl", default="C", help="comma list of CDL models to evaluate")
    ap.add_argument("--drops", type=int, default=200)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--seed", type=int, default=777, help="frozen-bank Sionna seed")
    ap.add_argument("--m-max", type=int, default=64)
    ap.add_argument("--decoders", default="learned,omp,ls")
    ap.add_argument("--no-codebooks", action="store_true")
    ap.add_argument("--measurement-snr-db", type=float, default=None)
    ap.add_argument("--ablation", action="store_true",
                    help="also score the random-basis (distribution-blind) variant")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/ml"))
    args = ap.parse_args()

    n1, n2, n3 = (int(x) for x in args.geometry.split("x"))
    ant = AntennaConfig.standard(n1, n2)
    args.out.mkdir(parents=True, exist_ok=True)

    codec, decs = load_codec_and_decoders(
        ant, n3, args.model, args.m_max, 0, tuple(args.decoders.split(",")))
    roster = [] if args.no_codebooks else codebook_roster(ant, n3)
    roster += glimpse_roster(ant, n3, codec, decs)
    if args.ablation:
        roster += ablation_roster(ant, n3, codec, decs.get("learned"), args.m_max, seed=0)

    for cdl_model in args.cdl.split(","):
        print(f"=== CDL-{cdl_model}, {args.drops} drops, rank {args.rank} ===")
        bank = make_bank(ant, n3, cdl_model, args.drops, args.seed)
        rows = run_roster(roster, bank, args.rank, args.drops,
                          args.measurement_snr_db)
        noise_tag = (f"_msnr{int(args.measurement_snr_db)}"
                     if args.measurement_snr_db is not None else "")
        rank_tag = f"_rank{args.rank}" if args.rank != 1 else ""
        name = f"frontier_cdl{cdl_model}{rank_tag}{noise_tag}{args.tag}.json"
        payload = dict(
            geometry=dict(N1=n1, N2=n2, P=ant.P, N3=n3), cdl_model=cdl_model,
            drops=args.drops, rank=args.rank, seed=args.seed,
            measurement_snr_db=args.measurement_snr_db,
            model=str(args.model), points=rows,
        )
        (args.out / name).write_text(json.dumps(payload, indent=1))
        print("wrote", args.out / name)


if __name__ == "__main__":
    main()
