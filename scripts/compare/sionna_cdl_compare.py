"""Compare 3GPP codebooks (+ an ML stand-in) on realistic 38.901 CDL channels.

The companion of ``scripts/compare_schemes.py``: the same five schemes
(Type I, R15/R16 Type II, R17 port-selection, and the ``OracleSVDScheme`` ML
stand-in), but scored on Sionna CDL drops instead of the synthetic random-ray
channel, and rendered as figures under ``results/sionna_cdl/``.

Requires the optional ``sionna`` extra (``pip install -e ".[sionna]"``).

Because ``SionnaCDLChannel.generate`` is driven by TensorFlow's RNG (it ignores
the NumPy ``rng``), the harness's paired-seed trick cannot make every scheme
see the same drops.  We therefore pre-generate a *frozen bank* of CDL drops
once and replay it for every scheme: this makes the comparison paired and
reproducible, and generates the (expensive) Sionna channel only once instead of
per scheme.

Figures (each written as ``<name>.png`` + ``<name>.json``):
  * ``se_vs_snr``  -- SE vs SNR for all schemes + the eigen upper bound.
  * ``summary``    -- SGCS, feedback bits, and SE@{0,10,20} dB as bar charts.
  * ``cdl_models`` -- per-scheme SGCS across CDL-A..E (sparse NLOS -> LoS).

Run: .venv/bin/python scripts/sionna_cdl_compare.py            # full
     .venv/bin/python scripts/sionna_cdl_compare.py --fast     # smoke test
     .venv/bin/python scripts/sionna_cdl_compare.py --model A --drops 120
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF before any import

import argparse  # noqa: E402
import pathlib  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (figlib set the Agg backend)
import numpy as np  # noqa: E402
from compare_schemes import OracleSVDScheme  # noqa: E402  (ML stand-in)
from figlib import RESULTS, BeamDomainChannel, save, style  # noqa: E402

from nr_csi.channel.base import ChannelSource  # noqa: E402
from nr_csi.channel.sionna_adapter import SionnaCDLChannel  # noqa: E402
from nr_csi.codebooks import (  # noqa: E402
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    Type1Codebook,
)
from nr_csi.config import AntennaConfig  # noqa: E402
from nr_csi.eval import evaluate  # noqa: E402

ANT = AntennaConfig.standard(16, 1)  # P = 32 CSI-RS ports
N3 = 8
N_RX = 2
SNR_SWEEP = list(np.arange(-5.0, 31.0, 5.0))
SUMMARY_SNR = [0.0, 10.0, 20.0]
CDL_MODELS = ["A", "B", "C", "D", "E"]  # sparse NLOS -> dominant LoS
ORACLE_STYLE = dict(color="#A2142F", marker="*")  # OracleSVD has no STYLE entry


def plot_style(name: str) -> dict:
    """``figlib.style`` with a fallback for the ML stand-in (no STYLE key)."""
    return style(name) or dict(ORACLE_STYLE)


def roster(ant: AntennaConfig = ANT, n3: int = N3) -> list[tuple]:
    """The five-scheme comparison set as (scheme, domain) pairs; R17 is a
    port-selection codebook, evaluated through the unitary DFT PEB (its real
    deployment model -- see figlib.BeamDomainChannel)."""
    return [
        (Type1Codebook(ant, N3=n3), "antenna"),
        (R15Type2Codebook(ant, N3=n3, L=3), "antenna"),
        (R16Type2Codebook(ant, N3=n3, param_combination=4), "antenna"),
        (R17Type2Codebook(ant, N3=n3, param_combination=5), "beam"),
        (OracleSVDScheme(), "antenna"),
    ]


def seed_sionna(seed: int) -> None:
    """Seed the TF/Sionna RNG so a frozen bank is reproducible across runs."""
    import tensorflow as tf

    tf.random.set_seed(seed)
    try:
        from sionna.phy import config as sionna_config

        sionna_config.seed = seed
    except Exception:  # pragma: no cover - older/newer Sionna layouts
        pass


class FrozenChannel(ChannelSource):
    """Replays a precomputed list of CDL drops (paired & reproducible).

    A fresh instance restarts at drop 0, so handing each scheme its own
    ``FrozenChannel`` over a shared ``bank`` makes every scheme see the same
    sequence of channel realizations.
    """

    def __init__(self, bank: list[np.ndarray]) -> None:
        self.bank = bank
        self.i = 0

    def generate(self, n_slots: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        H = self.bank[self.i % len(self.bank)]
        self.i += 1
        if H.shape[0] < n_slots:
            raise ValueError(f"frozen drop has {H.shape[0]} slots, need {n_slots}")
        return H[:n_slots]


def build_bank(model: str, n_drops: int, *, seed: int, speed: float) -> list[np.ndarray]:
    """Pre-generate ``n_drops`` single-slot CDL drops for one model."""
    seed_sionna(seed)
    cdl = SionnaCDLChannel(ANT, N3=N3, model=model, n_rx=N_RX, ue_speed_kmh=speed)
    return [cdl.generate(n_slots=1) for _ in range(n_drops)]


def eval_on_bank(scheme, domain: str, bank: list[np.ndarray], snr_db, *, seed: int, rank: int = 1):
    """Evaluate one scheme on a frozen bank (R17 via the DFT PEB)."""
    chan: ChannelSource = FrozenChannel(bank)
    if domain == "beam":
        chan = BeamDomainChannel(chan, ANT)
    return evaluate(scheme, chan, snr_db=snr_db, rank=rank, n_drops=len(bank),
                    rng=np.random.default_rng(seed))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_se_and_summary(bank, model, drops, seed, out):
    """SE-vs-SNR sweep + the SGCS/bits/SE summary, from one pass over the bank."""
    results = {}  # name -> EvalResult (SE over SNR_SWEEP)
    for scheme, domain in roster():
        results[scheme.name] = eval_on_bank(scheme, domain, bank, SNR_SWEEP, seed=seed)

    # ---- Figure 1: SE vs SNR -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for scheme, domain in roster():
        res = results[scheme.name]
        label = scheme.name + (" (via PEB)" if domain == "beam" else "")
        ax.plot(SNR_SWEEP, res.se, label=label, **plot_style(scheme.name))
    ub = results["R15 Type I"].se_upper_bound  # same drops -> shared bound
    ax.plot(SNR_SWEEP, ub, label="eigen upper bound", **style("eigen upper bound"))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("spectral efficiency (bits/s/Hz)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"SU spectral efficiency vs SNR -- 38.901 CDL-{model}, (16,1) array, "
        f"P={ANT.P}, N3={N3}, {drops} drops"
    )
    save(fig, out, "se_vs_snr", {
        "model": model, "drops": drops, "seed": seed, "snr_db": SNR_SWEEP,
        "se": {n: r.se for n, r in results.items()},
        "eigen_upper_bound": ub,
        "sgcs": {n: r.sgcs for n, r in results.items()},
        "overhead_bits": {n: r.overhead_bits for n, r in results.items()},
    })

    # ---- Figure 2: summary bars ---------------------------------------------
    names = [s.name for s, _ in roster()]
    short = [n.replace("Type ", "T").replace("eType ", "eT").replace("FeType ", "FeT")
             for n in names]
    colors = [plot_style(n).get("color", "0.5") for n in names]
    snr_idx = [SNR_SWEEP.index(s) for s in SUMMARY_SNR]
    sgcs = [results[n].sgcs for n in names]
    bits = [results[n].overhead_bits for n in names]
    se_at = {s: [results[n].se[i] for n in names] for s, i in zip(SUMMARY_SNR, snr_idx)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(names))

    axes[0].bar(x, sgcs, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("SGCS")
    axes[0].set_title("subspace alignment (SGCS)")

    axes[1].bar(x, bits, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("feedback bits / report")
    axes[1].set_title("overhead (log scale)")

    width = 0.26
    for off, s in zip((-width, 0.0, width), SUMMARY_SNR):
        axes[2].bar(x + off, se_at[s], width=width, label=f"{int(s)} dB")
    axes[2].set_ylabel("spectral efficiency (bits/s/Hz)")
    axes[2].set_title("SE at {0, 10, 20} dB")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Codebook comparison on 38.901 CDL-{model} ({drops} drops, P={ANT.P}, N3={N3})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "summary", {
        "model": model, "drops": drops, "seed": seed, "schemes": names,
        "sgcs": dict(zip(names, sgcs)),
        "overhead_bits": dict(zip(names, bits)),
        "se_at_db": {str(int(s)): dict(zip(names, se_at[s])) for s in SUMMARY_SNR},
    })


def fig_cdl_models(drops, seed, speed, out):
    """Per-scheme SGCS across CDL-A..E at a single operating point."""
    names = [s.name for s, _ in roster()]
    sgcs = {n: [] for n in names}  # name -> per-model SGCS
    for model in CDL_MODELS:
        bank = build_bank(model, drops, seed=seed, speed=speed)
        for scheme, domain in roster():
            res = eval_on_bank(scheme, domain, bank, [10.0], seed=seed)
            sgcs[scheme.name].append(res.sgcs)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CDL_MODELS))
    width = 0.16
    for k, name in enumerate(names):
        ax.bar(x + (k - 2) * width, sgcs[name], width=width, label=name,
               color=plot_style(name).get("color", "0.5"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"CDL-{m}" for m in CDL_MODELS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("SGCS (vs eigen target)")
    ax.set_xlabel("channel model (sparse NLOS  ->  dominant LoS)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))  # outside axes
    fig.suptitle(f"Subspace alignment across 38.901 CDL models @ 10 dB ({drops} drops, P={ANT.P})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "cdl_models", {
        "models": CDL_MODELS, "drops": drops, "seed": seed, "snr_db": 10.0,
        "sgcs": sgcs,
    })


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drops", type=int, default=80, help="Monte-Carlo CDL drops per bank")
    p.add_argument("--seed", type=int, default=0, help="TF/Sionna seed (reproducible banks)")
    p.add_argument("--out", type=pathlib.Path, default=RESULTS / "sionna_cdl",
                   help="output directory")
    p.add_argument("--model", default="C", help="CDL model for the SE/summary figures (A..E)")
    p.add_argument("--speed", type=float, default=3.0, help="UE speed (km/h)")
    p.add_argument("--fast", action="store_true", help="smoke-test sizes (few drops)")
    args = p.parse_args()
    if args.fast:
        args.drops = min(args.drops, 8)
    return args


def main() -> None:
    args = cli()
    args.out.mkdir(parents=True, exist_ok=True)
    bank = build_bank(args.model, args.drops, seed=args.seed, speed=args.speed)
    fig_se_and_summary(bank, args.model, args.drops, args.seed, args.out)
    fig_cdl_models(args.drops, args.seed, args.speed, args.out)


if __name__ == "__main__":
    main()
