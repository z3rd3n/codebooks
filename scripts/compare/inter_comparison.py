"""Inter-codebook comparison figures -- every family on the same axes,
scored with the realism the codebooks were designed for.

Four figures, all on matched channel drops (same seed per figure):

1. ``fig1_mu_pareto``     -- feedback bits vs MU-MIMO ZF sum rate (K users,
   residual interference) with 95% CIs, plus the p5 pooled per-user rate
   (cell edge).  The headline comparison: this is the axis Type II exists on.
2. ``fig2_delay``         -- SGCS and SE vs CSI feedback delay on a fast
   channel: static reports age, the R18 Doppler codebook predicts.
3. ``fig3_rate_distortion`` -- bits vs SGCS and bits vs SE/capacity for the
   SU intermediate-KPI view; FeType II appears twice -- raw antenna ports
   (dashed) vs eigen-beamformed CSI-RS (solid) -- to show what evaluating a
   port-selection codebook on the wrong physics does.
4. ``fig4_receiver``      -- rank-2 SE with joint decoding vs a per-layer
   linear MMSE receiver (layer-orthogonality cost), and fixed-rank vs
   auto-RI SE.

Colors are fixed per family across all figures (identity, not rank).

Run:  .venv/bin/python scripts/compare/inter_comparison.py
      .venv/bin/python scripts/compare/inter_comparison.py --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nr_csi.channel import RandomRayChannel  # noqa: E402
from nr_csi.codebooks import (  # noqa: E402
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
)
from nr_csi.config import AntennaConfig  # noqa: E402
from nr_csi.eval import (  # noqa: E402
    BeamformedPortsScheme,
    delay_sweep,
    edge_rate,
    evaluate,
    evaluate_mu,
    mean_ci,
)

# fixed categorical slots (validated palette; color follows the family)
COLOR = {
    "Type I": "#2a78d6",       # blue
    "Type II": "#1baf7a",      # aqua
    "eType II": "#eda100",     # yellow
    "R18 Doppler": "#4a3aa7",  # violet
    "FeType II": "#e34948",    # red
    "full CSI": "#6e6e6e",     # neutral reference
}
SNR_DB = 10.0
EDGE_Q = 5.0


def _antenna(args):
    return AntennaConfig.standard(args.n1, args.n2)


def _fe_beamformed(antenna, n3, pc):
    kb = antenna.P // 4
    inner = R17Type2Codebook(AntennaConfig.standard(kb, 1), N3=n3, param_combination=pc)
    return BeamformedPortsScheme(inner, n_raw_ports=antenna.P, n_beams_per_pol=kb)


def build_ladders(antenna, n3, quick=False):
    """family -> [(point_label, scheme)], ordered by overhead."""
    ladders = {
        "Type I": [("", Type1Codebook(antenna, N3=n3))],
        "Type II": [
            ("L=2", R15Type2Codebook(antenna, N3=n3, L=2, subband_amplitude=True)),
        ] + ([] if quick else [
            ("L=3", R15Type2Codebook(antenna, N3=n3, L=3, subband_amplitude=True)),
            ("L=4", R15Type2Codebook(antenna, N3=n3, L=4, subband_amplitude=True)),
        ]),
        "eType II": [
            ("pc2", R16Type2Codebook(antenna, N3=n3, param_combination=2)),
        ] + ([] if quick else [
            ("pc4", R16Type2Codebook(antenna, N3=n3, param_combination=4)),
            ("pc6", R16Type2Codebook(antenna, N3=n3, param_combination=6)),
        ]),
        "FeType II": [
            ("pc2", _fe_beamformed(antenna, n3, 2)),
        ] + ([] if quick else [
            ("pc4", _fe_beamformed(antenna, n3, 4)),
        ]),
    }
    return ladders


def _style_axes(ax):
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


# ------------------------------------------------------------------ figure 1

def fig1_mu_pareto(args, out_dir, results):
    antenna = _antenna(args)
    channel = RandomRayChannel(antenna, N3=args.n3, n_rx=1, n_paths=args.n_paths)
    ladders = build_ladders(antenna, args.n3, args.quick)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=130)
    data = {}
    full_csi = []
    for family, points in ladders.items():
        pts = []
        for plabel, scheme in points:
            res = evaluate_mu(scheme, channel, n_users=args.users, snr_db=[SNR_DB],
                              n_drops=args.drops, rng=np.random.default_rng(args.seed))
            per_drop = np.asarray(res.per_drop_user_rates)[:, 0, :]
            mean, hw = mean_ci(per_drop.sum(axis=1))
            pts.append({"label": plabel, "bits": res.overhead_bits, "sum_rate": mean,
                        "ci": hw, "edge": edge_rate(per_drop, q=EDGE_Q)})
            full_csi.append(res.sum_rate_full_csi[0])
        data[family] = sorted(pts, key=lambda d: d["bits"])

    ref = float(np.mean(full_csi))
    for ax, key, title in [(axes[0], "sum_rate", f"MU ZF sum rate (K={args.users}, {SNR_DB:.0f} dB)"),
                           (axes[1], "edge", f"p{EDGE_Q:.0f} pooled user rate (cell edge)")]:
        for family, pts in data.items():
            x = [p["bits"] for p in pts]
            y = [p[key] for p in pts]
            yerr = [p["ci"] for p in pts] if key == "sum_rate" else None
            ax.errorbar(x, y, yerr=yerr, color=COLOR[family], marker="o", ms=4.5,
                        lw=1.6, capsize=2.5, label=family)
            for p in pts:
                if p["label"]:
                    ax.annotate(p["label"], (p["bits"], p[key]), fontsize=6,
                                color="#444444", textcoords="offset points", xytext=(3, 4))
        if key == "sum_rate":
            ax.axhline(ref, ls="--", lw=1.0, color=COLOR["full CSI"])
            ax.annotate("full-CSI eigen ZF", (0.98, ref), xycoords=("axes fraction", "data"),
                        fontsize=6.5, color=COLOR["full CSI"], ha="right", va="bottom")
        ax.set_xlabel("feedback bits per report", fontsize=8.5)
        ax.set_ylabel("bits/s/Hz", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        _style_axes(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_mu_pareto.png")
    plt.close(fig)
    results["fig1_mu_pareto"] = {"full_csi": ref, "families": data}


# ------------------------------------------------------------------ figure 2

def fig2_delay(args, out_dir, results):
    antenna = _antenna(args)
    channel = RandomRayChannel(antenna, N3=args.n3, n_rx=2, n_paths=args.n_paths,
                               max_doppler=1.0, doppler_period=8)
    delays = (0, 1, 2) if args.quick else (0, 1, 2, 3, 4, 6)
    n4 = 4
    schemes = [
        ("Type I", Type1Codebook(antenna, N3=args.n3), {}),
        ("Type II", R15Type2Codebook(antenna, N3=args.n3, L=4, subband_amplitude=True), {}),
        ("eType II", R16Type2Codebook(antenna, N3=args.n3, param_combination=6), {}),
        ("R18 Doppler", R18Type2Codebook(antenna, N3=args.n3, N4=n4, param_combination=7),
         {"n_slots": n4, "delay_aware": True}),
    ]
    if args.quick:
        schemes = schemes[2:]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=130)
    data = {}
    for family, scheme, extra in schemes:
        # same observation length for single-shot codebooks as R18's window
        kw = dict(snr_db=[SNR_DB], rank=1, n_drops=args.drops, seed=args.seed,
                  measurement_slots=None if extra else n4, **extra)
        out = delay_sweep(scheme, channel, delays=delays, **kw)
        data[family] = {
            "delays": list(delays),
            "sgcs": [out[d].sgcs for d in delays],
            "se": [out[d].se[0] for d in delays],
            "bits": float(np.mean([out[d].overhead_bits for d in delays])),
        }
    for ax, key, title in [(axes[0], "sgcs", "SGCS vs CSI feedback delay"),
                           (axes[1], "se", f"SE at {SNR_DB:.0f} dB vs delay")]:
        for family, d in data.items():
            ax.plot(d["delays"], d[key], color=COLOR[family], marker="o", ms=4.5,
                    lw=1.6, label=f"{family} ({d['bits']:.0f} b)")
        ax.set_xlabel("feedback delay (slot intervals)", fontsize=8.5)
        ax.set_ylabel("SGCS" if key == "sgcs" else "bits/s/Hz", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        _style_axes(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_delay.png")
    plt.close(fig)
    results["fig2_delay"] = data


# ------------------------------------------------------------------ figure 3

def fig3_rate_distortion(args, out_dir, results):
    antenna = _antenna(args)
    channel = RandomRayChannel(antenna, N3=args.n3, n_rx=2, n_paths=args.n_paths)
    ladders = build_ladders(antenna, args.n3, args.quick)
    # FeType II twice: same hue, raw = dashed, beamformed = solid
    fe_pcs = (2,) if args.quick else (2, 4)
    variants = {"FeType II (beamformed)": ladders.pop("FeType II")}
    variants["FeType II (raw ports)"] = [
        (f"pc{pc}", R17Type2Codebook(antenna, N3=args.n3, param_combination=pc))
        for pc in fe_pcs
    ]

    def _eval(scheme):
        res = evaluate(scheme, channel, snr_db=[SNR_DB], rank=1, n_drops=args.drops,
                       rng=np.random.default_rng(args.seed))
        return {"bits": res.overhead_bits, "sgcs": res.sgcs,
                "se_frac": res.se[0] / res.capacity_upper_bound[0]}

    series = {f: [( _eval(s) | {"label": pl}) for pl, s in pts]
              for f, pts in {**ladders, **variants}.items()}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=130)
    for ax, key, title in [(axes[0], "sgcs", "SGCS vs feedback bits (rank 1)"),
                           (axes[1], "se_frac", f"SE / capacity at {SNR_DB:.0f} dB")]:
        for fam, pts in series.items():
            pts = sorted(pts, key=lambda d: d["bits"])
            base = fam.split(" (")[0]
            ls = "--" if "raw" in fam else "-"
            mfc = "white" if "raw" in fam else None
            ax.plot([p["bits"] for p in pts], [p[key] for p in pts], ls,
                    color=COLOR[base], marker="o", ms=4.5, lw=1.6, mfc=mfc,
                    label=fam)
            for p in pts:
                if p["label"]:
                    ax.annotate(p["label"], (p["bits"], p[key]), fontsize=6,
                                color="#444444", textcoords="offset points", xytext=(3, 4))
        ax.set_xlabel("feedback bits per report", fontsize=8.5)
        ax.set_ylabel("SGCS" if key == "sgcs" else "fraction of capacity", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        _style_axes(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_rate_distortion.png")
    plt.close(fig)
    results["fig3_rate_distortion"] = series


# ------------------------------------------------------------------ figure 4

def fig4_receiver(args, out_dir, results):
    antenna = _antenna(args)
    channel = RandomRayChannel(antenna, N3=args.n3, n_rx=2, n_paths=max(6, args.n_paths))
    configs = [
        ("Type I", Type1Codebook(antenna, N3=args.n3)),
        ("Type II", R15Type2Codebook(antenna, N3=args.n3, L=4, subband_amplitude=True)),
        ("eType II", R16Type2Codebook(antenna, N3=args.n3, param_combination=6)),
        ("FeType II", _fe_beamformed(antenna, args.n3, 4)),
    ]
    if args.quick:
        configs = configs[:2]

    rows = {}
    for family, scheme in configs:
        kw = dict(snr_db=[SNR_DB], n_drops=args.drops)
        r2 = evaluate(scheme, channel, rank=2, rng=np.random.default_rng(args.seed), **kw)
        r1 = evaluate(scheme, channel, rank=1, rng=np.random.default_rng(args.seed), **kw)
        auto = evaluate(scheme, channel, rank="auto", auto_ranks=(1, 2),
                        rng=np.random.default_rng(args.seed), **kw)
        rows[family] = {
            "joint": r2.se[0], "mmse": r2.se_mmse[0],
            "rank1": r1.se[0], "rank2": r2.se[0], "auto": auto.se[0],
            "auto_rank_share": float(np.mean(np.asarray(auto.per_drop_rank) == 2)),
        }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=130)
    fams = list(rows)
    x = np.arange(len(fams))
    w = 0.34
    ax = axes[0]
    ax.bar(x - w / 2, [rows[f]["joint"] for f in fams], w, color=[COLOR[f] for f in fams],
           label="joint decoding")
    ax.bar(x + w / 2, [rows[f]["mmse"] for f in fams], w, color=[COLOR[f] for f in fams],
           alpha=0.45, hatch="//", label="per-layer MMSE")
    ax.set_xticks(x, fams, fontsize=8)
    ax.set_ylabel("bits/s/Hz", fontsize=8.5)
    ax.set_title(f"rank-2 SE at {SNR_DB:.0f} dB: receiver model", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style_axes(ax)

    ax = axes[1]
    for i, (marker, key, label) in enumerate([("s", "rank1", "fixed rank 1"),
                                              ("^", "rank2", "fixed rank 2"),
                                              ("o", "auto", "auto-RI")]):
        ax.plot(x, [rows[f][key] for f in fams], marker, ms=6,
                color="#444444" if key != "auto" else "#111111",
                mfc="white" if key != "auto" else None, label=label)
    ax.set_xticks(x, fams, fontsize=8)
    ax.set_ylabel("bits/s/Hz", fontsize=8.5)
    ax.set_title("fixed rank vs auto-RI SE", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_receiver.png")
    plt.close(fig)
    results["fig4_receiver"] = rows


# ---------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n1", type=int, default=4)
    p.add_argument("--n2", type=int, default=2)
    p.add_argument("--n3", type=int, default=8)
    p.add_argument("--users", type=int, default=4)
    p.add_argument("--drops", type=int, default=40)
    p.add_argument("--n-paths", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--figs", default="1,2,3,4", help="comma list of figures to build")
    args = p.parse_args()

    if args.quick:
        args.n1, args.n2, args.n3 = 2, 2, 4
        args.drops = min(args.drops, 3)
        args.users = min(args.users, 2)

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[2] / "results" / "comparison" / "inter_comparison"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {"1": fig1_mu_pareto, "2": fig2_delay,
                "3": fig3_rate_distortion, "4": fig4_receiver}
    results: dict = {"config": {k: getattr(args, k) for k in
                                ("n1", "n2", "n3", "users", "drops", "n_paths", "seed", "quick")}}
    for key in args.figs.split(","):
        builders[key.strip()](args, out_dir, results)
        print(f"fig{key.strip()} done", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"Wrote figures + summary.json to {out_dir}")


if __name__ == "__main__":
    main()
