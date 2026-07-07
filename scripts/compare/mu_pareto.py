"""MU-MIMO overhead-vs-throughput Pareto comparison -- the headline
codebook comparison, done the way RAN1 compared these codebooks.

For each codebook configuration on the roster, K users each report a PMI on
their own channel drop, the gNB zero-forces across all reported directions,
and the residual-interference sum rate is scored (``evaluate_mu``).  Every
scheme sees the *same* channel drops (matched seed), and each (bits, rate)
point carries a Student-t confidence interval over drops plus a cell-edge
proxy (5th percentile of the pooled per-user rates) -- mean-only,
unmatched comparisons are how codebook myths get made.

Port-selection codebooks are evaluated behind ``BeamformedPortsScheme``
(eigen-beamformed CSI-RS), the physics they are specified for; running them
on raw antenna elements would say nothing about the codebook.

Outputs: <out>/mu_pareto.csv + <out>/mu_pareto.png + a console table.

Run:  .venv/bin/python scripts/compare/mu_pareto.py
      .venv/bin/python scripts/compare/mu_pareto.py --quick   (CI smoke)
"""

from __future__ import annotations

import argparse
import csv
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
    Type1Codebook,
)
from nr_csi.config import AntennaConfig  # noqa: E402
from nr_csi.eval import (  # noqa: E402
    BeamformedPortsScheme,
    edge_rate,
    evaluate_mu,
    mean_ci,
)

EDGE_Q = 5.0  # cell-edge percentile of the pooled per-user rates


def build_roster(antenna: AntennaConfig, n3: int, quick: bool = False):
    """(label, scheme) ladder: one point per (family, overhead knob)."""
    roster = [
        ("Type I", Type1Codebook(antenna, N3=n3)),
        ("Type II L=2", R15Type2Codebook(antenna, N3=n3, L=2, subband_amplitude=True)),
        ("eType II pc2", R16Type2Codebook(antenna, N3=n3, param_combination=2)),
        ("eType II pc4", R16Type2Codebook(antenna, N3=n3, param_combination=4)),
    ]
    if not quick:
        roster += [
            ("Type II L=4", R15Type2Codebook(antenna, N3=n3, L=4, subband_amplitude=True)),
            ("eType II pc6", R16Type2Codebook(antenna, N3=n3, param_combination=6)),
        ]
    # FeType II R17 behind eigen-beamformed CSI-RS ports (its intended physics);
    # Kb = P/4 beams per pol halves the effective port count.
    kb = antenna.P // 4
    try:
        eff_ant = AntennaConfig.standard(kb, 1)
        for pc in (2,) if quick else (2, 4):
            inner = R17Type2Codebook(eff_ant, N3=n3, param_combination=pc)
            roster.append((
                f"FeType II pc{pc} (beamformed)",
                BeamformedPortsScheme(inner, n_raw_ports=antenna.P, n_beams_per_pol=kb),
            ))
    except ValueError as exc:  # no valid effective antenna for this geometry
        print(f"[skip] FeType II beamformed: {exc}")
    return roster


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n1", type=int, default=4)
    p.add_argument("--n2", type=int, default=2)
    p.add_argument("--n3", type=int, default=8)
    p.add_argument("--users", type=int, default=4)
    p.add_argument("--rank", type=int, default=1, help="layers per user")
    p.add_argument("--snr-db", type=float, default=10.0)
    p.add_argument("--drops", type=int, default=50)
    p.add_argument("--n-paths", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="output dir (default results/comparison/mu_pareto)")
    p.add_argument("--quick", action="store_true",
                   help="tiny roster/geometry for smoke testing")
    args = p.parse_args()

    if args.quick:
        args.n1, args.n2, args.n3 = 2, 2, 4
        args.drops = min(args.drops, 3)
        args.users = min(args.users, 2)

    antenna = AntennaConfig.standard(args.n1, args.n2)
    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[2] / "results" / "comparison" / "mu_pareto"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    channel = RandomRayChannel(antenna, N3=args.n3, n_rx=max(1, args.rank),
                               n_paths=args.n_paths)

    rows = []
    full_csi_rates = []
    for label, scheme in build_roster(antenna, args.n3, quick=args.quick):
        # same seed => every scheme is scored on identical channel drops
        res = evaluate_mu(scheme, channel, n_users=args.users, snr_db=[args.snr_db],
                          n_drops=args.drops, rank=args.rank,
                          rng=np.random.default_rng(args.seed))
        per_drop = np.asarray(res.per_drop_user_rates)[:, 0, :]  # (drops, K)
        mean, hw = mean_ci(per_drop.sum(axis=1))
        rows.append({
            "label": label,
            "bits": round(res.overhead_bits, 1),
            "sum_rate": round(mean, 4),
            "sum_rate_ci95": round(hw, 4),
            "edge_rate_p5": round(edge_rate(per_drop, q=EDGE_Q), 4),
            "sum_rate_full_csi": round(res.sum_rate_full_csi[0], 4),
        })
        full_csi_rates.append(res.sum_rate_full_csi[0])
        print(f"{label:32s} bits={res.overhead_bits:7.1f}  "
              f"sum-rate={mean:6.3f} +/- {hw:5.3f}  "
              f"p{EDGE_Q:.0f} user rate={rows[-1]['edge_rate_p5']:6.3f}")

    # full-CSI reference is scheme-independent (same drops); report the mean
    full_csi = float(np.mean(full_csi_rates))
    print(f"{'full-CSI eigen ZF reference':32s} {'':14s} sum-rate={full_csi:6.3f}")

    csv_path = out_dir / "mu_pareto.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=120, sharex=True)
    for ax, key, title in [
        (axes[0], "sum_rate", f"ZF sum rate, K={args.users}, {args.snr_db:.0f} dB"),
        (axes[1], "edge_rate_p5", f"p{EDGE_Q:.0f} user rate (cell edge)"),
    ]:
        for r in rows:
            err = r["sum_rate_ci95"] if key == "sum_rate" else None
            ax.errorbar(r["bits"], r[key], yerr=err, marker="o", ms=4, capsize=3)
            ax.annotate(r["label"], (r["bits"], r[key]), fontsize=6,
                        textcoords="offset points", xytext=(4, 3))
        if key == "sum_rate":
            ax.axhline(full_csi, ls="--", lw=0.9, color="gray")
            ax.annotate("full CSI", (0.02, full_csi), fontsize=6, color="gray",
                        xycoords=("axes fraction", "data"), va="bottom")
        ax.set_xlabel("feedback bits per report")
        ax.set_ylabel("bits/s/Hz")
        ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / "mu_pareto.png"
    fig.savefig(png_path)
    print(f"\nWrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
