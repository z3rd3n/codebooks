"""Publication figures: GLIMPSE vs the 3GPP codebook families.

Consumes the ``frontier_cdl*.json`` produced by ``eval_glimpse.py`` and
renders (each PNG paired with the JSON it was drawn from):

* ``fig_glimpse_frontier``  -- SGCS and SE@10 dB vs feedback bits, every
  codebook configuration + the GLIMPSE (learned / OMP / LS) grids, with the
  Pareto frontier and the eigen upper bound.  The headline result.
* ``fig_glimpse_gain``      -- bits needed for a target SGCS: GLIMPSE vs the
  best codebook at matched fidelity (the overhead-reduction bar).
* ``fig_glimpse_decoders``  -- learned vs OMP vs LS at matched bits (the
  value of the gNB-side prior; identical UE report).
* ``fig_glimpse_models``    -- SGCS across CDL-A..E at matched bits
  (cross-profile generalization) when those JSONs exist.

Run: .venv/bin/python scripts/ml/make_glimpse_figures.py --results results/ml
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {
    "R15 Type I": dict(color="#0072BD", marker="o"),
    "R15 Type II": dict(color="#D95319", marker="s"),
    "R16 eType II": dict(color="#EDB120", marker="^"),
    "R17 FeType II PS": dict(color="#7E2F8E", marker="D"),
    "GLIMPSE (learned)": dict(color="#119911", marker="*"),
    "GLIMPSE (OMP)": dict(color="#4DBEEE", marker="P"),
    "GLIMPSE (LS)": dict(color="#333333", marker="x"),
    "GLIMPSE (learned, uniform-B)": dict(color="#77AC30", marker="v"),
    "GLIMPSE-random (OMP)": dict(color="#A0A0A0", marker="P"),
    "GLIMPSE-random (LS)": dict(color="#C8C8C8", marker="x"),
}
CODEBOOK_FAMILIES = ["R15 Type I", "R15 Type II", "R16 eType II", "R17 FeType II PS"]
# families shown on the headline frontier (ablation variants get their own fig)
FRONTIER_FAMILIES = CODEBOOK_FAMILIES + ["GLIMPSE (learned)", "GLIMPSE (OMP)",
                                         "GLIMPSE (LS)"]


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    front, best = [], -1.0
    for bits, fid in sorted(points):
        if fid > best:
            front.append((bits, fid))
            best = fid
    return front


def se10(row: dict) -> float:
    return row["se"]["10.0"]


def frontier_figure(data: dict, out: pathlib.Path,
                    families: list[str] | None = None) -> None:
    rows = data["points"]
    keep = families or FRONTIER_FAMILIES
    fams = [f for f in dict.fromkeys(r["family"] for r in rows) if f in keep]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for metric, ax, ylabel in ((lambda r: r["sgcs"], axes[0], "mean SGCS"),
                               (se10, axes[1], "SE @ 10 dB (bits/s/Hz)")):
        for fam in fams:
            st = STYLE.get(fam, dict(color="k", marker="."))
            pts = [(r["bits"], metric(r)) for r in rows if r["family"] == fam]
            if not pts:
                continue
            xs, ys = zip(*sorted(pts))
            learned = fam == "GLIMPSE (learned)"
            # unfilled markers (x, +, ...) are drawn by their stroke, so they
            # need linewidths > 0 or they render invisible (plot AND legend)
            unfilled = st["marker"] in ("x", "+", "|", "_")
            if unfilled:
                # LS sits on the learned curve (LS ~= learned on CDL-C); size it
                # to straddle the stars so it reads as "coincides with learned"
                ax.scatter(xs, ys, s=70, color=st["color"], marker=st["marker"],
                           zorder=4, linewidths=1.6, label=fam, alpha=0.95)
            else:
                ax.scatter(xs, ys, s=90 if learned else 45, color=st["color"],
                           marker=st["marker"], zorder=5 if learned else 3,
                           edgecolors="k" if learned else "none",
                           linewidths=0.5 if learned else 0,
                           label=fam, alpha=0.95)
        cb_pts = [(r["bits"], metric(r)) for r in rows
                  if r["family"] in CODEBOOK_FAMILIES]
        gl_pts = [(r["bits"], metric(r)) for r in rows
                  if r["family"] == "GLIMPSE (learned)"]
        if cb_pts:
            ax.plot(*zip(*pareto(cb_pts)), color="0.55", ls=":", lw=1.4,
                    label="codebook Pareto", zorder=1)
        if gl_pts:
            ax.plot(*zip(*pareto(gl_pts)), color=STYLE["GLIMPSE (learned)"]["color"],
                    ls="-", lw=1.6, alpha=0.7, zorder=2)
        if metric is se10:
            ub = np.mean([r["se_ub"]["10.0"] for r in rows])
            ax.axhline(ub, color="0.35", ls="--", lw=1, label="eigen upper bound")
        ax.set_xscale("log")
        ax.set_xlabel("feedback overhead (bits per report)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="lower right", ncol=1)
    g = data["geometry"]
    fig.suptitle(f"GLIMPSE vs 3GPP codebooks -- CDL-{data['cdl_model']}, "
                 f"({g['N1']},{g['N2']}) P={g['P']}, N3={g['N3']}, "
                 f"rank {data['rank']}, {data['drops']} drops")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _best_codebook_bits_for(rows, target_sgcs) -> float | None:
    ok = [r["bits"] for r in rows
          if r["family"] in CODEBOOK_FAMILIES and r["sgcs"] >= target_sgcs]
    return min(ok) if ok else None


def _best_glimpse_bits_for(rows, target_sgcs) -> float | None:
    ok = [r["bits"] for r in rows
          if r["family"] == "GLIMPSE (learned)" and r["sgcs"] >= target_sgcs]
    return min(ok) if ok else None


def gain_figure(data: dict, out: pathlib.Path) -> dict:
    rows = data["points"]
    targets = [0.70, 0.80, 0.85, 0.90, 0.92]
    cb, gl, labels = [], [], []
    for t in targets:
        c, g = _best_codebook_bits_for(rows, t), _best_glimpse_bits_for(rows, t)
        if c is not None and g is not None:
            cb.append(c)
            gl.append(g)
            labels.append(f"{t:.2f}")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, cb, 0.4, color="#EDB120", label="best codebook")
    ax.bar(x + 0.2, gl, 0.4, color=STYLE["GLIMPSE (learned)"]["color"], label="GLIMPSE")
    for i, (c, g) in enumerate(zip(cb, gl)):
        ax.text(i, max(c, g) * 1.02, f"-{100 * (1 - g / c):.0f}%", ha="center",
                fontsize=9, color="#119911", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("target mean SGCS")
    ax.set_ylabel("feedback bits to reach target")
    ax.set_title(f"Overhead at matched fidelity -- CDL-{data['cdl_model']}")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"targets": labels, "codebook_bits": cb, "glimpse_bits": gl,
            "reduction_pct": [100 * (1 - g / c) for c, g in zip(cb, gl)]}


def decoder_figure(data: dict, out: pathlib.Path) -> None:
    rows = data["points"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for fam in ("GLIMPSE (learned)", "GLIMPSE (OMP)", "GLIMPSE (LS)"):
        pts = sorted((r["bits"], r["sgcs"], r["sgcs_sem"])
                     for r in rows if r["family"] == fam)
        if not pts:
            continue
        b, s, e = zip(*pts)
        st = STYLE[fam]
        ax.errorbar(b, s, yerr=e, color=st["color"], marker=st["marker"],
                    capsize=2, lw=1.2, label=fam.replace("GLIMPSE ", ""))
    ax.set_xscale("log")
    ax.set_xlabel("feedback overhead (bits per report)")
    ax.set_ylabel("mean SGCS")
    ax.set_title(f"Same UE report, three gNB decoders -- CDL-{data['cdl_model']}")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def ablation_figure(data: dict, out: pathlib.Path) -> bool:
    """SGCS vs bits isolating each design factor: KLT vs random basis, and
    water-filled vs uniform bit allocation (holding the decoder fixed)."""
    rows = data["points"]
    series = [
        ("GLIMPSE (learned)", "KLT + water-fill + learned (full)", "-"),
        ("GLIMPSE (learned, uniform-B)", "KLT + uniform-B + learned", "--"),
        ("GLIMPSE (LS)", "KLT + water-fill + LS (linear)", ":"),
        ("GLIMPSE-random (OMP)", "random basis + OMP", "-."),
        ("GLIMPSE-random (LS)", "random basis + LS", ":"),
    ]
    present = [s for s in series if any(r["family"] == s[0] for r in rows)]
    if len(present) < 3:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for fam, label, ls in present:
        pts = sorted((r["bits"], r["sgcs"]) for r in rows if r["family"] == fam)
        b, s = zip(*pts)
        st = STYLE.get(fam, dict(color="k", marker="."))
        ax.plot(b, s, ls=ls, color=st["color"], marker=st["marker"], ms=5,
                lw=1.3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("feedback overhead (bits per report)")
    ax.set_ylabel("mean SGCS")
    ax.set_title(f"Ablation: basis, quantizer, decoder -- CDL-{data['cdl_model']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def generalization_figure(results: pathlib.Path, models: list[str], bits_target: int,
                          out: pathlib.Path) -> dict | None:
    """SGCS across CDL profiles at ~matched bits: GLIMPSE (learned) vs the best
    codebook, showing cross-distribution generalization (CDL-D/E are near-LoS,
    outside the CDL-A/B/C training mix)."""
    def nearest(rows, family, bits):
        cand = [r for r in rows if r["family"] == family]
        return min(cand, key=lambda r: abs(r["bits"] - bits)) if cand else None

    have = []
    for mdl in models:
        p = results / f"frontier_cdl{mdl}.json"
        if p.exists():
            have.append((mdl, load(p)["points"]))
    if len(have) < 2:
        return None
    gl, cb, cb_bits, gl_bits = [], [], [], []
    for _, rows in have:
        g = nearest(rows, "GLIMPSE (learned)", bits_target)
        cbest = max((r for r in rows if r["family"] in CODEBOOK_FAMILIES
                     and r["bits"] <= bits_target * 1.15), key=lambda r: r["sgcs"],
                    default=None)
        gl.append(g["sgcs"] if g else np.nan)
        gl_bits.append(g["bits"] if g else np.nan)
        cb.append(cbest["sgcs"] if cbest else np.nan)
        cb_bits.append(cbest["bits"] if cbest else np.nan)
    labels = [m for m, _ in have]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, cb, 0.4, color="#EDB120", label="best codebook (<= target bits)")
    ax.bar(x + 0.2, gl, 0.4, color=STYLE["GLIMPSE (learned)"]["color"],
           label="GLIMPSE (learned)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"CDL-{m}" for m in labels])
    ax.set_ylabel(f"mean SGCS at ~{bits_target} bits")
    ax.set_ylim(0.5, 1.0)
    ax.axvspan(2.5, len(labels) - 0.5, color="0.9", zorder=0)
    ax.text(len(labels) - 1, 0.52, "near-LoS\n(out of training mix)", fontsize=8,
            ha="center", color="0.4")
    ax.set_title(f"Cross-profile generalization (~{bits_target} bits/report)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"models": labels, "glimpse_sgcs": gl, "codebook_sgcs": cb,
            "glimpse_bits": gl_bits, "codebook_bits": cb_bits}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results/ml"))
    ap.add_argument("--cdl", default="C", help="primary CDL model for the frontier/gain figs")
    ap.add_argument("--gen-models", default="A,B,C,D,E",
                    help="CDL models for the generalization bar (those present are used)")
    ap.add_argument("--gen-bits", type=int, default=96)
    args = ap.parse_args()

    primary = args.results / f"frontier_cdl{args.cdl}.json"
    data = load(primary)
    frontier_figure(data, args.results / "fig_glimpse_frontier.png")
    gain = gain_figure(data, args.results / "fig_glimpse_gain.png")
    decoder_figure(data, args.results / "fig_glimpse_decoders.png")
    if ablation_figure(data, args.results / "fig_glimpse_ablation.png"):
        print("wrote ablation figure")
    (args.results / "glimpse_gain_summary.json").write_text(json.dumps(gain, indent=2))
    gen = generalization_figure(args.results, args.gen_models.split(","),
                                args.gen_bits, args.results / "fig_glimpse_models.png")
    if gen:
        (args.results / "glimpse_generalization.json").write_text(json.dumps(gen, indent=2))
    print("wrote figures to", args.results)
    print("overhead reduction vs best codebook:")
    for t, r in zip(gain["targets"], gain["reduction_pct"]):
        print(f"  SGCS>={t}: -{r:.0f}%")


if __name__ == "__main__":
    main()
