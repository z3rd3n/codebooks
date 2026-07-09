"""Publication figures: PRISM vs GLIMPSE vs the 3GPP codebook families.

Consumes ``results/prism/frontier_cdl*.json`` (stored codebook + GLIMPSE rows
merged with the PRISM rows) and renders:

* ``fig_prism_frontier``   -- SGCS + SE vs bits on the primary CDL: codebook
  Pareto, GLIMPSE (learned), PRISM K4, PRISM K1 (pooled).
* ``fig_prism_profiles``   -- the headline: SGCS at ~matched bits across
  CDL-A..E for best codebook / GLIMPSE / PRISM K4 (the OOD fix).
* ``fig_prism_ablation``   -- mixture-size sweep K in {1,2,4,8} on the
  primary CDL + the no-E variant on CDL-E (regime vs profile coverage).
* ``fig_prism_gain``       -- bits to reach a target SGCS: PRISM vs the best
  codebook and vs GLIMPSE.

    .venv/bin/python scripts/prism/make_prism_figures.py --results results/prism
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CODEBOOK_FAMILIES = ["R15 Type I", "R15 Type II", "R16 eType II", "R17 FeType II PS"]
STYLE = {
    "GLIMPSE (learned)": dict(color="#119911", marker="*"),
    "PRISM K4": dict(color="#C4151C", marker="o"),
    "PRISM K1 (pooled)": dict(color="#888833", marker="v"),
    "PRISM K2": dict(color="#DD7711", marker="^"),
    "PRISM K8": dict(color="#7E2F8E", marker="s"),
    "PRISM K4 (no-E)": dict(color="#0072BD", marker="D"),
}


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def pareto(points):
    front, best = [], -1.0
    for bits, fid in sorted(points):
        if fid > best:
            front.append((bits, fid))
            best = fid
    return front


def frontier_figure(data: dict, out: pathlib.Path) -> None:
    rows = data["points"]
    fams = ["GLIMPSE (learned)", "PRISM K1 (pooled)", "PRISM K4"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for metric, ax, ylabel in ((lambda r: r["sgcs"], axes[0], "mean SGCS"),
                               (lambda r: r["se"]["10.0"], axes[1],
                                "SE @ 10 dB (bits/s/Hz)")):
        cb = [(r["bits"], metric(r)) for r in rows if r["family"] in CODEBOOK_FAMILIES]
        ax.plot(*zip(*pareto(cb)), color="0.55", ls=":", lw=1.6,
                label="codebook Pareto", zorder=1)
        for fam in fams:
            st = STYLE[fam]
            pts = sorted((r["bits"], metric(r)) for r in rows if r["family"] == fam)
            if not pts:
                continue
            xs, ys = zip(*pts)
            lead = fam == "PRISM K4"
            ax.scatter(xs, ys, s=70 if lead else 40, color=st["color"],
                       marker=st["marker"], zorder=5 if lead else 3,
                       edgecolors="k" if lead else "none",
                       linewidths=0.5 if lead else 0, label=fam, alpha=0.95)
            ax.plot(*zip(*pareto(pts)), color=st["color"], lw=1.4 if lead else 0.9,
                    alpha=0.65, zorder=2)
        if ylabel.startswith("SE"):
            ub = np.mean([r["se_ub"]["10.0"] for r in rows])
            ax.axhline(ub, color="0.35", ls="--", lw=1, label="eigen upper bound")
        ax.set_xscale("log")
        ax.set_xlabel("feedback overhead (bits per report)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="lower right")
    g = data["geometry"]
    fig.suptitle(f"PRISM vs GLIMPSE vs 3GPP codebooks -- CDL-{data['cdl_model']}, "
                 f"({g['N1']},{g['N2']}) P={g['P']}, N3={g['N3']}, "
                 f"rank {data['rank']}, {data['drops']} drops")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def nearest(rows, family, bits):
    cand = [r for r in rows if r["family"] == family]
    return min(cand, key=lambda r: abs(r["bits"] - bits)) if cand else None


def best_codebook_at(rows, bits, tol=1.15):
    cand = [r for r in rows if r["family"] in CODEBOOK_FAMILIES
            and r["bits"] <= bits * tol]
    return max(cand, key=lambda r: r["sgcs"]) if cand else None


def profiles_figure(results: pathlib.Path, models: list[str], bits_target: int,
                    out: pathlib.Path) -> dict:
    fams = ["GLIMPSE (learned)", "PRISM K4"]
    colors = {"best codebook": "#EDB120", "GLIMPSE (learned)": "#119911",
              "PRISM K4": "#C4151C"}
    series = {k: [] for k in ["best codebook"] + fams}
    labels = []
    for mdl in models:
        p = results / f"frontier_cdl{mdl}.json"
        if not p.exists():
            continue
        rows = load(p)["points"]
        labels.append(mdl)
        cb = best_codebook_at(rows, bits_target)
        series["best codebook"].append(cb["sgcs"] if cb else np.nan)
        for fam in fams:
            r = nearest(rows, fam, bits_target)
            series[fam].append(r["sgcs"] if r else np.nan)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    w = 0.26
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1) * w, vals, w, color=colors[name], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"CDL-{m}" for m in labels])
    ax.set_ylabel(f"mean SGCS at ~{bits_target} bits")
    ax.set_ylim(0.5, 1.0)
    ax.axvspan(2.5, len(labels) - 0.5, color="0.92", zorder=0)
    ax.text(len(labels) - 1.5, 0.53, "near-LoS (outside GLIMPSE's\ntraining mix)",
            fontsize=8, ha="center", color="0.4")
    ax.set_title(f"Cross-profile fidelity (~{bits_target} bits/report)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"models": labels, **{k: v for k, v in series.items()}}


def ablation_figure(primary: dict, e_data: dict, out: pathlib.Path) -> dict:
    """Left: K sweep frontier on the primary CDL.  Right: the no-E variant on
    CDL-E vs the full mixture and the pooled basis."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fams_l = ["PRISM K1 (pooled)", "PRISM K2", "PRISM K4", "PRISM K8"]
    for fam in fams_l:
        pts = sorted((r["bits"], r["sgcs"]) for r in primary["points"]
                     if r["family"] == fam)
        if pts:
            st = STYLE[fam]
            axes[0].plot(*zip(*pts), marker=st["marker"], ms=4, lw=1.2,
                         color=st["color"], label=fam)
    axes[0].set_title(f"Mixture size (CDL-{primary['cdl_model']})")
    fams_r = ["PRISM K1 (pooled)", "PRISM K4 (no-E)", "PRISM K4",
              "GLIMPSE (learned)"]
    for fam in fams_r:
        pts = sorted((r["bits"], r["sgcs"]) for r in e_data["points"]
                     if r["family"] == fam)
        if pts:
            st = STYLE.get(fam, dict(color="#119911", marker="*"))
            axes[1].plot(*zip(*pts), marker=st["marker"], ms=4, lw=1.2,
                         color=st["color"], label=fam)
    cb = [(r["bits"], r["sgcs"]) for r in e_data["points"]
          if r["family"] in CODEBOOK_FAMILIES]
    axes[1].plot(*zip(*pareto(cb)), color="0.55", ls=":", lw=1.6,
                 label="codebook Pareto")
    axes[1].set_title("Unseen profile: CDL-E (no-E variant never saw E)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("feedback overhead (bits per report)")
        ax.set_ylabel("mean SGCS")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    ks = {}
    for fam in fams_l:
        r = nearest(primary["points"], fam, 96)
        if r:
            ks[fam] = r["sgcs"]
    return ks


def gain_figure(data: dict, out: pathlib.Path) -> dict:
    rows = data["points"]
    targets = [0.80, 0.85, 0.90, 0.92, 0.95]

    def cheapest(fams, t):
        ok = [r["bits"] for r in rows if r["family"] in fams and r["sgcs"] >= t]
        return min(ok) if ok else None

    out_d = {"targets": [], "codebook": [], "glimpse": [], "prism": []}
    for t in targets:
        c = cheapest(CODEBOOK_FAMILIES, t)
        gl = cheapest(["GLIMPSE (learned)"], t)
        pr = cheapest(["PRISM K4"], t)
        if pr is None:
            continue
        out_d["targets"].append(f"{t:.2f}")
        out_d["codebook"].append(c)
        out_d["glimpse"].append(gl)
        out_d["prism"].append(pr)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(out_d["targets"]))
    w = 0.26
    ax.bar(x - w, [c if c else 0 for c in out_d["codebook"]], w,
           color="#EDB120", label="best codebook")
    ax.bar(x, [g if g else 0 for g in out_d["glimpse"]], w,
           color="#119911", label="GLIMPSE")
    ax.bar(x + w, out_d["prism"], w, color="#C4151C", label="PRISM K4")
    for i, (c, p) in enumerate(zip(out_d["codebook"], out_d["prism"])):
        if c:
            ax.text(i + w, p + 3, f"-{100 * (1 - p / c):.0f}%", ha="center",
                    fontsize=9, color="#C4151C", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(out_d["targets"])
    ax.set_xlabel("target mean SGCS")
    ax.set_ylabel("feedback bits to reach target")
    ax.set_title(f"Overhead at matched fidelity -- CDL-{data['cdl_model']} "
                 "(missing bar = target unreachable)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out_d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results/prism"))
    ap.add_argument("--cdl", default="C")
    ap.add_argument("--gen-models", default="A,B,C,D,E")
    ap.add_argument("--gen-bits", type=int, default=144)
    args = ap.parse_args()

    primary = load(args.results / f"frontier_cdl{args.cdl}.json")
    e_data = load(args.results / "frontier_cdlE.json")
    frontier_figure(primary, args.results / "fig_prism_frontier.png")
    prof = profiles_figure(args.results, args.gen_models.split(","), args.gen_bits,
                           args.results / "fig_prism_profiles.png")
    ks = ablation_figure(primary, e_data, args.results / "fig_prism_ablation.png")
    gain = gain_figure(primary, args.results / "fig_prism_gain.png")
    (args.results / "prism_summary.json").write_text(json.dumps(
        {"profiles": prof, "k_sweep_at_96b": ks, "gain": gain}, indent=2))
    print("wrote figures to", args.results)
    print(json.dumps({"profiles": prof, "k_sweep_at_96b": ks, "gain": gain}, indent=2))


if __name__ == "__main__":
    main()
