"""Camera-ready figures for the PRISM paper: PRISM vs the 3GPP codebooks.

Consumes ``results/prism/frontier_mixed.json`` (deployment-level mixed-profile
bank) and the per-profile ``frontier_cdl*.json``; renders:

* ``fig_paper_frontier``  -- SGCS + SE vs bits on the MIXED bank: every
  codebook configuration, the codebook Pareto frontier, PRISM K=4.
* ``fig_paper_profiles``  -- mean SGCS at ~matched bits: mixed bank + each
  profile, best codebook vs PRISM K=4.
* ``fig_paper_ablation``  -- left: mixture size K on the mixed bank;
  right: unseen-profile test on CDL-E (no-E variant vs codebook Pareto).

    .venv/bin/python scripts/prism/make_paper_figures.py --results results/prism
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
CB_STYLE = {
    "R15 Type I": dict(color="#0072BD", marker="o"),
    "R15 Type II": dict(color="#D95319", marker="s"),
    "R16 eType II": dict(color="#EDB120", marker="^"),
    "R17 FeType II PS": dict(color="#7E2F8E", marker="D"),
}
PRISM_STYLE = {
    "PRISM K4": dict(color="#C4151C", marker="o"),
    "PRISM K1 (pooled)": dict(color="#888833", marker="v"),
    "PRISM K2": dict(color="#DD7711", marker="^"),
    "PRISM K8": dict(color="#4B0082", marker="s"),
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


def nearest(rows, family, bits):
    cand = [r for r in rows if r["family"] == family]
    return min(cand, key=lambda r: abs(r["bits"] - bits)) if cand else None


def best_codebook_at(rows, bits, tol=1.15):
    cand = [r for r in rows if r["family"] in CODEBOOK_FAMILIES
            and r["bits"] <= bits * tol]
    return max(cand, key=lambda r: r["sgcs"]) if cand else None


def frontier_figure(data: dict, out: pathlib.Path) -> None:
    rows = data["points"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for metric, ax, ylabel in ((lambda r: r["sgcs"], axes[0], "mean SGCS"),
                               (lambda r: r["se"]["10.0"], axes[1],
                                "SE @ 10 dB (bits/s/Hz)")):
        for fam in CODEBOOK_FAMILIES:
            st = CB_STYLE[fam]
            pts = [(r["bits"], metric(r)) for r in rows if r["family"] == fam]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.scatter(xs, ys, s=42, color=st["color"], marker=st["marker"],
                           zorder=3, label=fam, alpha=0.9)
        cb = [(r["bits"], metric(r)) for r in rows if r["family"] in CODEBOOK_FAMILIES]
        ax.plot(*zip(*pareto(cb)), color="0.55", ls=":", lw=1.6,
                label="codebook Pareto", zorder=2)
        pts = sorted((r["bits"], metric(r)) for r in rows if r["family"] == "PRISM K4")
        xs, ys = zip(*pts)
        st = PRISM_STYLE["PRISM K4"]
        ax.scatter(xs, ys, s=75, color=st["color"], marker=st["marker"], zorder=5,
                   edgecolors="k", linewidths=0.5, label="PRISM (K=4)")
        ax.plot(*zip(*pareto(pts)), color=st["color"], lw=1.6, alpha=0.7, zorder=4)
        if ylabel.startswith("SE"):
            ub = np.mean([r["se_ub"]["10.0"] for r in rows])
            ax.axhline(ub, color="0.35", ls="--", lw=1, label="eigen upper bound")
        ax.set_xscale("log")
        ax.set_xlabel("feedback overhead (bits per report)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="lower right")
    g = data["geometry"]
    fig.suptitle(f"PRISM vs 3GPP codebooks -- mixed CDL-A..E bank, "
                 f"({g['N1']},{g['N2']}) P={g['P']}, N3={g['N3']}, "
                 f"rank {data['rank']}, {data['drops']} drops")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def profiles_figure(results: pathlib.Path, models: list[str], bits_target: int,
                    out: pathlib.Path) -> dict:
    labels, cb_vals, pr_vals = [], [], []
    for name, path in ([("mixed", results / "frontier_mixed.json")]
                       + [(m, results / f"frontier_cdl{m}.json") for m in models]):
        if not path.exists():
            continue
        rows = load(path)["points"]
        labels.append(name)
        cb = best_codebook_at(rows, bits_target)
        pr = nearest(rows, "PRISM K4", bits_target)
        cb_vals.append(cb["sgcs"] if cb else np.nan)
        pr_vals.append(pr["sgcs"] if pr else np.nan)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.19, cb_vals, 0.38, color="#EDB120", label="best 3GPP codebook")
    ax.bar(x + 0.19, pr_vals, 0.38, color="#C4151C", label="PRISM (K=4)")
    for i, (c, p) in enumerate(zip(cb_vals, pr_vals)):
        ax.text(i + 0.19, p + 0.004, f"{p:.3f}", ha="center", fontsize=8,
                color="#C4151C")
        ax.text(i - 0.19, c + 0.004, f"{c:.3f}", ha="center", fontsize=8,
                color="#7a6a10")
    ax.set_xticks(x)
    ax.set_xticklabels(["mixed\nA..E" if l == "mixed" else f"CDL-{l}"
                        for l in labels])
    ax.set_ylabel(f"mean SGCS at ~{bits_target} bits")
    ax.set_ylim(0.80, 1.005)
    ax.set_title(f"Deployment mix and every profile (~{bits_target} bits/report)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"labels": labels, "codebook": cb_vals, "prism": pr_vals}


def ablation_figure(mixed: dict, e_data: dict, out: pathlib.Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for fam in ("PRISM K1 (pooled)", "PRISM K2", "PRISM K4", "PRISM K8"):
        pts = sorted((r["bits"], r["sgcs"]) for r in mixed["points"]
                     if r["family"] == fam)
        if pts:
            st = PRISM_STYLE[fam]
            label = fam.replace("PRISM ", "").replace("K", "K=")
            axes[0].plot(*zip(*pts), marker=st["marker"], ms=4, lw=1.2,
                         color=st["color"], label=label)
    cb = [(r["bits"], r["sgcs"]) for r in mixed["points"]
          if r["family"] in CODEBOOK_FAMILIES]
    axes[0].plot(*zip(*pareto(cb)), color="0.55", ls=":", lw=1.6,
                 label="codebook Pareto")
    axes[0].set_title("How many bases? (mixed CDL-A..E bank)")
    for fam in ("PRISM K1 (pooled)", "PRISM K4 (no-E)", "PRISM K4"):
        pts = sorted((r["bits"], r["sgcs"]) for r in e_data["points"]
                     if r["family"] == fam)
        if pts:
            st = PRISM_STYLE[fam]
            label = fam.replace("PRISM ", "").replace("K", "K=")
            axes[1].plot(*zip(*pts), marker=st["marker"], ms=4, lw=1.2,
                         color=st["color"], label=label)
    cbE = [(r["bits"], r["sgcs"]) for r in e_data["points"]
           if r["family"] in CODEBOOK_FAMILIES]
    axes[1].plot(*zip(*pareto(cbE)), color="0.55", ls=":", lw=1.6,
                 label="codebook Pareto")
    axes[1].set_title("Unseen profile: CDL-E (the no-E fit never saw E)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("feedback overhead (bits per report)")
        ax.set_ylabel("mean SGCS")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {fam: nearest(mixed["points"], fam, 98)["sgcs"]
            for fam in ("PRISM K1 (pooled)", "PRISM K2", "PRISM K4", "PRISM K8")}


def gain_table(rows, targets=(0.85, 0.90, 0.92, 0.95, 0.97)) -> dict:
    out = {}
    for t in targets:
        def cheapest(fams):
            ok = [r["bits"] for r in rows if r["family"] in fams and r["sgcs"] >= t]
            return min(ok) if ok else None
        out[f"{t:.2f}"] = {"codebook": cheapest(CODEBOOK_FAMILIES),
                           "prism": cheapest(["PRISM K4"])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results/prism"))
    ap.add_argument("--gen-bits", type=int, default=144)
    args = ap.parse_args()

    mixed = load(args.results / "frontier_mixed.json")
    e_data = load(args.results / "frontier_cdlE.json")
    frontier_figure(mixed, args.results / "fig_paper_frontier.png")
    prof = profiles_figure(args.results, list("ABCDE"), args.gen_bits,
                           args.results / "fig_paper_profiles.png")
    ks = ablation_figure(mixed, e_data, args.results / "fig_paper_ablation.png")
    gains = {"mixed": gain_table(mixed["points"])}
    for m in "CDE":
        gains[m] = gain_table(load(args.results / f"frontier_cdl{m}.json")["points"])
    summary = {"profiles": prof, "k_sweep_mixed_at_98b": ks, "gain": gains}
    (args.results / "paper_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
