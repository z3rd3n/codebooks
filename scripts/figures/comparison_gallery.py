"""Codebook-comparison gallery: restyled, annotated figures from the gallery data.

Reads the JSON data files the main gallery (``make_all_figures.py``) already
produced under ``results/`` and renders a compact, presentation-grade
comparison set into ``results/comparison/`` — no Monte-Carlo re-runs, so the
numbers here are bit-identical to the main gallery's.

Design rules (shared across all panels):
* one fixed color per codebook family, everywhere;
* direct labels at the line ends instead of legend hunting where possible;
* every figure carries its one-line takeaway as a title annotation;
* the eigen/full-CSI bound is always a grey dashed reference.

Run: python scripts/figures/comparison_gallery.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "comparison"

FAMILIES = [
    "R15 Type I",
    "R15 Type II",
    "R16 eType II",
    "R17 FeType II PS",
    "R18 eType II Doppler",
]
COLOR = {
    "R15 Type I": "#1f77b4",
    "R15 Type II": "#d62728",
    "R16 eType II": "#e6a817",
    "R17 FeType II PS": "#9467bd",
    "R18 eType II Doppler": "#2ca02c",
    "bound": "#555555",
}
SHORT = {
    "R15 Type I": "Type I",
    "R15 Type II": "R15 Type II",
    "R16 eType II": "R16 eType II",
    "R17 FeType II PS": "R17 FeType II PS",
    "R18 eType II Doppler": "R18 Doppler",
}

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT / name}")


def end_label(ax, x, y, text, color, dy=0.0, fontsize=8.5):
    ax.annotate(
        text,
        (x[-1], y[-1]),
        xytext=(6, dy),
        textcoords="offset points",
        color=color,
        fontsize=fontsize,
        va="center",
        fontweight="bold",
        annotation_clip=False,
    )


# ---------------------------------------------------------------- 1: SE vs SNR
def fig_se_vs_snr():
    d = load("fig_01_se_vs_snr.json")
    snr = np.array(d["snr_db"])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharex=True)
    for ax, rank in zip(axes, ("1", "2")):
        blob = d["ranks"][rank]
        ub = np.array(blob["eigen upper bound"])
        ax.plot(snr, ub, "--", color=COLOR["bound"], lw=1.2)
        # label the bound mid-curve, above the dashed line, to keep the
        # right edge free for the scheme end-labels
        mid = len(snr) // 2
        ax.annotate(
            "eigen bound",
            (snr[mid], ub[mid]),
            xytext=(-8, 9),
            textcoords="offset points",
            color=COLOR["bound"],
            fontsize=8.5,
            fontweight="bold",
            rotation=29,
        )
        for fam in FAMILIES:
            ax.plot(snr, np.array(blob[fam]), lw=1.8, color=COLOR[fam])
        for fam, dy in (("R15 Type I", -6), ("R18 eType II Doppler", 6)):
            end_label(ax, snr, np.array(blob[fam]), SHORT[fam], COLOR[fam], dy=dy)
        gap_t1 = ub[-1] - np.array(blob["R15 Type I"])[-1]
        ax.annotate(
            f"Type I gap at 30 dB: {gap_t1:.1f} b/s/Hz",
            (snr[-1], np.array(blob["R15 Type I"])[-1] + gap_t1 / 2),
            xytext=(0.56, 0.14),
            textcoords="axes fraction",
            fontsize=8.5,
            color=COLOR["R15 Type I"],
            arrowprops=dict(arrowstyle="->", color=COLOR["R15 Type I"], lw=0.7,
                            connectionstyle="arc3,rad=0.15"),
        )
        ax.set_title(f"rank {rank}", loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("SNR (dB)")
        ax.set_xlim(snr[0] - 1, snr[-1] + 6)  # room for end labels
    axes[0].set_ylabel("spectral efficiency (b/s/Hz)")
    handles = [plt.Line2D([], [], color=COLOR[f], lw=1.8, label=SHORT[f]) for f in FAMILIES]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8)
    fig.suptitle(
        "SU spectral efficiency vs SNR — (4,2) array P=16, N₃=8, sparse 4-ray channel",
        y=1.06,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.965,
        "Type II variants ride the eigen bound; Type I pays a fixed beam-quantization gap "
        "at rank 1 and roughly double at rank 2 (rigid orthogonal layer pair)",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    save(fig, "cmp_1_se_vs_snr.png")


# ------------------------------------------------------- 2: rate-distortion
def fig_rate_distortion():
    d = load("fig_02_rate_distortion.json")
    pts = d["points"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fam_pts: dict[str, list] = {}
    for p in pts:
        fam_pts.setdefault(p["family"], []).append(p)
    for fam, ps in fam_pts.items():
        ps.sort(key=lambda p: p["bits"])
        b = [p["bits"] for p in ps]
        s = [p["sgcs"] for p in ps]
        ax.plot(b, s, "o", color=COLOR[fam], ms=5, alpha=0.85, label=SHORT[fam])
        # per-family running best (its own frontier)
        best = np.maximum.accumulate(s)
        ax.plot(b, best, "-", color=COLOR[fam], lw=1.1, alpha=0.5)
    # global Pareto frontier
    allp = sorted(pts, key=lambda p: p["bits"])
    fb, fs, cur = [], [], -1.0
    for p in allp:
        if p["sgcs"] > cur:
            cur = p["sgcs"]
            fb.append(p["bits"])
            fs.append(p["sgcs"])
    ax.step(fb, fs, where="post", color="k", lw=1.4, ls=":", label="Pareto frontier")
    ax.fill_between(fb, fs, 1.0, step="post", color="k", alpha=0.05)
    ax.annotate(
        "an ML scheme must land in here\n(more fidelity for fewer bits)",
        (fb[1] * 1.1, min(0.985, fs[-1] + 0.045)),
        fontsize=8.5,
        color="k",
        ha="left",
        va="top",
    )
    best16 = max((p for p in fam_pts["R16 eType II"]), key=lambda p: p["sgcs"])
    ax.annotate(
        f"R16 pc-sweep tops out:\nSGCS {best16['sgcs']:.3f} @ {best16['bits']:.0f} b",
        (best16["bits"], best16["sgcs"]),
        xytext=(10, -30),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.7),
    )
    ax.set_xscale("log")
    ax.set_xlabel("feedback overhead (bits per report, log)")
    ax.set_ylabel("mean SGCS (rank 1)")
    ax.set_title(
        "Rate–distortion: every marker is one spec configuration — R16 owns the frontier; "
        "R15 pays per-subband, R17 pays port overhead",
        loc="left",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "cmp_2_rate_distortion.png")


# ------------------------------------------------------- 3: overhead anatomy
CATEGORY = {  # PMI element -> semantic category
    "i11": "spatial basis", "i12": "spatial basis",
    "i15": "delay basis", "i16": "delay basis",
    "i110": "Doppler basis",
    "i13": "selection", "i17": "selection", "i18": "selection",
    "i14": "amplitudes", "i22": "amplitudes", "i23": "amplitudes", "i24": "amplitudes",
    "i2": "phases", "i21": "phases", "i25": "phases",
}
CAT_ORDER = ["spatial basis", "delay basis", "Doppler basis", "selection", "amplitudes", "phases"]
CAT_COLOR = {
    "spatial basis": "#1f77b4", "delay basis": "#2ca02c", "Doppler basis": "#17becf",
    "selection": "#9467bd", "amplitudes": "#d62728", "phases": "#e6a817",
}


def fig_overhead_anatomy():
    d = load("fig_03_overhead_breakdown.json")
    fams = [f for f in FAMILIES if f in d]
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ypos = np.arange(len(fams))[::-1]
    labeled: set[str] = set()
    for y, fam in zip(ypos, fams):
        cat_bits = {c: 0 for c in CAT_ORDER}
        for el, b in d[fam].items():
            cat_bits[CATEGORY[el]] += b
        total = sum(cat_bits.values())
        x = 0
        for c in CAT_ORDER:
            b = cat_bits[c]
            if not b:
                continue
            ax.barh(y, b, left=x, color=CAT_COLOR[c], height=0.62,
                    label=None if c in labeled else c)
            labeled.add(c)
            if b > 45:  # label only segments wide enough to hold the text
                ax.text(x + b / 2, y, f"{100 * b / total:.0f}%", ha="center",
                        va="center", fontsize=7.5, color="white", fontweight="bold")
            x += b
        ax.text(x + 8, y, f"{total} b", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(ypos, [SHORT[f] for f in fams])
    ax.set_xlabel("feedback bits per report (rank 2, N₃ = 18)")
    ax.set_title(
        "Where the bits go — R15 is ~94% per-subband values; "
        "R16/R18 trade them for basis + bitmap + compact coefficients",
        loc="left",
        fontsize=9.5,
        pad=14,
    )
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(c) for c in CAT_ORDER if c in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="y", visible=False)
    save(fig, "cmp_3_overhead_anatomy.png")


# ------------------------------------------------------- 4: scaling laws
def fig_scaling_laws():
    d = load("fig_04_overhead_scaling.json")
    panels = [
        ("N3 (PMI frequency units)", "N₃ (frequency granularity)",
         "R15 grows linearly (~38 b per unit),\nR16/R18 sub-linearly (M_v = ⌈N₃/4⌉)"),
        ("L (spatial bases per polarization)", "L (beams per polarization)",
         "all grow linearly in L —\nspatial cost is never compressed"),
        ("N4 (slot intervals covered)", "N₄ (slot intervals covered)",
         "R15/R16 re-report ×N₄;\nR18 sends ONE predicted report"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
    for ax, (key, xlabel, note) in zip(axes, panels):
        blob = d[key]
        x = blob["x"]
        for fam in ("R15 Type II", "R16 eType II", "R18 eType II Doppler"):
            ax.plot(x, blob[fam], "o-", color=COLOR[fam], lw=1.7, ms=4)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=8, va="top")
    axes[0].set_ylabel("feedback bits (log)")
    r18 = d["N4 (slot intervals covered)"]["R18 eType II Doppler"]
    axes[2].annotate(
        f"flat: {r18[0]} → {r18[-1]} b",
        (d["N4 (slot intervals covered)"]["x"][-1], r18[-1]),
        xytext=(-66, 14), textcoords="offset points", fontsize=8.5,
        color=COLOR["R18 eType II Doppler"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=0.7, color=COLOR["R18 eType II Doppler"]),
    )
    handles = [plt.Line2D([], [], color=COLOR[f], lw=1.7, marker="o", ms=4, label=SHORT[f])
               for f in ("R15 Type II", "R16 eType II", "R18 eType II Doppler")]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.84))
    fig.suptitle(
        "Overhead scaling laws — (16,1) array, rank 2, spec bit compositions "
        "(K^NZ = K₀ budget convention)",
        y=1.03, fontsize=11,
    )
    save(fig, "cmp_4_scaling_laws.png")


# ------------------------------------------------------- 5: mobility
def fig_mobility():
    d = load("fig_05_mobility.json")
    per = d["per_interval"]
    aging = d["aging"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    fig.subplots_adjust(wspace=0.3)

    ax = axes[0]
    slots = np.arange(len(per["R15 Type II"]))
    greens = {"R18 eType II Doppler N4=2": 0.35, "R18 eType II Doppler N4=4": 0.6,
              "R18 eType II Doppler N4=8": 1.0}
    for name, vals in per.items():
        if name.startswith("R18"):
            a = greens[name]
            n4 = int(name.split("N4=")[1])
            ax.plot(slots, vals, "v-", color=COLOR["R18 eType II Doppler"], alpha=a,
                    lw=1.7, ms=4, label=f"R18 N₄={n4}")
            # dotted marker where this report's predicted window ends
            ax.axvline(n4 - 1, color=COLOR["R18 eType II Doppler"], alpha=0.45 * a,
                       ls=":", lw=1.1)
        else:
            ax.plot(slots, vals, "s-", color=COLOR[name], lw=1.7, ms=4,
                    label=SHORT[name])
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlabel("slot interval s after the (single) report")
    ax.set_ylabel("mean SGCS at interval s")
    ax.set_title(
        "one report under Doppler: a held R15/R16 precoder decays;\n"
        "R18 stays flat up to its predicted window end (dotted)",
        loc="left", fontsize=9,
    )

    ax = axes[1]
    delays = aging["delays"]
    for name in FAMILIES + ["R18 eType II Doppler (delay-aware)"]:
        vals = aging[name]
        base = name.replace(" (delay-aware)", "")
        ls = ":" if "delay-aware" in name else "-"
        ax.plot(delays, vals, ls, color=COLOR[base], lw=1.7,
                marker="o" if ls == "-" else None, ms=3.5)
    end_label(ax, delays, aging["R18 eType II Doppler (delay-aware)"],
              "R18 delay-aware", COLOR["R18 eType II Doppler"], dy=4, fontsize=8)
    end_label(ax, delays, aging["R15 Type I"], "Type I (flat but low)",
              COLOR["R15 Type I"], dy=-2, fontsize=8)
    ax.set_xlabel("feedback delay (slot intervals)")
    ax.set_ylabel("mean SGCS over the scoring window")
    ax.set_title(
        "CSI aging: everyone decays alike when the gNB replays stale reports —\n"
        "R18's gain needs the gNB to apply the *predicted* interval (dotted)",
        loc="left", fontsize=9,
    )
    fig.suptitle("Mobility — off-grid Doppler over an 8-interval period, (4,2) array, N₃=8",
                 y=1.05, fontsize=11)
    save(fig, "cmp_5_mobility.png")


# ------------------------------------------------------- 6: scorecard heatmap
def fig_scorecard():
    d = load("fig_12_summary.json")["raw"]
    metrics = [
        ("SE@10dB rank1", "SE@10dB r1\n(b/s/Hz)", False),
        ("SE@10dB rank2", "SE@10dB r2\n(b/s/Hz)", False),
        ("SGCS rank1", "SGCS r1", False),
        ("mobility SGCS", "mobility\nSGCS", False),
        ("bits", "bits/report", True),  # lower is better
    ]
    fams = FAMILIES
    vals = np.array([[d[f][m] for m, _, _ in metrics] for f in fams])
    norm = np.zeros_like(vals)
    for j, (_, _, lower_better) in enumerate(metrics):
        col = vals[:, j]
        n = (col - col.min()) / (col.max() - col.min())
        norm[:, j] = 1 - n if lower_better else n
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    im = ax.imshow(norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(fams)):
        for j, (m, _, _) in enumerate(metrics):
            v = vals[i, j]
            ax.text(j, i, f"{v:.0f}" if m == "bits" else f"{v:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black")
    ax.set_xticks(range(len(metrics)), [lbl for _, lbl, _ in metrics], fontsize=8.5)
    ax.set_yticks(range(len(fams)), [SHORT[f] for f in fams], fontsize=9)
    ax.set_title(
        "Scorecard — green = best in column, red = worst (bits inverted: fewer is greener).\n"
        "R16 is the balanced default; R18 buys mobility with bits; R17 buys bits with fidelity; "
        "Type I is the 23-bit floor",
        loc="left", fontsize=9,
    )
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="min–max normalized (per column)")
    save(fig, "cmp_6_scorecard.png")


def main() -> None:
    fig_se_vs_snr()
    fig_rate_distortion()
    fig_overhead_anatomy()
    fig_scaling_laws()
    fig_mobility()
    fig_scorecard()


if __name__ == "__main__":
    main()
