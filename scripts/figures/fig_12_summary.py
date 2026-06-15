"""Fig 12 -- Scorecard: every family on every axis, in one picture.

Five normalized axes (min-max over the five families; 1 = best here):

* SE @ 10 dB, rank 1 (static benchmark channel)
* SE @ 10 dB, rank 2
* SGCS, rank 1
* compactness = (fewest bits) / (own bits)  -- higher is cheaper
* mobility = mean SGCS on future intervals 1..3 of one report under
  Doppler (R18 predicts, the others hold)

The radar shows the *shape* of each trade; the raw numbers are written to
results/fig_12_summary_table.md (markdown) and .json -- quantifying the
paper's qualitative comparison table.  (fig_12_summary.md holds the
hand-written analysis of the figure.)

Run: python scripts/figures/fig_12_summary.py -> results/fig_12_summary.png
"""

import matplotlib.pyplot as plt
import numpy as np

from nr_csi.baselines import eigen_precoder
from nr_csi.figtools.figlib import (
    ANT,
    cli,
    default_channel,
    run_eval,
    save,
    standard_schemes,
    style,
)
from nr_csi.metrics import sgcs

AXES = ["SE@10dB rank1", "SE@10dB rank2", "SGCS rank1", "compactness", "mobility SGCS"]


def mobility_scores(args) -> dict[str, float]:
    """Mean SGCS over future intervals 1..3 of a single report under Doppler."""
    horizon = 4
    chan = default_channel(max_doppler=1.0, doppler_period=horizon)
    out: dict[str, float] = {}
    for scheme, domain in standard_schemes(N4=horizon):
        from nr_csi.figtools.figlib import BeamDomainChannel

        ch = BeamDomainChannel(chan, ANT) if domain == "beam" else chan
        n_meas = getattr(scheme, "N4", 1)
        rng = np.random.default_rng(args.seed)
        vals = []
        for _ in range(args.drops):
            H = ch.generate(n_slots=horizon, rng=rng)
            targets = eigen_precoder(H, rank=1)
            W = scheme.precoder(scheme.select(H[:n_meas], rank=1))
            vals.append(np.mean([
                sgcs(targets[s], W[min(s, W.shape[0] - 1)]) for s in range(1, horizon)
            ]))
        out[scheme.name] = float(np.mean(vals))
    return out


def main() -> None:
    args = cli(__doc__, drops=60)
    chan = default_channel()
    raw: dict[str, dict[str, float]] = {}
    for scheme, domain in standard_schemes():
        r1 = run_eval(scheme, chan, domain, seed=args.seed,
                      snr_db=[10.0], rank=1, n_drops=args.drops)
        r2 = run_eval(scheme, chan, domain, seed=args.seed,
                      snr_db=[10.0], rank=2, n_drops=args.drops)
        raw[scheme.name] = {
            "SE@10dB rank1": r1.se[0],
            "SE@10dB rank2": r2.se[0],
            "SGCS rank1": r1.sgcs,
            "bits": r1.overhead_bits,
        }
    for name, m in mobility_scores(args).items():
        raw[name]["mobility SGCS"] = m
    min_bits = min(d["bits"] for d in raw.values())
    for d in raw.values():
        d["compactness"] = min_bits / d["bits"]

    names = list(raw)
    matrix = np.array([[raw[n][a] for a in AXES] for n in names])
    lo, hi = matrix.min(axis=0), matrix.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    norm = (matrix - lo) / span

    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
    fig, ax = plt.subplots(figsize=(7.5, 7), subplot_kw=dict(polar=True))
    for i, name in enumerate(names):
        vals = np.concatenate([norm[i], norm[i][:1]])
        ang = np.concatenate([angles, angles[:1]])
        st = style(name)
        st.pop("marker", None)
        st.pop("linestyle", None)
        ax.plot(ang, vals, linewidth=1.8, **st, label=name)
        ax.fill(ang, vals, alpha=0.08, color=st["color"])
    ax.set_xticks(angles, AXES, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_title(f"Codebook scorecard (min-max normalized per axis, {args.drops} drops)",
                 pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1), fontsize=8)
    save(fig, args.out, "fig_12_summary", {"raw": raw, "normalized": norm.tolist(),
                                           "axes": AXES})

    cols = ["SE@10dB rank1", "SE@10dB rank2", "SGCS rank1", "bits",
            "compactness", "mobility SGCS"]
    lines = ["| scheme | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for n in names:
        lines.append("| " + n + " | "
                     + " | ".join(f"{raw[n][c]:.3f}" if c != "bits" else f"{raw[n][c]:.0f}"
                                  for c in cols) + " |")
    md = "\n".join(lines) + "\n"
    (args.out / "fig_12_summary_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
