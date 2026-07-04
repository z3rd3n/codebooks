"""Fig 02 -- Rate-distortion plane: fidelity vs feedback overhead.

Every configuration knob of every family swept on the same channel drops:

* Type I: codebook modes 1 and 2;
* R15 Type II: L in {2,3,4} x N_PSK in {4,8} x subband amplitude on/off;
* R16 eType II: paramCombination 1..8 (L, p_v, beta);
* R17 FeType II: paramCombination 1..8 (alpha, M, beta), via the DFT PEB.

Left panel: mean SGCS vs mean feedback bits per report; right: SE@10 dB vs
bits.  The Pareto frontier over all points is highlighted -- this is the
plane a learned CSI feedback scheme has to beat (R18 lives on the mobility
axis and is compared in fig_05/fig_04 instead; on a static channel its
report is R16's plus Doppler bits).

Run: python scripts/figures/fig_02_rate_distortion.py -> results/fig_02_rate_distortion.png
"""

import itertools

import matplotlib.pyplot as plt

from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    Type1Codebook,
)
from nr_csi.figtools.figlib import (
    ANT,
    N3,
    ant_tag,
    cli,
    default_channel,
    run_eval,
    save,
    select_families,
    style,
)

SNR_REF_DB = 10.0


def sweep_points() -> list[tuple]:
    """(scheme, domain, short config label) for every swept configuration.

    Configurations the standard bars at the current antenna (e.g. R16
    paramCombination 7-8 need P_CSI-RS >= 32) are skipped rather than raising,
    so the figure renders for any supported array -- with fewer markers on the
    smaller ones.
    """
    pts: list[tuple] = []

    def add(make, domain: str, label: str) -> None:
        try:
            pts.append((make(), domain, label))
        except ValueError:
            pass  # configuration not supported at this antenna -- drop the point

    for m in (1, 2):
        add(lambda m=m: Type1Codebook(ANT, N3=N3, mode=m), "antenna", f"mode {m}")
    for L, n_psk, sa in itertools.product((2, 3, 4), (4, 8), (False, True)):
        add(lambda L=L, n_psk=n_psk, sa=sa: R15Type2Codebook(
            ANT, N3=N3, L=L, n_psk=n_psk, subband_amplitude=sa),
            "antenna", f"L{L} {n_psk}PSK{' SA' if sa else ''}")
    for pc in range(1, 9):
        add(lambda pc=pc: R16Type2Codebook(ANT, N3=N3, param_combination=pc),
            "antenna", f"pc{pc}")
    for pc in range(1, 9):
        add(lambda pc=pc: R17Type2Codebook(ANT, N3=N3, param_combination=pc),
            "beam", f"pc{pc}")
    return select_families(pts)


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-left frontier: fewest bits for each achieved fidelity level."""
    front: list[tuple[float, float]] = []
    best = -1.0
    for bits, fid in sorted(points):
        if fid > best:
            front.append((bits, fid))
            best = fid
    return front


def main() -> None:
    args = cli(__doc__, drops=80)
    chan = default_channel()
    rows = []
    for scheme, domain, label in sweep_points():
        res = run_eval(scheme, chan, domain, seed=args.seed,
                       snr_db=[SNR_REF_DB], rank=1, n_drops=args.drops)
        rows.append(dict(family=scheme.name, config=label, bits=res.overhead_bits,
                         sgcs=res.sgcs, se=res.se[0], se_ub=res.se_upper_bound[0]))
        print(f"{scheme.name:<18} {label:<14} bits={res.overhead_bits:6.0f} "
              f"sgcs={res.sgcs:.3f} se@10={res.se[0]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for metric, ax, ylabel in ((("sgcs"), axes[0], "mean SGCS"),
                               (("se"), axes[1], f"SE @ {SNR_REF_DB:.0f} dB (bits/s/Hz)")):
        seen = set()
        for r in rows:
            st = style(r["family"])
            st.pop("linestyle", None)
            ax.scatter(r["bits"], r[metric], s=45, color=st["color"],
                       marker=st.get("marker", "o"),
                       label=r["family"] if r["family"] not in seen else None)
            seen.add(r["family"])
        front = pareto([(r["bits"], r[metric]) for r in rows])
        ax.plot(*zip(*front), color="0.4", linestyle=":", linewidth=1.2,
                label="Pareto frontier", zorder=0)
        if metric == "se":
            ub = max(r["se_ub"] for r in rows)
            ax.axhline(ub, color="0.35", linestyle="--", linewidth=1,
                       label="eigen upper bound")
        ax.set_xscale("log")
        ax.set_xlabel("feedback overhead (bits per report)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"Rate-distortion plane, rank 1 -- {ant_tag(ANT)}, N3={N3}, {args.drops} drops "
        "(R17 via PEB; every marker = one configuration)"
    )
    save(fig, args.out, "fig_02_rate_distortion", {"points": rows})


if __name__ == "__main__":
    main()
