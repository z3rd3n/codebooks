"""Fig 10 -- Scaling with the antenna array: P = 8 .. 32 CSI-RS ports.

Supported (N1,N2) sweep {(2,2),(4,2),(6,2),(8,2)} at rank 1, all five
families (matched L = 4 for the Type II families; R17 via the unitary DFT
PEB with alpha = 1/2 so its port budget K1 = P/2 scales with the array;
R18 with N4 = 4 on the same static drops).  A sixth configuration, (16,1)
at P = 32, is overlaid as hollow markers to separate "more ports" from
"different aspect ratio" at fixed P.

* top-left: SE@10 dB vs P with the eigen upper bound -- array gain grows
  ~log2(P) for everyone;
* top-right: SE *gap to the bound* vs P -- the paper's Fig. f1 claim
  ("as the number of antennas increases, the gap becomes more
  pronounced"): a single Type I beam gets relatively narrower as the grid
  densifies, while the L-beam combinations track the bound;
* bottom-left: mean SGCS vs P -- every codebook's fidelity *decreases*
  with P (beams narrow, off-grid rays spill across more bases while L,
  K0, and the quantizers stay fixed); Type I falls fastest;
* bottom-right: feedback bits vs P -- the cost stays nearly flat: only
  the basis indicator i12 = log2 C(N1N2, L) grows (0 bits at (2,2) where
  L = N1N2 means "all beams", i.e. a complete basis), coefficient counts
  are set by L/M_v/K0, not P.

Run: python scripts/fig_10_array_scaling.py -> results/fig_10_array_scaling.png
"""

import matplotlib.pyplot as plt
from figlib import N3, cli, default_channel, run_eval, save, style

from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
)
from nr_csi.config import AntennaConfig

GEOMETRIES = [(2, 2), (4, 2), (6, 2), (8, 2)]  # P = 8, 16, 24, 32
CONTRAST = (16, 1)  # second geometry at P = 32: aspect ratio, not port count


def schemes_for(ant: AntennaConfig) -> list[tuple]:
    return [
        (Type1Codebook(ant, N3=N3), "antenna"),
        (R15Type2Codebook(ant, N3=N3, L=4), "antenna"),
        (R16Type2Codebook(ant, N3=N3, param_combination=6), "antenna"),
        (R17Type2Codebook(ant, N3=N3, param_combination=5), "beam"),  # K1 = P/2
        (R18Type2Codebook(ant, N3=N3, N4=4, param_combination=7), "antenna"),
    ]


def measure(ant: AntennaConfig, args) -> dict[str, dict]:
    chan = default_channel(ant=ant)
    out: dict[str, dict] = {}
    for scheme, domain in schemes_for(ant):
        res = run_eval(scheme, chan, domain, seed=args.seed, antenna=ant,
                       snr_db=[10.0], rank=1, n_drops=args.drops)
        out[scheme.name] = dict(se=res.se[0], ub=res.se_upper_bound[0],
                                sgcs=res.sgcs, bits=res.overhead_bits)
    return out


def main() -> None:
    args = cli(__doc__, drops=50)
    ports, sweep = [], []
    for n1, n2 in GEOMETRIES:
        ant = AntennaConfig.standard(n1, n2)
        ports.append(ant.P)
        sweep.append(measure(ant, args))
        print(f"P={ant.P:>2} ({n1},{n2})  " + "  ".join(
            f"{n}: se={d['se']:.2f} sgcs={d['sgcs']:.3f}" for n, d in sweep[-1].items()))
    contrast = measure(AntennaConfig.standard(*CONTRAST), args)
    print(f"P=32 {CONTRAST}  " + "  ".join(
        f"{n}: se={d['se']:.2f} sgcs={d['sgcs']:.3f}" for n, d in contrast.items()))

    names = list(sweep[0])
    ub = [pt["R15 Type I"]["ub"] for pt in sweep]  # same drops for 1-slot schemes
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    panels = {
        "se": (axes[0, 0], "SE @ 10 dB (bits/s/Hz)", "beamforming gain vs array size"),
        "gap": (axes[0, 1], "SE gap to eigen bound (bits/s/Hz)",
                "quantization loss grows with the array"),
        "sgcs": (axes[1, 0], "mean SGCS", "precoder fidelity vs array size"),
        "bits": (axes[1, 1], "feedback bits per report", "overhead vs array size"),
    }
    for name in names:
        st = style(name)
        vals = {
            "se": [pt[name]["se"] for pt in sweep],
            "gap": [pt[name]["ub"] - pt[name]["se"] for pt in sweep],
            "sgcs": [pt[name]["sgcs"] for pt in sweep],
            "bits": [pt[name]["bits"] for pt in sweep],
        }
        cvals = {
            "se": contrast[name]["se"],
            "gap": contrast[name]["ub"] - contrast[name]["se"],
            "sgcs": contrast[name]["sgcs"],
            "bits": contrast[name]["bits"],
        }
        for key, (ax, _, _) in panels.items():
            ax.plot(ports, vals[key], label=name, **st)
            ax.scatter([32.6], [cvals[key]], facecolors="none",
                       edgecolors=st["color"], marker=st.get("marker", "o"), s=55)
    axes[0, 0].plot(ports, ub, label="eigen upper bound", **style("eigen upper bound"))
    axes[1, 1].set_yscale("log")
    for key, (ax, ylabel, title) in panels.items():
        ax.set_xlabel("CSI-RS ports P = 2 N1 N2   (hollow marker: (16,1) at P=32)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(ports)
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Array scaling -- (N1,N2) in {GEOMETRIES} + {CONTRAST}, rank 1, "
                 f"N3=8, {args.drops} drops (R17 via PEB, K1=P/2; R18 N4=4)")
    fig.tight_layout()
    save(fig, args.out, "fig_10_array_scaling", {
        "ports": ports,
        "sweep": {name: {k: [pt[name][k] for pt in sweep] for k in ("se", "ub", "sgcs", "bits")}
                  for name in names},
        "contrast_16x1": contrast,
        "se_upper_bound": ub,
    })


if __name__ == "__main__":
    main()
