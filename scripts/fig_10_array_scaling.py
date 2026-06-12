"""Fig 10 -- Scaling with the antenna array: P = 8 .. 32 CSI-RS ports.

Supported (N1,N2) sweep {(2,2),(4,2),(6,2),(8,2)} at fixed L = 4 and rank 1:

* left: SE@10 dB vs P with the eigen upper bound -- the paper-f1 claim
  that Type I's single-beam gap *grows* with the array while Type II
  tracks the bound;
* middle: mean SGCS vs P;
* right: feedback bits vs P -- the price of that tracking (R15's i2 is
  flat in P, the basis indicators grow combinatorially; R16's coefficient
  budget is set by L and M_v, not P).

Run: python scripts/fig_10_array_scaling.py -> results/fig_10_array_scaling.png
"""

import matplotlib.pyplot as plt
from figlib import N3, cli, default_channel, run_eval, save, style

from nr_csi.codebooks import R15Type2Codebook, R16Type2Codebook, Type1Codebook
from nr_csi.config import AntennaConfig

GEOMETRIES = [(2, 2), (4, 2), (6, 2), (8, 2)]  # P = 8, 16, 24, 32


def main() -> None:
    args = cli(__doc__, drops=50)
    names = ["R15 Type I", "R15 Type II", "R16 eType II"]
    se = {n: [] for n in names}
    fid = {n: [] for n in names}
    bits = {n: [] for n in names}
    ub = []
    ports = []
    for n1, n2 in GEOMETRIES:
        ant = AntennaConfig.standard(n1, n2)
        ports.append(ant.P)
        chan = default_channel(ant=ant)
        schemes = [
            Type1Codebook(ant, N3=N3),
            R15Type2Codebook(ant, N3=N3, L=4),
            R16Type2Codebook(ant, N3=N3, param_combination=6),
        ]
        for scheme in schemes:
            res = run_eval(scheme, chan, "antenna", seed=args.seed, antenna=ant,
                           snr_db=[10.0], rank=1, n_drops=args.drops)
            se[scheme.name].append(res.se[0])
            fid[scheme.name].append(res.sgcs)
            bits[scheme.name].append(res.overhead_bits)
            if scheme.name == "R15 Type I":
                ub.append(res.se_upper_bound[0])
        print(f"P={ant.P:>2}  " + "  ".join(
            f"{n}: se={se[n][-1]:.2f} sgcs={fid[n][-1]:.3f} bits={bits[n][-1]:.0f}"
            for n in names))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for n in names:
        axes[0].plot(ports, se[n], label=n, **style(n))
        axes[1].plot(ports, fid[n], label=n, **style(n))
        axes[2].plot(ports, bits[n], label=n, **style(n))
    axes[0].plot(ports, ub, label="eigen upper bound", **style("eigen upper bound"))
    axes[0].set_ylabel("SE @ 10 dB (bits/s/Hz)")
    axes[0].set_title("beamforming gain vs array size")
    axes[1].set_ylabel("mean SGCS")
    axes[1].set_title("precoder fidelity vs array size")
    axes[2].set_ylabel("feedback bits per report")
    axes[2].set_yscale("log")
    axes[2].set_title("overhead vs array size")
    for ax in axes:
        ax.set_xlabel("CSI-RS ports P = 2 N1 N2")
        ax.set_xticks(ports)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"Array scaling -- (N1,N2) in {GEOMETRIES}, L=4, rank 1, N3=8, "
                 f"{args.drops} drops")
    save(fig, args.out, "fig_10_array_scaling",
         {"ports": ports, "se": se, "sgcs": fid, "bits": bits, "se_upper_bound": ub})


if __name__ == "__main__":
    main()
