"""Fig 04 -- Overhead scaling laws (Tables bit1/bit2 formulas, no Monte-Carlo).

Three panels at the paper's f2 operating point ((16,1) array, rank v = 2):

* bits vs N3: R15's per-subband i2 grows linearly; R16/R18 grow only via
  the M_v = ceil(p_v*N3/R) tap count and its log-sized indicators;
* bits vs L: combinatorial i12 + linear coefficient terms, all families;
* bits vs N4 intervals covered: R15/R16 must re-report every interval,
  R18 covers all N4 with one predicted-PMI report (the f2 convention).

Run: python scripts/figures/fig_04_overhead_scaling.py -> results/fig_04_overhead_scaling.png
"""

import matplotlib.pyplot as plt

from nr_csi.config import AntennaConfig, m_v
from nr_csi.figtools.figlib import cli, save, style
from nr_csi.metrics.overhead import r15_bits, r16_bits, r18_bits

ANT_F2 = AntennaConfig.standard(16, 1)
V = 2
P_V = "1/4"  # R16/R18 p_v for ranks 1-2 (combos 3/4-style)
BETA_TIMES_2L = 2.0  # beta = 1/2, L = 4 -> K_nz = K0 = 2*L*Mv*beta = 4*Mv


def totals_vs_n3(n3_grid: list[int], L: int = 4) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"R15 Type II": [], "R16 eType II": [], "R18 eType II Doppler": []}
    for n3 in n3_grid:
        Mv = m_v(0.25, n3, 1)
        K_nz = int(BETA_TIMES_2L * L * Mv / 2)  # K0 = ceil(beta*2L*Mv), beta=1/2
        out["R15 Type II"].append(sum(r15_bits(ANT_F2, L, V, n3).values()))
        out["R16 eType II"].append(sum(r16_bits(ANT_F2, L, V, n3, Mv, K_nz).values()))
        out["R18 eType II Doppler"].append(
            sum(r18_bits(ANT_F2, L, V, n3, Mv, 2, 4, K_nz).values())
        )
    return out


def totals_vs_l(ls: list[int], n3: int = 18) -> dict[str, list[int]]:
    Mv = m_v(0.25, n3, 1)
    out: dict[str, list[int]] = {"R15 Type II": [], "R16 eType II": [], "R18 eType II Doppler": []}
    for L in ls:
        K_nz = int(BETA_TIMES_2L * L * Mv / 2)
        out["R15 Type II"].append(sum(r15_bits(ANT_F2, L, V, n3).values()))
        out["R16 eType II"].append(sum(r16_bits(ANT_F2, L, V, n3, Mv, K_nz).values()))
        out["R18 eType II Doppler"].append(
            sum(r18_bits(ANT_F2, L, V, n3, Mv, 2, 4, K_nz).values())
        )
    return out


def totals_vs_n4(n4s: list[int], L: int = 4, n3: int = 18) -> dict[str, list[int]]:
    """Bits to cover N4 slot intervals (R15/R16 re-report, R18 predicts)."""
    Mv = m_v(0.25, n3, 1)
    K_nz = int(BETA_TIMES_2L * L * Mv / 2)
    b15 = sum(r15_bits(ANT_F2, L, V, n3).values())
    b16 = sum(r16_bits(ANT_F2, L, V, n3, Mv, K_nz).values())
    return {
        "R15 Type II": [n4 * b15 for n4 in n4s],
        "R16 eType II": [n4 * b16 for n4 in n4s],
        "R18 eType II Doppler": [
            sum(r18_bits(ANT_F2, L, V, n3, Mv, 2 if n4 > 1 else 1, n4, K_nz).values())
            for n4 in n4s
        ],
    }


def main() -> None:
    args = cli(__doc__, drops=1)
    n3_grid = [4, 8, 12, 18, 24, 36, 72]
    ls = [1, 2, 3, 4, 6]
    n4s = [1, 2, 4, 8]
    panels = [
        (totals_vs_n3(n3_grid), n3_grid, "N3 (PMI frequency units)",
         "bits per report vs frequency granularity (L=4)"),
        (totals_vs_l(ls), ls, "L (spatial bases per polarization)",
         "bits per report vs number of beams (N3=18)"),
        (totals_vs_n4(n4s), n4s, "N4 (slot intervals covered)",
         "bits to cover N4 intervals (L=4, N3=18)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    data = {}
    for ax, (curves, xs, xlabel, title) in zip(axes, panels):
        for name, ys in curves.items():
            ax.plot(xs, ys, label=name, **style(name))
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        data[xlabel] = {"x": xs, **curves}
    axes[0].set_ylabel("feedback overhead (bits)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Overhead scaling laws -- Tables bit1/bit2, (16,1) array, rank 2,"
                 " p_v=1/4, beta=1/2, N_PSK=4 (R15), Q=2")
    save(fig, args.out, "fig_04_overhead_scaling", data)
    for xlabel, d in data.items():
        print(xlabel, {k: v for k, v in d.items() if k != "x"})


if __name__ == "__main__":
    main()
