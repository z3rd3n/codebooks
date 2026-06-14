"""Fig 07 -- Rank (RI) adaptation: layers vs SNR.

R15 Type I and R16 eType II on a 4-rx, 6-ray channel:

* left: SE vs SNR at fixed ranks 1..4, plus the auto-RI envelope
  (``eval.select_rank`` re-decides the rank per drop and per SNR) and the
  rank-4 eigen upper bound;
* right: the rank distribution auto-RI actually picks vs SNR -- low SNR
  favors beamforming gain (rank 1), high SNR favors multiplexing.

Run: python scripts/fig_07_rank_adaptation.py -> results/fig_07_rank_adaptation.png
"""

import matplotlib.pyplot as plt
import numpy as np
from figlib import ANT, N3, cli, default_channel, save

from nr_csi.baselines import eigen_precoder
from nr_csi.codebooks import R16Type2Codebook, Type1Codebook
from nr_csi.eval.harness import select_rank
from nr_csi.metrics import su_rate

SNR_DB = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
RANKS = (1, 2, 3, 4)


def main() -> None:
    args = cli(__doc__, drops=60)
    chan = default_channel(n_rx=4, n_paths=6)
    schemes = {
        "Type I": Type1Codebook(ANT, N3=N3),
        "R16": R16Type2Codebook(ANT, N3=N3, param_combination=6),
    }

    se_fixed = {
        name: {r: np.zeros(len(SNR_DB)) for r in RANKS} for name in schemes
    }
    se_auto = {name: np.zeros(len(SNR_DB)) for name in schemes}
    se_ub = np.zeros(len(SNR_DB))
    picks = np.zeros((len(SNR_DB), len(RANKS)))
    rng = np.random.default_rng(args.seed)
    for _ in range(args.drops):
        H = chan.generate(n_slots=1, rng=rng)
        W_fixed = {
            name: {r: scheme.precoder(scheme.select(H, rank=r)) for r in RANKS}
            for name, scheme in schemes.items()
        }
        W_ub = eigen_precoder(H, rank=4)
        for i, snr in enumerate(SNR_DB):
            rho = 10 ** (snr / 10)
            for name, scheme in schemes.items():
                for r in RANKS:
                    se_fixed[name][r][i] += su_rate(H, W_fixed[name][r], rho)
                rank, _, _, se = select_rank(scheme, H, rho=rho, ranks=RANKS)
                se_auto[name][i] += se
                if name == "R16":
                    picks[i, rank - 1] += 1
            se_ub[i] += su_rate(H, W_ub, rho)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    shades = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    for r, shade in zip(RANKS, shades):
        axes[0].plot(
            SNR_DB,
            se_fixed["R16"][r] / args.drops,
            color=shade,
            marker="^",
            label=f"R16, fixed rank {r}",
        )
        axes[0].plot(
            SNR_DB,
            se_fixed["Type I"][r] / args.drops,
            color=shade,
            linestyle=":",
            label=f"Type I, fixed rank {r}",
        )
    axes[0].plot(
        SNR_DB,
        se_auto["R16"] / args.drops,
        color="#D95319",
        marker="o",
        linewidth=2,
        label="R16, auto-RI",
    )
    axes[0].plot(
        SNR_DB,
        se_auto["Type I"] / args.drops,
        color="#0072BD",
        marker="o",
        linewidth=2,
        linestyle=":",
        label="Type I, auto-RI",
    )
    axes[0].plot(SNR_DB, se_ub / args.drops, color="0.35", linestyle="--",
                 label="eigen upper bound (rank 4)")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("spectral efficiency (bits/s/Hz)")
    axes[0].set_title("fixed rank vs auto-RI")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    frac = picks / args.drops
    axes[1].stackplot(SNR_DB, frac.T, labels=[f"rank {r}" for r in RANKS],
                      colors=shades, alpha=0.9)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("fraction of drops")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("rank chosen by R16 auto-RI")
    axes[1].legend(fontsize=8, loc="center left")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Rank adaptation -- Type I vs R16 eType II pc6, 4 rx, 6-ray channel, "
                 f"{args.drops} drops")
    save(fig, args.out, "fig_07_rank_adaptation", {
        "snr_db": SNR_DB,
        "se_fixed": {
            name: {r: list(values[r] / args.drops) for r in RANKS}
            for name, values in se_fixed.items()
        },
        "se_auto": {name: list(values / args.drops) for name, values in se_auto.items()},
        "se_upper_bound": list(se_ub / args.drops),
        "rank_fractions": frac.tolist(),
    })


if __name__ == "__main__":
    main()
