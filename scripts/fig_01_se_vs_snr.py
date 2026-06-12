"""Fig 01 -- Spectral efficiency vs SNR for all codebook families.

The headline comparison (paper Fig. f1 generalized to every family on the
same dual-pol channel): SU achievable rate of the reported precoder vs the
per-subband eigen-beamforming upper bound, at rank 1 and rank 2, on the
benchmark sparse multipath channel.

R17 is evaluated through the unitary DFT PEB (its deployment model); R18
reports once per N4 = 4 intervals (static channel, so its curve isolates
the fidelity cost of also encoding the Doppler axis).

Run: python scripts/fig_01_se_vs_snr.py  ->  results/fig_01_se_vs_snr.png
"""

import matplotlib.pyplot as plt
import numpy as np
from figlib import cli, default_channel, run_eval, save, standard_schemes, style

SNR_DB = list(np.arange(-5.0, 31.0, 5.0))


def main() -> None:
    args = cli(__doc__, drops=100)
    chan = default_channel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    data: dict = {"snr_db": SNR_DB, "ranks": {}}

    for ax, rank in zip(axes, (1, 2)):
        results = {}
        for scheme, domain in standard_schemes():
            res = run_eval(scheme, chan, domain, seed=args.seed,
                           snr_db=SNR_DB, rank=rank, n_drops=args.drops)
            results[scheme.name] = res
            label = scheme.name + (" (via PEB)" if domain == "beam" else "")
            ax.plot(SNR_DB, res.se, label=label, **style(scheme.name))
        # upper bound from the Type I run (same drops for all 1-slot schemes)
        ub = results["R15 Type I"].se_upper_bound
        ax.plot(SNR_DB, ub, label="eigen upper bound", **style("eigen upper bound"))
        ax.set_title(f"rank {rank}")
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.3)
        data["ranks"][rank] = {
            **{name: r.se for name, r in results.items()},
            "eigen upper bound": ub,
            "sgcs": {name: r.sgcs for name, r in results.items()},
            "overhead_bits": {name: r.overhead_bits for name, r in results.items()},
        }

    axes[0].set_ylabel("spectral efficiency (bits/s/Hz)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"SU spectral efficiency vs SNR -- (4,2) array, P=16, N3=8, "
        f"{args.drops} drops, sparse 4-ray channel"
    )
    save(fig, args.out, "fig_01_se_vs_snr", data)


if __name__ == "__main__":
    main()
