"""Fig 05 -- Mobility: CSI aging vs the R18 predicted PMI.

Doppler-spread channel (off-grid Doppler up to one DFT shift over an
8-interval horizon).  Two views:

* left: per-interval SGCS of ONE report -- R15/R16 hold their precoder
  (it ages), R18 with N4 in {2,4,8} reports a *predicted* precoder per
  future interval (held beyond its window);
* right: harness-level CSI aging -- mean SGCS when the report is applied
  ``feedback_delay_slots`` intervals late (the `evaluate` knob), standard
  scheme set.

Together with fig_04's right panel (bits to cover N4 intervals) this is
the R18 trade: prediction fidelity at a fraction of the re-reporting cost.

Run: python scripts/fig_05_mobility.py -> results/fig_05_mobility.png
"""

import matplotlib.pyplot as plt
import numpy as np
from figlib import ANT, N3, cli, default_channel, run_eval, save, standard_schemes, style

from nr_csi.baselines import eigen_precoder
from nr_csi.codebooks import R15Type2Codebook, R16Type2Codebook, R18Type2Codebook
from nr_csi.metrics import sgcs

HORIZON = 8  # intervals scored in the left panel (= channel Doppler period)


def per_interval_sgcs(args) -> dict[str, np.ndarray]:
    chan = default_channel(max_doppler=1.0, doppler_period=HORIZON)
    schemes: dict[str, tuple] = {
        "R15 Type II": (R15Type2Codebook(ANT, N3=N3, L=4), 1),
        "R16 eType II": (R16Type2Codebook(ANT, N3=N3, param_combination=6), 1),
    }
    for n4 in (2, 4, 8):
        schemes[f"R18 eType II Doppler N4={n4}"] = (
            R18Type2Codebook(ANT, N3=N3, N4=n4, param_combination=7), n4)
    curves = {name: np.zeros(HORIZON) for name in schemes}
    rng = np.random.default_rng(args.seed)
    for _ in range(args.drops):
        H = chan.generate(n_slots=HORIZON, rng=rng)
        targets = eigen_precoder(H, rank=1)
        for name, (scheme, n_meas) in schemes.items():
            W = scheme.precoder(scheme.select(H[:n_meas], rank=1))  # (n_meas, ...)
            for s in range(HORIZON):
                w = W[min(s, W.shape[0] - 1)]  # hold the last (predicted) interval
                curves[name][s] += sgcs(targets[s], w)
    return {k: v / args.drops for k, v in curves.items()}


def aging_sgcs(args, delays: list[int]) -> dict[str, list[float]]:
    chan = default_channel(max_doppler=1.0, doppler_period=HORIZON)
    out: dict[str, list[float]] = {}
    for scheme, domain in standard_schemes():
        ys = []
        for d in delays:
            res = run_eval(scheme, chan, domain, seed=args.seed, snr_db=[10.0],
                           rank=1, n_drops=args.drops, feedback_delay_slots=d)
            ys.append(res.sgcs)
        out[scheme.name] = ys
    return out


def main() -> None:
    args = cli(__doc__, drops=60)
    delays = [0, 1, 2, 3, 4, 6]

    curves = per_interval_sgcs(args)
    aging = aging_sgcs(args, delays)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for name, ys in curves.items():
        st = style(name)
        if "N4=" in name:
            st["alpha"] = {2: 0.45, 4: 0.7, 8: 1.0}[int(name.split("N4=")[1])]
        axes[0].plot(range(HORIZON), ys, label=name, **st)
    axes[0].set_xlabel("slot interval s after the (single) report")
    axes[0].set_ylabel("mean SGCS at interval s")
    axes[0].set_title("one report under Doppler: held vs predicted precoders")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for name, ys in aging.items():
        axes[1].plot(delays, ys, label=name, **style(name))
    axes[1].set_xlabel("feedback delay (slot intervals)")
    axes[1].set_ylabel("mean SGCS over the scoring window")
    axes[1].set_title("CSI aging: report applied late (harness knob)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Mobility -- off-grid Doppler over a {HORIZON}-interval period, "
                 f"(4,2) array, N3=8, {args.drops} drops")
    save(fig, args.out, "fig_05_mobility",
         {"per_interval": curves, "aging": {"delays": delays, **aging}})


if __name__ == "__main__":
    main()
