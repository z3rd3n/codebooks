"""Fig 08 -- Robustness to the channel: sparsity and estimation noise.

* left: SGCS vs number of multipath rays.  Sparse channels are Type II's
  design regime (few dominant directions fit L beams / K0 coefficients);
  as the channel densifies, the fixed coefficient budgets saturate and all
  codebooks degrade -- at different rates.
* right: SGCS vs measurement SNR (complex Gaussian estimation noise added
  to the channel the UE sees, harness ``measurement_snr_db`` knob; scoring
  always uses the true channel).  R18's apparent noise robustness is its
  N4 = 4 observation slots (iid noise averages out); the dashed
  "R16 eType II (4-slot meas.)" curve gives R16 the same observation
  window via ``measurement_slots=4`` (S3) -- the fair comparison.

Standard scheme set, rank 1.

Run: python scripts/figures/fig_08_channel_sensitivity.py -> results/fig_08_channel_sensitivity.png
"""

import matplotlib.pyplot as plt

from nr_csi.figtools.figlib import (
    N3,
    ant_tag,
    cli,
    default_channel,
    run_eval,
    save,
    standard_schemes,
    style,
)

PATHS = [1, 2, 3, 4, 6, 8, 12]
MEAS_SNR_DB = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, None]  # None = noiseless


def main() -> None:
    args = cli(__doc__, drops=60)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    data: dict = {"vs_paths": {"paths": PATHS}, "vs_meas_snr": {"meas_snr_db": MEAS_SNR_DB}}

    for scheme, domain in standard_schemes():
        ys = []
        for p in PATHS:
            res = run_eval(scheme, default_channel(n_paths=p), domain, seed=args.seed,
                           snr_db=[10.0], rank=1, n_drops=args.drops)
            ys.append(res.sgcs)
        label = scheme.name + (" (via PEB)" if domain == "beam" else "")
        axes[0].plot(PATHS, ys, label=label, **style(scheme.name))
        data["vs_paths"][scheme.name] = ys

    chan_kwargs = dict(n_paths=4)
    xs = [m if m is not None else 25.0 for m in MEAS_SNR_DB]
    for scheme, domain in standard_schemes():
        ys = []
        for m in MEAS_SNR_DB:
            res = run_eval(scheme, default_channel(**chan_kwargs), domain, seed=args.seed,
                           snr_db=[10.0], rank=1, n_drops=args.drops,
                           measurement_snr_db=m)
            ys.append(res.sgcs)
        label = scheme.name + (" (via PEB)" if domain == "beam" else "")
        axes[1].plot(xs, ys, label=label, **style(scheme.name))
        data["vs_meas_snr"][scheme.name] = ys

    # fairness check (S3): give R16 the same 4-slot observation window R18
    # silently enjoys -- its single report then averages the noise too.  Skipped
    # when a family filter (NRCSI_FAMILIES) excludes R16.
    r16 = next((s for s, _ in standard_schemes() if s.name == "R16 eType II"), None)
    if r16 is not None:
        ys = []
        for m in MEAS_SNR_DB:
            res = run_eval(r16, default_channel(**chan_kwargs), "antenna", seed=args.seed,
                           snr_db=[10.0], rank=1, n_drops=args.drops,
                           measurement_snr_db=m, measurement_slots=4)
            ys.append(res.sgcs)
        st = style(r16.name)
        st["linestyle"] = "--"
        axes[1].plot(xs, ys, label="R16 eType II (4-slot meas.)", **st)
        data["vs_meas_snr"]["R16 eType II (4-slot meas.)"] = ys

    axes[0].set_xlabel("number of multipath rays")
    axes[0].set_ylabel("mean SGCS")
    axes[0].set_title("channel sparsity (Type II design regime is the left edge)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("measurement SNR (dB); rightmost point = noiseless")
    axes[1].set_ylabel("mean SGCS")
    axes[1].set_title("channel-estimation noise at the UE")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Channel sensitivity -- {ant_tag()}, N3={N3}, rank 1, {args.drops} drops")
    save(fig, args.out, "fig_08_channel_sensitivity", data)


if __name__ == "__main__":
    main()
