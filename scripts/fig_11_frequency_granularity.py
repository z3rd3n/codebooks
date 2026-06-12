"""Fig 11 -- Frequency granularity: fidelity and cost vs N3.

N3 swept 4..32 with the *physical* selectivity held fixed (ray delays drawn
up to 0.375*N3 taps, i.e. a constant fraction of the band), so only the
reporting granularity changes:

* left: mean SGCS vs N3 -- R15 quantizes every subband independently and
  holds fidelity; R16's M_v = ceil(p_v*N3) DFT taps track the (sparse)
  delay structure at a fraction of the elements; Type I's single wideband
  beam + per-subband co-phase is the floor.
* right: measured bits per report vs N3 (log) -- R15 grows linearly in N3
  (the per-subband i2), R16's tap indicators grow ~logarithmically.
  N3 = 24, 32 exercise the R16 two-level (i15/i16) tap indication path.

Run: python scripts/fig_11_frequency_granularity.py -> results/fig_11_frequency_granularity.png
"""

import matplotlib.pyplot as plt
from figlib import ANT, cli, default_channel, run_eval, save, style

from nr_csi.codebooks import R15Type2Codebook, R16Type2Codebook, Type1Codebook

N3_GRID = [4, 8, 12, 18, 24, 32]
DELAY_FRACTION = 0.375  # max ray delay as a fraction of N3 (3 taps at N3=8)


def schemes_for(n3: int) -> list:
    return [
        Type1Codebook(ANT, N3=n3),
        R15Type2Codebook(ANT, N3=n3, L=4),
        R16Type2Codebook(ANT, N3=n3, param_combination=4),  # L=4, p_v=1/4, beta=1/2
        R16Type2Codebook(ANT, N3=n3, param_combination=6),  # L=4, p_v=1/2, beta=1/2
    ]


def label_of(scheme) -> str:
    if isinstance(scheme, R16Type2Codebook):
        return f"R16 eType II pc{scheme.combo.index} (p_v={scheme.combo.p_v12})"
    return scheme.name


def main() -> None:
    args = cli(__doc__, drops=60)
    labels = [label_of(s) for s in schemes_for(8)]
    fid = {lab: [] for lab in labels}
    bits = {lab: [] for lab in labels}
    for n3 in N3_GRID:
        chan = default_channel(n3=n3, max_delay=DELAY_FRACTION * n3)
        for scheme, lab in zip(schemes_for(n3), labels):
            res = run_eval(scheme, chan, "antenna", seed=args.seed,
                           snr_db=[10.0], rank=1, n_drops=args.drops)
            fid[lab].append(res.sgcs)
            bits[lab].append(res.overhead_bits)
        print(f"N3={n3:>2}  " + "  ".join(f"{lab}: {fid[lab][-1]:.3f}/{bits[lab][-1]:.0f}b"
                                          for lab in labels))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for lab in labels:
        st = style(lab)
        if "pc4" in lab:
            st["alpha"] = 0.55
        axes[0].plot(N3_GRID, fid[lab], label=lab, **st)
        axes[1].plot(N3_GRID, bits[lab], label=lab, **st)
    axes[0].set_xlabel("N3 (PMI frequency units)")
    axes[0].set_ylabel("mean SGCS")
    axes[0].set_title("fidelity vs reporting granularity")
    axes[1].set_xlabel("N3 (PMI frequency units)")
    axes[1].set_ylabel("measured bits per report")
    axes[1].set_yscale("log")
    axes[1].set_title("overhead vs granularity (R15 linear, R16 ~log)")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"Frequency-domain compression -- (4,2) array, rank 1, "
                 f"delay spread = {DELAY_FRACTION:.0%} of band, {args.drops} drops")
    save(fig, args.out, "fig_11_frequency_granularity",
         {"n3": N3_GRID, "sgcs": fid, "bits": bits})


if __name__ == "__main__":
    main()
