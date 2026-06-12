"""Fig 09 -- Applicability boundary: regular vs port-selection codebooks.

Every variant evaluated on BOTH views of the same 2-ray physical drops:

* antenna domain: the raw channel (regular codebooks' home turf);
* beam domain: the same drops through a unitary per-polarization DFT PEB
  (the port-selection deployment model -- the gNB beamforms the CSI-RS, so
  the channel is concentrated on few ports).

Regular codebooks search the DFT beam structure themselves and win on the
antenna domain; PS codebooks assume the PEB already did that and win after
it (paper "Applicable Scenarios").  Bars annotated with feedback bits.

Run: python scripts/fig_09_port_selection.py -> results/fig_09_port_selection.png
"""

import matplotlib.pyplot as plt
import numpy as np
from figlib import ANT, N3, cli, default_channel, run_eval, save

from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    Type1Codebook,
)


def schemes():
    return [
        Type1Codebook(ANT, N3=N3),
        R15Type2Codebook(ANT, N3=N3, L=4),
        R15Type2Codebook(ANT, N3=N3, L=4, port_selection=True, d=1),
        R16Type2Codebook(ANT, N3=N3, param_combination=6),
        R16Type2Codebook(ANT, N3=N3, param_combination=6, port_selection=True, d=1),
        R17Type2Codebook(ANT, N3=N3, param_combination=5),
    ]


def main() -> None:
    args = cli(__doc__, drops=60)
    chan = default_channel(n_paths=2)  # few rays: the PEB concentrates the channel
    rows = []
    for scheme in schemes():
        r_ant = run_eval(scheme, chan, "antenna", seed=args.seed,
                         snr_db=[10.0], rank=1, n_drops=args.drops)
        r_beam = run_eval(scheme, chan, "beam", seed=args.seed,
                          snr_db=[10.0], rank=1, n_drops=args.drops)
        rows.append(dict(name=scheme.name, antenna=r_ant.sgcs, beam=r_beam.sgcs,
                         bits=r_ant.overhead_bits))
        print(f"{scheme.name:<18} antenna={r_ant.sgcs:.3f} beam={r_beam.sgcs:.3f} "
              f"bits={r_ant.overhead_bits:.0f}")

    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(rows))
    width = 0.38
    b1 = ax.bar(xs - width / 2, [r["antenna"] for r in rows], width,
                label="antenna-domain channel", color="#0072BD")
    b2 = ax.bar(xs + width / 2, [r["beam"] for r in rows], width,
                label="beam-domain channel (after DFT PEB)", color="#D95319")
    for bar, r in zip(b1, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{r['antenna']:.2f}", ha="center", fontsize=8)
    for bar, r in zip(b2, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{r['beam']:.2f}", ha="center", fontsize=8)
    labels = [f"{r['name']}\n({r['bits']:.0f} b)" for r in rows]
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylabel("mean SGCS @ rank 1")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"Regular vs port-selection on both channel domains -- 2-ray drops, "
                 f"(4,2) array, N3=8, {args.drops} drops")
    save(fig, args.out, "fig_09_port_selection", {"rows": rows})


if __name__ == "__main__":
    main()
