"""Fig 09 -- Applicability boundary: regular vs port-selection codebooks.

Every variant evaluated on THREE views of the same 2-ray physical drops:

* antenna domain: the raw channel (regular codebooks' home turf);
* beam domain: the same drops through a unitary per-polarization DFT PEB
  (the port-selection deployment model -- the gNB beamforms the CSI-RS, so
  the channel is concentrated on few ports);
* tuned PEB: the beam domain with ports re-sorted by per-drop energy (S5).
  The R15/R16 PS basis is a *consecutive port window*, so DFT-PEB energy
  landing on non-adjacent ports defeats it; a gNB that orders its PEB
  beams by user energy restores the windowed codebooks.  R17's free port
  selection is order-invariant -- the control.

Regular codebooks search the DFT beam structure themselves and win on the
antenna domain; PS codebooks assume the PEB already did that and win after
it (paper "Applicable Scenarios").  Bars annotated with feedback bits.

Run: python scripts/fig_09_port_selection.py -> results/fig_09_port_selection.png
"""

import matplotlib.pyplot as plt
import numpy as np
from figlib import ANT, N3, BeamDomainChannel, cli, default_channel, run_eval, save

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
    tuned = BeamDomainChannel(chan, ANT, sort_by_energy=True)
    rows = []
    for scheme in schemes():
        r_ant = run_eval(scheme, chan, "antenna", seed=args.seed,
                         snr_db=[10.0], rank=1, n_drops=args.drops)
        r_beam = run_eval(scheme, chan, "beam", seed=args.seed,
                          snr_db=[10.0], rank=1, n_drops=args.drops)
        r_tuned = run_eval(scheme, tuned, "antenna", seed=args.seed,
                           snr_db=[10.0], rank=1, n_drops=args.drops)
        rows.append(dict(name=scheme.name, antenna=r_ant.sgcs, beam=r_beam.sgcs,
                         tuned=r_tuned.sgcs, bits=r_ant.overhead_bits))
        print(f"{scheme.name:<18} antenna={r_ant.sgcs:.3f} beam={r_beam.sgcs:.3f} "
              f"tuned={r_tuned.sgcs:.3f} bits={r_ant.overhead_bits:.0f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(rows))
    width = 0.27
    groups = [
        ("antenna", -width, "antenna-domain channel", "#0072BD"),
        ("beam", 0.0, "beam-domain channel (after DFT PEB)", "#D95319"),
        ("tuned", width, "tuned PEB (per-drop sorted -- upper bound of PEB tuning)",
         "#77AC30"),
    ]
    for key, off, label, color in groups:
        bars = ax.bar(xs + off, [r[key] for r in rows], width, label=label, color=color)
        for bar, r in zip(bars, rows):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{r[key]:.2f}", ha="center", fontsize=7)
    labels = [f"{r['name']}\n({r['bits']:.0f} b)" for r in rows]
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylabel("mean SGCS @ rank 1")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"Regular vs port-selection across channel domains -- 2-ray drops, "
                 f"(4,2) array, N3=8, {args.drops} drops")
    save(fig, args.out, "fig_09_port_selection", {"rows": rows})


if __name__ == "__main__":
    main()
