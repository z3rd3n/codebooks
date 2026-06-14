"""Fig 03 -- Overhead anatomy: where each codebook generation spends its bits.

Per-PMI-element bit counts from *actual reported PMIs* (rank 2, N3 = 18 --
the paper's f2-like operating point), grouped by what the element encodes:

* spatial basis (i11 rotation, i12 beam/port combination)
* delay basis (i15/i16 tap selection)            [R16+]
* Doppler basis (i110 shift offsets)             [R18]
* selection structure (i13/i18 strongest coefficient, i17 bitmap)
* amplitudes (i14 wideband, i22 subband, i23 reference, i24 differential)
* phases (i21 subband co-phase, i25 coefficient phases)

The split explains the compression story: R15's per-subband i2 dominates
its budget, while R16+ moves the cost into a fixed basis + bitmap whose
size no longer scales with N3.  These totals are the serialization-honest
numbers (len(pack(pmi)) is asserted equal in the test suite).

Run: python scripts/fig_03_overhead_breakdown.py -> results/fig_03_overhead_breakdown.png
"""

import matplotlib.pyplot as plt
import numpy as np
from figlib import ANT, BeamDomainChannel, cli, default_channel, save

from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
)

N3_HERE = 18
RANK = 2

GROUPS = {  # element -> (group, color)
    "i11": ("spatial basis", "#0072BD"),
    "i12": ("spatial basis", "#4DA6E0"),
    "i15": ("delay basis", "#2CA02C"),
    "i16": ("delay basis", "#7CC47C"),
    "i110": ("Doppler basis", "#17BECF"),
    "i13": ("selection structure", "#9467BD"),
    "i18": ("selection structure", "#9467BD"),
    "i17": ("selection structure", "#C5B0D5"),
    "i14": ("amplitudes", "#D95319"),
    "i22": ("amplitudes", "#F28E2B"),
    "i23": ("amplitudes", "#D95319"),
    "i24": ("amplitudes", "#F28E2B"),
    "i2": ("phases", "#EDB120"),
    "i21": ("phases", "#EDB120"),
    "i25": ("phases", "#EDB120"),
}


def main() -> None:
    args = cli(__doc__, drops=1)
    schemes = [
        (Type1Codebook(ANT, N3=N3_HERE), "antenna"),
        (R15Type2Codebook(ANT, N3=N3_HERE, L=4, subband_amplitude=True), "antenna"),
        (R16Type2Codebook(ANT, N3=N3_HERE, param_combination=6), "antenna"),
        (R17Type2Codebook(ANT, N3=N3_HERE, param_combination=5), "beam"),
        (R18Type2Codebook(ANT, N3=N3_HERE, N4=4, param_combination=7), "antenna"),
    ]
    chan = default_channel(n3=N3_HERE)
    beam_chan = BeamDomainChannel(chan, ANT)
    rng = np.random.default_rng(args.seed)
    H = chan.generate(n_slots=4, rng=rng)
    H_beam = beam_chan.generate(n_slots=4, rng=np.random.default_rng(args.seed))

    breakdown: dict[str, dict[str, int]] = {}
    for scheme, domain in schemes:
        n_slots = getattr(scheme, "N4", 1)
        Hs = (H_beam if domain == "beam" else H)[:n_slots]
        pmi = scheme.select(Hs, rank=RANK)
        breakdown[scheme.name] = dict(scheme.overhead_bits(pmi))

    fig, ax = plt.subplots(figsize=(11, 5))
    names = list(breakdown)
    axis_max = max(sum(d.values()) for d in breakdown.values())
    min_inline = 0.028 * axis_max  # narrower segments get a callout above the bar
    min_gap = 0.036 * axis_max  # horizontal spacing between callout labels
    for y, name in enumerate(names):
        left = 0
        callout_x = -np.inf
        for elem, bits in breakdown[name].items():
            group, color = GROUPS.get(elem, ("other", "0.6"))
            ax.barh(y, bits, left=left, color=color, edgecolor="white", height=0.62)
            center = left + bits / 2
            if bits >= min_inline:
                ax.text(center, y, f"{elem}\n{bits}", ha="center",
                        va="center", fontsize=7)
            elif bits > 0:
                # too narrow on the shared axis to label inline: callout above
                # the bar, greedily pushed right so adjacent narrow segments
                # don't print on top of each other
                callout_x = max(center, callout_x + min_gap)
                ax.plot([center, callout_x], [y - 0.32, y - 0.42],
                        lw=0.6, color="0.45")
                ax.text(callout_x, y - 0.44, f"{elem}\n{bits}", ha="center",
                        va="bottom", fontsize=6, color="0.25")
            left += bits
        ax.text(left + 0.008 * axis_max, y, f"{left} b", va="center", fontsize=9,
                fontweight="bold")
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.45, -0.85)  # inverted y, headroom for the callouts
    ax.set_xlabel("feedback bits per report (per element)")
    handles = []
    for group, color in dict(
        (g, c) for g, c in
        [("spatial basis", "#0072BD"), ("delay basis", "#2CA02C"),
         ("Doppler basis", "#17BECF"), ("selection structure", "#9467BD"),
         ("amplitudes", "#D95319"), ("phases", "#EDB120")]
    ).items():
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=group))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_title(
        f"PMI overhead breakdown, rank {RANK}, N3={N3_HERE} -- one report each "
        "(R17 via PEB; R18 covers 4 intervals)"
    )
    save(fig, args.out, "fig_03_overhead_breakdown", breakdown)
    for name, d in breakdown.items():
        print(f"{name:<22} total={sum(d.values()):5d}  {d}")


if __name__ == "__main__":
    main()
