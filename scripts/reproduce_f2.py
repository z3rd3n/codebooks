"""Reproduce the structure of the paper's Fig. f2: feedback overhead vs L.

Configuration follows the paper: N1*N2 = 16 logical ports per polarization
((N1,N2) = (16,1), O1*O2 = 4), N3 = 18, N4 = 4, Q = 2, rank v = 2, Mv = 5,
N_PSK = 4 (R15), K^(2) = 6, K^NZ = 20.

Accounting convention (documented in the README): R15/R16 re-report once per
slot interval (no temporal compression), R18 covers all N4 intervals with a
single predicted-PMI report.  The absolute bar heights of the paper's figure
are not derivable from its own bit tables (see README errata); the orderings
and growth with L are.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nr_csi.config import AntennaConfig
from nr_csi.metrics.overhead import f2_comparison

OUT = pathlib.Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    antenna = AntennaConfig.standard(16, 1)  # N1N2=16, O1O2=4
    Ls = (1, 2, 3, 4)
    data = f2_comparison(antenna, Ls=Ls)

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    xs = np.arange(len(Ls))
    colors = {"R15 Regular": "#0072BD", "R16 Regular": "#D95319", "R18 Regular": "#EDB120"}
    for k, (name, vals) in enumerate(data.items()):
        ax.bar(xs + (k - 1) * width, vals, width, label=name, color=colors[name])
        for x, val in zip(xs + (k - 1) * width, vals):
            ax.text(x, val * 1.05, str(val), ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(xs, [f"$L = {L}$" for L in Ls])
    ax.set_ylabel("Feedback Overhead (bits, covering $N_4$ = 4 intervals)")
    ax.legend()
    ax.set_title("Feedback overhead vs number of spatial bases (cf. paper Fig. f2)")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "f2.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT / 'f2.png'}")
    for name, vals in data.items():
        print(f"{name:14s}", dict(zip(Ls, vals)))
    r15, r16, r18 = (data[k] for k in ("R15 Regular", "R16 Regular", "R18 Regular"))
    print("R15/R16 ratio:", [round(a / b, 2) for a, b in zip(r15, r16)])
    print("R16/R18 ratio:", [round(a / b, 2) for a, b in zip(r16, r18)])


if __name__ == "__main__":
    main()
