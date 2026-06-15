"""Reproduce the paper's Fig. f1: spectral efficiency vs SNR.

Cases: (N1,N2) = (4,1) and (16,1) with (O1,O2) = (4,1), single polarization;
1 stream for both sizes, 2 streams for N=16.  Type II uses L=4 beams, 3-bit
wideband amplitude, subband amplitude off, 3-bit (8-PSK) phase.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nr_csi.eval.f1 import F1_REPRODUCTION, run_f1_case

OUT = pathlib.Path(__file__).resolve().parents[2] / "results"


def main() -> None:
    snr = np.arange(-10, 31, 2.5)
    cases = [
        (4, 1, "N=4, 1 stream"),
        (16, 1, "N=16, 1 stream"),
        (16, 2, "N=16, 2 streams"),
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    styles = {"upper_bound": ("-", "Ideal eigen BF"), "type2": ("--", "Type II (L=4)"),
              "type1": (":", "Type I")}
    colors = ["#0072BD", "#D95319", "#77AC30"]
    for (N, ns, label), color in zip(cases, colors):
        cv = run_f1_case(
            N, ns, snr, n_drops=400, seed=42,
            n_paths=F1_REPRODUCTION["n_paths"],
            n_psk=F1_REPRODUCTION["n_psk"],
            t1_second_beam=F1_REPRODUCTION["t1_second_beam"],
        )
        for attr, (ls, name) in styles.items():
            ax.plot(snr, getattr(cv, attr), ls, color=color,
                    label=f"{label}: {name}")
        print(f"{label}: SE @ 0 dB  UB={np.interp(0, snr, cv.upper_bound):.2f} "
              f"T2={np.interp(0, snr, cv.type2):.2f} T1={np.interp(0, snr, cv.type1):.2f}")
        print(f"{label}: SE @ 20 dB UB={np.interp(20, snr, cv.upper_bound):.2f} "
              f"T2={np.interp(20, snr, cv.type2):.2f} T1={np.interp(20, snr, cv.type1):.2f}")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Spectral efficiency (bits/s/Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("Type I vs Type II vs ideal beamforming (cf. paper Fig. f1)")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "f1.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT / 'f1.png'}")


if __name__ == "__main__":
    main()
