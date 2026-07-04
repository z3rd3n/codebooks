"""Fig 13 -- The spec-completion codebooks: 2-port Type I, R19 refined
Type I (modeA/modeB/multi-panel), and the R18 CJT pair.

Four panels:

(a) 2-port Type I (Table 5.2.2.2.1-1): SU SE vs SNR at ranks 1-2 against the
    eigen bound -- the whole codebook is 6 precoders, so the quantization gap
    is the story.
(b) R19 refined Type I single-panel, 48 ports: SE at 10 dB vs rank 1-8 for
    codebookMode modeA vs modeB (modeB spends more bits on per-layer beams).
(c) R18 eType II CJT, N_TRP = 2: mean SGCS vs the inter-TRP delay offset for
    codebookMode mode1 (reported psi ramp) vs mode2 (no ramp) -- mode1 is
    flat in the delay, mode2 collapses once the offset leaves the common
    delay-tap window.
(d) R19 refined Type I multi-panel (2,4,3) vs single-panel (8,3) at 48
    ports: mean SGCS vs rank on panel-coherent channels.

Run: python scripts/figures/fig_13_new_codebooks.py -> results/fig_13_new_codebooks.png
"""

import matplotlib.pyplot as plt
import numpy as np

from nr_csi.baselines.ideal import eigen_precoder
from nr_csi.channel import RandomRayChannel
from nr_csi.codebooks import (
    R18CJTCodebook,
    RefinedType1MultiPanelCodebook,
    RefinedType1SinglePanelCodebook,
    TwoPortType1Codebook,
)
from nr_csi.config import AntennaConfig
from nr_csi.figtools.figlib import cli, run_eval, save
from nr_csi.metrics.similarity import sgcs

SNR_DB = list(np.arange(-5.0, 31.0, 5.0))
COLOR = {
    "modeA": "#1f77b4",
    "modeB": "#d62728",
    "multi-panel": "#9467bd",
    "mode1": "#2ca02c",
    "mode2": "#e6a817",
    "bound": "0.35",
}


def mean_sgcs(cbk, H, rank):
    """Mean per-subband SGCS of the reported precoder vs the eigen target."""
    targets = eigen_precoder(H[-1], rank=rank)
    W = cbk.precoder(cbk.select(H, rank=rank))
    N3 = H.shape[1]
    return float(np.mean([sgcs(targets[t], W[0, t]) for t in range(N3)]))


# ------------------------------------------------------- (a) 2-port Type I
def panel_two_port(ax, drops, seed, data):
    ant2 = AntennaConfig(N1=1, N2=1, O1=1, O2=1, strict=False)  # P = 2
    chan = RandomRayChannel(ant2, N3=4, n_rx=2, n_paths=4)
    blob = {}
    for rank, ls in ((1, "-"), (2, "--")):
        cbk = TwoPortType1Codebook(N3=4)
        res = run_eval(cbk, chan, seed=seed, antenna=ant2, snr_db=SNR_DB,
                       rank=rank, n_drops=drops)
        ax.plot(SNR_DB, res.se, ls, color=COLOR["modeA"], marker="o", ms=3.5,
                lw=1.7, label=f"2-port Type I, rank {rank}")
        ax.plot(SNR_DB, res.se_upper_bound, ls, color=COLOR["bound"], lw=1.1)
        blob[f"rank{rank}"] = {"se": res.se, "bound": res.se_upper_bound,
                               "bits": res.overhead_bits}
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SE (b/s/Hz)")
    ax.set_title("(a) 2-port Type I vs eigen bound (grey)\n"
                 "Table 5.2.2.2.1-1: 4+2 fixed precoders", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    data["two_port"] = {"snr_db": SNR_DB, **blob}


# --------------------------------- (b) refined Type I: modeA vs modeB by rank
def panel_refined_modes(ax, drops, seed, data):
    ant = AntennaConfig.standard(8, 3)  # 48 ports
    chan = RandomRayChannel(ant, N3=4, n_rx=8, n_paths=6)
    ranks = list(range(1, 9))
    blob = {}
    for mode in ("modeA", "modeB"):
        cbk = RefinedType1SinglePanelCodebook(ant, N3=4, mode=mode)
        se, bits = [], []
        for rank in ranks:
            res = run_eval(cbk, chan, seed=seed, antenna=ant, snr_db=[10.0],
                           rank=rank, n_drops=max(drops // 4, 4))
            se.append(res.se[0])
            bits.append(res.overhead_bits)
        ax.plot(ranks, se, "o-", color=COLOR[mode], lw=1.7, ms=4, label=mode)
        blob[mode] = {"se_10db": se, "bits": bits}
    ax.set_xlabel("rank")
    ax.set_ylabel("SE at 10 dB (b/s/Hz)")
    ax.set_title("(b) R19 refined Type I, 48 ports (8,3):\n"
                 "codebookMode modeA vs modeB, ranks 1-8", loc="left", fontsize=9)
    ax.legend(fontsize=8)
    data["refined_modes"] = {"ranks": ranks, **blob}


# ----------------------------------------- (c) CJT: mode1 vs mode2 vs delay
def panel_cjt_delay(ax, drops, seed, data):
    ant = AntennaConfig.standard(4, 2)  # P = 16 per TRP
    N3 = 8
    delays = list(range(N3))
    chan = RandomRayChannel(ant, N3=N3, n_rx=2, n_paths=3, max_delay=1.0)
    blob = {}
    for mode in ("mode1", "mode2"):
        cbk = R18CJTCodebook(ant, N3=N3, n_trp=2, param_combination_L=4,
                             param_combination=2, mode=mode)
        means = []
        for delta in delays:
            ramp = np.exp(2j * np.pi * np.arange(N3) * delta / N3)
            vals = []
            for d in range(max(drops // 4, 4)):
                rng = np.random.default_rng(seed * 1000 + d)
                base = chan.generate(n_slots=1, rng=rng)
                H = np.concatenate(
                    [base, base * ramp[None, :, None, None]], axis=-1
                )
                vals.append(mean_sgcs(cbk, H, rank=1))
            means.append(float(np.mean(vals)))
        ax.plot(delays, means, "o-", color=COLOR[mode], lw=1.7, ms=4,
                label=f"CJT {mode}" + (" (psi ramp)" if mode == "mode1" else ""))
        blob[mode] = means
    ax.set_xlabel("inter-TRP delay offset (taps)")
    ax.set_ylabel("mean SGCS (rank 1)")
    ax.set_title("(c) R18 CJT, N_TRP=2: inter-TRP co-phasing\n"
                 "mode1 pre-compensates the offset, mode2 cannot",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8)
    data["cjt_delay"] = {"delays": delays, **blob}


# ------------------------------- (d) refined multi-panel vs single-panel
def panel_multipanel(ax, drops, seed, data):
    N3 = 4
    ant_sp = AntennaConfig.standard(8, 3)  # 48 ports, single panel
    ant_mp = AntennaConfig.standard(4, 3, Ng=2)  # 48 ports, 2 panels
    panel_geo = AntennaConfig.standard(4, 3)  # one panel = 24 ports
    chan_sp = RandomRayChannel(ant_sp, N3=N3, n_rx=4, n_paths=6)
    chan_panel = RandomRayChannel(panel_geo, N3=N3, n_rx=4, n_paths=6)
    ranks = [1, 2, 3, 4]
    n = max(drops // 4, 4)
    blob = {}

    def mp_channel(rng):
        """Panel-coherent drop: both panels see the same rays up to a random
        wideband inter-panel phase (co-located panels)."""
        base = chan_panel.generate(n_slots=1, rng=rng)
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi))
        return np.concatenate([base, base * phase], axis=-1)

    curves = {
        "modeA": (RefinedType1SinglePanelCodebook(ant_sp, N3=N3, mode="modeA"),
                  lambda rng: chan_sp.generate(1, rng)),
        "modeB": (RefinedType1SinglePanelCodebook(ant_sp, N3=N3, mode="modeB"),
                  lambda rng: chan_sp.generate(1, rng)),
        "multi-panel": (RefinedType1MultiPanelCodebook(ant_mp, N3=N3),
                        mp_channel),
    }
    for name, (cbk, gen) in curves.items():
        means = []
        for rank in ranks:
            vals = [
                mean_sgcs(cbk, gen(np.random.default_rng(seed * 2000 + d)), rank)
                for d in range(n)
            ]
            means.append(float(np.mean(vals)))
        label = {"modeA": "SP (8,3) modeA", "modeB": "SP (8,3) modeB",
                 "multi-panel": "MP (2,4,3)"}[name]
        ax.plot(ranks, means, "o-", color=COLOR[name], lw=1.7, ms=4, label=label)
        blob[name] = means
    ax.set_xticks(ranks)
    ax.set_xlabel("rank")
    ax.set_ylabel("mean SGCS")
    ax.set_title("(d) R19 refined Type I at 48 ports:\n"
                 "multi-panel (2,4,3) vs single-panel (8,3)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8)
    data["multipanel"] = {"ranks": ranks, **blob}


def main() -> None:
    args = cli(__doc__, drops=64)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.subplots_adjust(hspace=0.45, wspace=0.28)
    data: dict = {}
    panel_two_port(axes[0, 0], args.drops, args.seed, data)
    panel_refined_modes(axes[0, 1], args.drops, args.seed, data)
    panel_cjt_delay(axes[1, 0], args.drops, args.seed, data)
    panel_multipanel(axes[1, 1], args.drops, args.seed, data)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Spec-completion codebooks: 2-port Type I, R19 refined Type I "
        "(modeA / modeB / multi-panel), R18 CJT",
        y=0.98, fontsize=12,
    )
    save(fig, args.out, "fig_13_new_codebooks", data)


if __name__ == "__main__":
    main()
