"""Fig 06 -- MU-MIMO: ZF sum rate from reported PMIs (paper eq. 2).

Each of K users reports a rank-1 PMI on its own channel; the gNB
zero-forces across the reported directions; rates include the residual
inter-user interference (``evaluate_mu``).  The full-CSI reference applies
the same ZF to the true eigenvectors, so the gap is pure feedback
quantization loss -- the regime Type II exists for (Type I's coarse
direction makes ZF leak).

Plain ZF (``regularization=None``) throughout: since S6 the baseline uses
the pseudo-inverse, so colinear reported directions (the old crash at many
Type I users) degrade gracefully and the epsilon-RZF workaround is gone.

* left: sum rate vs SNR at K = 4 users;
* right: sum rate vs number of users at 15 dB.

R17 is evaluated in the beam domain (unitary PEB -- same physical drops).
R18 is omitted: per-report it is R16 plus Doppler bits, and `evaluate_mu`
is single-interval.

Run: python scripts/figures/fig_06_mu_mimo.py -> results/fig_06_mu_mimo.png
"""

import matplotlib.pyplot as plt
import numpy as np

from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    Type1Codebook,
)
from nr_csi.eval import evaluate_mu
from nr_csi.figtools.figlib import (
    ANT,
    N3,
    BeamDomainChannel,
    ant_tag,
    cli,
    default_channel,
    save,
    select_families,
    style,
)

SNR_DB = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
SNR_REF = 15.0
USERS = [2, 3, 4, 6, 8]


def schemes():
    return select_families([
        (Type1Codebook(ANT, N3=N3), "antenna"),
        (R15Type2Codebook(ANT, N3=N3, L=4), "antenna"),
        (R16Type2Codebook(ANT, N3=N3, param_combination=6), "antenna"),
        (R17Type2Codebook(ANT, N3=N3, param_combination=5), "beam"),
    ])


def run(scheme, domain, n_users, snr_db, args):
    chan = default_channel()
    if domain == "beam":
        chan = BeamDomainChannel(chan, ANT)
    # plain ZF: baselines.zf is pinv-based (S6), so two users reporting the
    # *same* coarse direction (Type I especially) degrade gracefully (they
    # share the direction and fully interfere) instead of crashing
    return evaluate_mu(scheme, chan, n_users=n_users, snr_db=snr_db,
                       n_drops=args.drops, rng=np.random.default_rng(args.seed),
                       regularization=None)


def main() -> None:
    args = cli(__doc__, drops=30)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    data: dict = {"vs_snr": {"snr_db": SNR_DB}, "vs_users": {"users": USERS}}

    full_csi = None
    for scheme, domain in schemes():
        res = run(scheme, domain, 4, SNR_DB, args)
        label = scheme.name + (" (via PEB)" if domain == "beam" else "")
        axes[0].plot(SNR_DB, res.sum_rate, label=label, **style(scheme.name))
        data["vs_snr"][scheme.name] = res.sum_rate
        if domain == "antenna":
            full_csi = res.sum_rate_full_csi  # identical drops for all antenna schemes
    axes[0].plot(SNR_DB, full_csi, label="full CSI ZF", **style("full CSI ZF"))
    data["vs_snr"]["full CSI ZF"] = full_csi
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("ZF sum rate (bits/s/Hz)")
    axes[0].set_title("K = 4 users")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    full_k = []
    for scheme, domain in schemes():
        ys = []
        for k in USERS:
            res = run(scheme, domain, k, [SNR_REF], args)
            ys.append(res.sum_rate[0])
            if domain == "antenna" and scheme.name == "R16 eType II":
                full_k.append(res.sum_rate_full_csi[0])
        axes[1].plot(USERS, ys, label=scheme.name + (" (via PEB)" if domain == "beam" else ""),
                     **style(scheme.name))
        data["vs_users"][scheme.name] = ys
    axes[1].plot(USERS, full_k, label="full CSI ZF", **style("full CSI ZF"))
    data["vs_users"]["full CSI ZF"] = full_k
    axes[1].set_xlabel("number of users K")
    axes[1].set_ylabel(f"ZF sum rate @ {SNR_REF:.0f} dB")
    axes[1].set_title("sum rate vs user count")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"MU-MIMO from reported PMIs -- {ant_tag(ANT)}, N3={N3}, "
                 f"{args.drops} drops, equal power split")
    save(fig, args.out, "fig_06_mu_mimo", data)


if __name__ == "__main__":
    main()
