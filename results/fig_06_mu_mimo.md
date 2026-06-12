# fig_06_mu_mimo.png — MU-MIMO ZF sum rate from reported PMIs

**What it shows.** K users each report a rank-1 PMI on their own drop; the
gNB zero-forces across the reported directions (plain ZF, pseudo-inverse);
per-user rates include residual inter-user interference (paper eq. 2).
The grey dashed reference applies the same precoding to the *true*
eigenvectors, so every gap to it is pure feedback-quantization loss.
Left: sum rate vs SNR at K = 4. Right: sum rate vs K at 15 dB.

**Why it looks like this.**

* **MU-MIMO is where precoder fidelity actually pays.** At K = 4 / 15 dB:
  full CSI 28.4, R16 23.6, R15 22.6, R17 19.2, Type I 19.1 b/s/Hz — the
  Type II-vs-Type I gap (~4.5 b/s/Hz, ~23%) is an order of magnitude larger
  than the SU gap of fig_01. ZF nulls are only as good as the directions
  they are computed from: direction error both leaks interference into
  other users and costs beamforming gain, and interference does not vanish
  with SNR — hence the visible slope loss of Type I/R17 above 15 dB.
* **The K sweep separates the families further.** R16 keeps growing to
  32.7 at K = 8 (full CSI: 44.5) while Type I saturates at ~21 from K = 6.
  With a coarse grid, eight users' best beams start to collide; colinear
  reported directions make the ZF Gram matrix singular, and the
  pseudo-inverse then lets those users share the direction and interfere
  fully — the physically right outcome. Type I cannot support dense
  MU-MIMO, which is precisely the paper's motivation for Type II
  ("higher data rates and system capacity, eMBB").
* R17 tracks Type I rather than the Type IIs here: on the default 4-ray
  channel its K₁ = P/2 port window captures the dominant direction about
  as coarsely as a single beam (cf. fig_02); on sparser channels it moves
  toward the Type II curves (fig_09).

**Notable.** Even full-CSI ZF visibly bends away from linear growth at
K = 8 (44.5 < 8/4 × 28.4): with K = 8 of P = 16 spatial degrees of freedom
per polarization pair committed to nulls, the per-user beamforming gain
shrinks — the saturation is partly fundamental, not only a feedback effect.

**Fix landed (S6).** `baselines.zf`/`ezf(xi=0)` now use the pseudo-inverse
(`np.linalg.pinv`), identical for full row rank and finite for colinear
reports, so this figure runs plain ZF (`regularization=None`) and the
ε-RZF workaround (`EPS_REG`) is gone from the script. Regenerated numbers
match the ε-RZF version within Monte-Carlo tolerance (max |Δ| 0.08 b/s/Hz);
the crash repro is locked in
`tests/test_baselines_appendix.py::TestZFRobustness`.

**Config.** `scripts/fig_06_mu_mimo.py`, 30 drops, equal power split,
R18 omitted (its per-report content is R16's; `evaluate_mu` is
single-interval).
