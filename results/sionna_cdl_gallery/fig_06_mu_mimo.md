# fig_06_mu_mimo.png — MU-MIMO ZF sum rate from reported PMIs (CDL-C)

**What it shows.** Each of K users reports a rank-1 PMI on its own CDL-C drop;
the gNB zero-forces across the reported directions; rates include the residual
inter-user interference (`evaluate_mu`). The full-CSI reference applies the same
ZF to the true eigenvectors, so the gap is pure feedback-quantization loss. Left:
sum rate vs SNR at K = 4; right: sum rate vs user count at 15 dB. R17 via the PEB.

**Why it looks like this.**

* **This is where Type II earns its bits.** At 10 dB, K = 4: R16 14.4, R15 14.0,
  R17 11.9, **Type I 6.9**, full-CSI 16.9. Unlike SU-SE (fig_01, where Type I is
  within ~0.6 b/s/Hz), here Type I's coarse direction makes ZF leak — its
  curve **saturates** around 9 b/s/Hz at high SNR (interference-limited), while
  the Type II families keep climbing because their accurate directions null the
  other users.
* **Sum rate vs users peaks then falls.** Every scheme rises from K = 2 to a
  peak (Type II near K = 4, ~18 b/s/Hz) then declines as the gNB runs out of
  spatial degrees of freedom (P = 16 ports, 2 rx) and inter-user interference
  dominates. **Type I peaks earliest and falls fastest** (12.1 at K = 2 →
  4.8 at K = 8): coarse directions collide sooner.
* **R17 trails the antenna-domain Type II families** here because MU-MIMO is
  evaluated on the physical channel and R17's value (cheap fidelity *after* a
  PEB) doesn't help when each user's full channel is available.

**CDL vs synthetic.** The Type-I-saturates / Type-II-scales separation is
*sharper* on CDL-C than on the sparse channel: realistic multi-user spatial
correlation punishes coarse feedback harder, widening the gap to full CSI.

**Config.** `scripts/cdl_fig_06_mu_mimo.py`, CDL-C, (4,2) array, P = 16, N₃ = 8,
30 drops, plain ZF (pseudo-inverse), equal power split.
