# fig_01_se_vs_snr.png — SU spectral efficiency vs SNR, all families (CDL-C)

**What it shows.** Achievable SU rate of each family's reported precoder vs the
per-subband eigen upper bound, on the same 38.901 CDL-C drops ((4,2) array,
P = 16, N₃ = 8, 2 rx, 3 km/h), at rank 1 (left) and rank 2 (right). R17 is fed
the unitary DFT-PEB view of the same drops; R18 reports once per N₄ = 4 intervals.

**Why it looks like this.**

* **Rank 1: every curve hugs the bound (within ~0.6 b/s/Hz).** At 10 dB:
  Type I 7.29, R15 7.79, R16 7.80, R17 7.72, R18 7.81, bound 7.87. As with the
  synthetic channel, `log₂(1+·)` plus 2-rx combining compress large precoder
  differences into a small SE gap. The fidelity spread is far wider than the SE
  spread — SGCS 0.63 (Type I) vs 0.94–0.96 (Type II families) — which is why the
  AI/ML study scores SGCS and why this SU-SE view undersells Type II (its real
  payoff is MU-MIMO, fig_06).
* **Rank 2 separates Type I.** Type I loses ~1.5 b/s/Hz to the bound at 10 dB
  (10.66 vs 12.18) while the Type II families stay within ~0.2: two-layer
  multiplexing needs two *accurate* directions, and one DFT beam plus a rigidly
  co-phased offset can't supply the second one. Type I's rank-2 SGCS collapses
  to 0.31 (vs 0.63 at rank 1) — the same rotation-within-the-span penalty the
  synthetic figure documents (`subspace_sgcs` is stored in the JSON).
* **R18 sits on top of R16:** on this (essentially static, 3 km/h) channel the
  Doppler axis carries no information, so R18's fidelity equals R16's — only its
  overhead is larger (fig_03).

**CDL vs synthetic.** The whole Type II cluster sits *closer* to the bound here
than on the sparse 4-ray channel (rank-1 SGCS ≈ 0.95 vs ≈ 0.91): CDL-C's
well-conditioned spatial covariance is easy for L = 4 beams to span. Type I's
deficit is unchanged — it is a structural single-beam limit, not a
channel-richness effect.

**Config.** `scripts/cdl_fig_01_se_vs_snr.py`, CDL-C, 100 drops, paired/reset CDL
bank; bound from the Type I run (identical drops for all 1-slot schemes).
