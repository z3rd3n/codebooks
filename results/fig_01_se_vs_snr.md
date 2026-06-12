# fig_01_se_vs_snr.png — SU spectral efficiency vs SNR, all families

**What it shows.** Achievable SU rate of each family's reported precoder vs
the per-subband eigen upper bound on the same static 4-ray drops ((4,2)
array, P = 16, N₃ = 8, 2 rx), at rank 1 (left) and rank 2 (right). R17 is
fed the unitary DFT-PEB view of the same drops (its deployment model);
R18 reports once per N₄ = 4 intervals.

**Why it looks like this.**

* **Rank 1: all curves bunch within ~0.6 b/s/Hz of the bound.** At 10 dB:
  Type I 7.09, R15 7.54, R16 7.56, R18 7.58, bound 7.67. Two effects
  compress the SE axis: log₂(1+·) flattens received-power differences, and
  the 2-rx receiver recovers part of any transmit-side mismatch. The
  *precoder fidelity* differences are much larger than the SE differences —
  SGCS 0.65 (Type I) vs 0.91–0.94 (Type II families) — which is exactly why
  3GPP's AI/ML study uses SGCS, and why the SU-SE view undersells Type II
  (its real payoff is MU-MIMO, fig_06).
* **Rank 2 separates the families.** Type I loses ~2.1 b/s/Hz to the bound
  (8.67 vs 10.80 at 10 dB) while the Type II families stay within ~0.35:
  multiplexing needs two *accurate* spatial directions, and one grid beam
  plus a rigidly co-phased offset beam can't deliver the second one.
* R18's curve sits on top of R16's: on a static channel the Doppler axis
  carries no information, but its report is still an R16 report (bin 0)
  with a Q-times larger coefficient budget (see fig_10 notes), so fidelity
  is never worse — only the overhead is (fig_03).

**Notable / unexpected — investigated.** Type I's rank-2 SGCS is 0.26
(vs 0.65 at rank 1), yet its rank-2 SE is still decent. Probing per-layer:
both layers individually score ~0.3, but the *subspace* spanned by the two
columns captures 0.52 of the reference subspace (vs 0.29 column-wise).
The spec's rank-2 structure w₁ = [v₁; φv₁], w₂ = [v₂; −φv₂] (same
amplitude on both polarizations, shared co-phase, second beam from three
allowed i₁,₃ offsets) cannot rotate *within its own span* to align with
the eigenvector basis. The log-det rate only depends on the subspace and
power split, so SE survives; column-wise SGCS — the 3GPP metric — punishes
the misrotation. Takeaway: for rank > 1, compare schemes on SE *and* SGCS;
SGCS alone overstates Type I's rank-2 deficit.

**Fix landed (S1).** The harness now reports a rotation-invariant
`subspace_sgcs` alongside SGCS (`nr_csi.metrics.subspace_sgcs`, stored in
this figure's JSON next to `sgcs`): at rank 2 it reads 0.52 for Type I
(vs 0.26 column-wise) — the rotation-within-the-span penalty made
explicit. Locked in
`tests/test_eval_spine.py::TestSubspaceSGCS::test_type1_rank2_subspace_gap`.

**Config.** `scripts/fig_01_se_vs_snr.py`, 100 drops, paired seeds; upper
bound taken from the Type I run (identical drops for all 1-slot schemes).
