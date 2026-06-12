# fig_10_array_scaling.png — what happens as the number of ports grows

**What it shows.** All five families on the supported geometry sweep
(2,2)→(8,2), P = 8 → 32 CSI-RS ports, rank 1, same sparse 4-ray drops per
P. R17 runs through the unitary DFT PEB with α = 1/2, so its port budget
K₁ = P/2 scales with the array. Hollow markers: a second geometry (16,1)
at the same P = 32, separating "more ports" from "different aspect ratio".

**Why it looks like this — the four port-scaling regimes.**

* **SE (top-left): everyone gains with P, sublinearly.** The eigen bound
  climbs 6.54 → 8.51 b/s/Hz (array gain ≈ log₂P at fixed SNR); every
  codebook rides it because even a mediocre precoder collects most of the
  larger aperture. Ordering at P = 32: R18 8.39 ≈ R16 8.38 ≈ R15 8.36 >
  R17 8.21 > Type I 7.92.
* **Gap to the bound (top-right): the paper's f1 claim, isolated.**
  Type I's quantization loss grows 0.48 → 0.59 b/s/Hz: the DFT grid
  densifies with N₁N₂ but one beam captures a shrinking fraction of a
  multipath channel — relative beam width ∝ 1/N₁N₂ while the channel's
  angular spread is fixed. The L = 4 combinations hold a near-flat ~0.1
  (R16) to ~0.15 (R15) gap: more, narrower beams per group exactly offset
  the narrowing. **R17 is the only curve that *improves* with P**
  (0.52 → 0.31): α = 1/2 ties its budget to the array — K₁ = P/2 selected
  ports and K₀ = ⌈βK₁M⌉ coefficients both double when P doubles, so it is
  the one family whose configured capacity scales with the hardware.
* **SGCS (bottom-left): every fixed-budget codebook *loses* fidelity with
  P.** Type I 0.70 → 0.64, R15 0.914 → 0.899, R16 0.935 → 0.911,
  R18 0.966 → 0.917 — the channel's dimensionality (P) grows while L = 4,
  M_v, K₀ and the quantizers stay fixed, so off-grid leakage spreads over
  more bases than the budget covers. R17 again runs the other way
  (0.69 → 0.81). The (2,2) end is special: L = N₁N₂ = 4 means the four
  beams form a *complete* basis per polarization — fidelity there is purely
  quantization-limited (hence R18's 0.966 with its larger budget).
* **Bits (bottom-right): the price of scaling.** R15/R16/R18 are nearly
  flat in P — only i₁,₂ = ⌈log₂C(N₁N₂, L)⌉ grows (0 bits at (2,2) where
  choosing 4 of 4 beams is no choice; ~11 bits at (16,1)); coefficient
  counts depend on L/M_v/K₀, not P. Type I is flat at ~24 b. R17's cost
  grows steeply (~70 → ~210 b) — the flip side of its scaling budget
  (port-combination index over C(P/2, P/4) plus a 2K₁M bitmap).

**Notable / unexpected — both investigated.**

1. **R18 > R16 on every static point** (e.g. 0.966 vs 0.935 at P = 8)
   although N₄ = 1 is byte-identical to R16. Cause: the R18 spec budget is
   K₀ = ⌈2βLM₁**Q**⌉ (paper, R18 section) — double R16's at Q = 2. On a
   static channel the second Doppler bin quantizes to zero, so the whole
   doubled budget serves the static component: up to 2× the delay-tap
   coefficients (verified per drop: identical beams, more kept taps on
   drops where R16's K₀ binds). The price is the doubled i₁,₇ bitmap +
   i₁,₁₀ (239 vs 156 b in fig_12's table) — R18 is "R16 with a bigger
   coefficient budget" when the channel doesn't move.
2. **At fixed P = 32, (16,1) beats (8,2) for every Type II family**
   (R16 0.928 vs 0.911). Off-grid rays in a 2-D grid leak across *both*
   beam dimensions (product of two Dirichlet kernels), needing more than
   L = 4 group-aligned bases; a 1-D grid leaks along one dimension only.
   The effect is small (~0.02 SGCS) but consistent — aspect ratio matters
   slightly at fixed budget, port count matters more.

**Takeaway for ML CSI feedback.** With fixed feedback budgets the 3GPP
fidelity *decreases* in P while the bound rises — the headroom for learned
schemes grows with the array; R17's α-scaling is the one standardized
mechanism that escapes this, at α-scaled cost.

**Config.** `scripts/fig_10_array_scaling.py`, 50 drops, N₃ = 8, paired
seeds per geometry; matched L = 4 (R16 pc6, R18 pc7), R17 pc5.
