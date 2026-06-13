# fig_10_array_scaling.png — scaling with the antenna array, P = 8…32 (CDL-C)

**What it shows.** The (N1,N2) sweep {(2,2),(4,2),(6,2),(8,2)} (P = 8…32) at
rank 1, all five families (matched L = 4; R17 via PEB with K1 = P/2; R18 N₄ = 4),
each on its own CDL-C panel array. A sixth point, (16,1) at P = 32, is overlaid as
hollow markers. Four panels: SE@10 dB, SE gap to bound, SGCS, feedback bits — all
vs P.

**Why it looks like this.**

* **SE grows ~log₂(P) for everyone** (top-left): 6.7 → 8.7 b/s/Hz from P = 8 to
  32 — array beamforming gain, available to all families.
* **The gap to the bound widens with P** (top-right): a single Type I beam gets
  *relatively* narrower as the grid densifies, so it captures a smaller fraction
  of the array gain; the L-beam Type II families stay glued to the bound. This is
  the paper's "the gap becomes more pronounced as antennas increase" claim,
  reproduced on CDL.
* **Every codebook's SGCS *decreases* with P** (bottom-left): beams narrow and the
  channel's energy spills across more bases while L, K0 and the quantizers stay
  fixed. **Type I falls fastest** (0.77 → 0.54); the Type II families decay
  gently (0.97 → 0.90).
* **Bits stay nearly flat** (bottom-right) — only the basis indicator
  i12 = ⌈log₂ C(N₁N₂, L)⌉ grows — **except R17**, whose port budget K1 = P/2
  scales with the array (41 → 162 bits), the deliberate design choice that lets it
  track the bound at large P.

**CDL vs synthetic.** Same four trends. CDL-C's better conditioning keeps the
Type II SGCS curves higher across the sweep, so the Type-I-falls-fastest contrast
is even starker.

**Config.** `scripts/cdl_fig_10_array_scaling.py`, CDL-C, geometries above +
(16,1), rank 1, N₃ = 8, 50 drops (R17 via PEB K1 = P/2; R18 N₄ = 4).
