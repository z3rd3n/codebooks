# fig_03_overhead_breakdown.png — where each codebook spends its bits (CDL-C)

**What it shows.** Per-PMI-element bit counts from *actual reported PMIs* at
rank 2, N₃ = 18, grouped by what each element encodes (spatial basis, delay
basis, Doppler basis, selection structure, amplitudes, phases). One report per
scheme; R17 via the PEB, R18 covering 4 intervals.

**Why it looks like this.**

* **R15's budget is dominated by per-subband phases.** Total 923 bits, of which
  i21 (subband co-phase) = 684 and i22 (subband amplitude) = 180: R15 quantizes
  a coefficient for *every* subband, so its cost scales with N₃.
* **R16 moves the cost into a fixed basis + a bitmap.** Total 549: a small
  delay-tap indicator (i16 = 30), a non-zero-coefficient bitmap (i17 = 144), and
  amplitudes/phases only for the *selected* coefficients (i24 = 150, i25 = 200).
  None of these grow with N₃ the way R15's per-subband terms do — the
  compression story of R16, made literal.
* **R17 is the cheapest non-trivial report** (155 bits): port selection replaces
  the spatial-basis combinatorics, and with M = 2 taps the coefficient block is
  small.
* **R18 = R16 + a Doppler basis** (671 bits): the extra i110 Doppler indicator is
  tiny (4 bits), but covering N₄ intervals enlarges the coefficient/bitmap block
  (i17 = 288) — the price of the prediction fig_05 shows.
* **Type I is a rounding error** (27 bits): one beam, one wideband amplitude,
  per-subband co-phase only.

**CDL vs synthetic.** These totals are **structural** — they depend on the
codebook configuration and rank, not on the channel values (the bits encode
*which* and *how many* coefficients are reported, asserted byte-exact against
`len(pack(pmi))` in the test suite). The CDL figure therefore matches the
synthetic one; it is included for a consistent gallery.

**Config.** `scripts/cdl_fig_03_overhead_breakdown.py`, CDL-C, 1 report, rank 2,
N₃ = 18.
