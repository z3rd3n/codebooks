# fig_03_overhead_breakdown.png — where each generation spends its bits

**What it shows.** Per-PMI-element bit counts from one actual rank-2 report
at N₃ = 18, stacked per family and colored by what the element encodes
(spatial basis / delay basis / Doppler basis / selection structure /
amplitudes / phases). These totals are serialization-honest:
`len(pack(pmi)) == total_overhead_bits(pmi)` is asserted by the test suite.

**Why it looks like this.**

* **Type I (~26 b)** is almost all per-subband co-phase i₂ (18 × 1–2 bits)
  plus the wideband beam index — there are no amplitudes at all.
* **R15 (~900 b)** is dominated by the yellow phase block i₂,₁ (~680 b) and
  subband amplitudes i₂,₂ (~180 b): *every one of the 18 subbands*
  re-quantizes 2L−1 = 7 coefficients per layer. The wideband part (beam
  selection + WB amplitudes) is a small prefix. This is the cost R16 was
  designed to remove.
* **R16 (~430 b)** flips the proportions: the per-subband payload is gone,
  replaced by the delay basis (i₁,₅/i₁,₆ tap indices), the 2L·M_v bitmap
  i₁,₇ (144 b — now the single largest element), and one quantized
  coefficient set (4-bit amp + 4-bit 16-PSK phase per kept coefficient).
  More than half the report is now *structure* rather than payload.
* **R17 (~155 b)** is the leanest rank-2 report: free port selection
  (i₁,₂ over C(P/2, K₁/2)) replaces the beam machinery, M = 2 taps replace
  the M_v = 5 delay basis, and the bitmap shrinks to 2K₁M. The PEB did the
  spatial work, so the UE pays mostly for coefficients.
* **R18 (~580 b covering 4 intervals)** is the R16 stack with the bitmap
  duplicated per Doppler bin (i₁,₇ = 288 b, Q = 2) plus the 2-bit-per-layer
  shift offset i₁,₁₀. Per covered interval (~145 b) it is the cheapest
  Type II report in the figure — the f2 story in anatomical form.

**Notable.** The bitmap-dominance of R16/R18 explains why their overhead is
almost independent of N₃ (fig_04 left) and why β (which caps how many bitmap
bits are *used* by nonzero coefficients) is a weaker overhead knob than p_v.

**Config.** `scripts/fig_03_overhead_breakdown.py`, one drop, rank 2,
N₃ = 18; element grouping documented in the script header.
