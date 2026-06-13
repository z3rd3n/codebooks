# fig_04_overhead_scaling.png — overhead scaling laws (channel-independent)

**What it shows.** Feedback bits per report from the spec bit-count formulas
(Tables bit1/bit2), (16,1) array, rank 2, in three panels: bits vs N₃, bits vs L,
and bits to cover N₄ slot intervals — for R15 Type II, R16 eType II, and R18.

**Why it looks like this.**

* **bits vs N₃:** R15 grows **linearly** (its per-subband i2 is paid for every
  one of the N₃ units); R16/R18 grow only through M_v = ⌈p_v·N₃/R⌉ DFT taps and
  their log-sized indicators — a much shallower slope.
* **bits vs L:** all families grow through the combinatorial basis indicator
  i12 = ⌈log₂ C(N₁N₂, L)⌉ plus linear coefficient terms; the Type II curves
  rise together.
* **bits vs N₄ intervals covered:** R15/R16 must **re-report every interval**
  (linear in N₄), while R18 covers all N₄ with one predicted-PMI report — the
  flatten-with-N₄ curve that pays for the prediction in fig_05.

**This figure uses no channel.** It is computed entirely from the standardized
overhead formulas (`nr_csi.metrics.overhead`), so the CDL version is **bit-for-bit
identical** to the synthetic `results/fig_04_overhead_scaling.png`. It is
regenerated here only so the CDL gallery is complete.

**Config.** `scripts/cdl_fig_04_overhead_scaling.py`, (16,1) array, rank 2,
p_v = 1/4, β = 1/2, N_PSK = 4 (R15), Q = 2.
