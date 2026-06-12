# fig_05_mobility.png — CSI aging vs the R18 predicted PMI

**What it shows.** A Doppler-spread channel (off-grid Doppler, one DFT-shift
period over 8 intervals). Left: SGCS of ONE report scored at each future
interval s — R15/R16 hold their precoder, R18 (N₄ ∈ {2,4,8}) applies its
per-interval *predicted* precoders, holding the last one beyond its window.
Right: the harness `feedback_delay_slots` knob — the report is applied d
intervals late, delay-obliviously, for the standard scheme set.

**Why it looks like this (left).**

* R15/R16 decay together (0.92 → 0.60 by s = 7): a held precoder ages at
  the channel's Doppler rate regardless of how well it was quantized.
* Each R18 curve stays flat *inside its window* and decays after it:
  N₄ = 4 holds ~0.91 through s = 3 then drops; N₄ = 8 holds ~0.86–0.93
  across the whole horizon. Prediction works because the reported Doppler
  shifts rotate the per-tap coefficients forward in time.
* **The starting points order inversely with N₄** (s = 0: N₄=2 0.933 >
  N₄=4 0.914 > N₄=8 0.858): a longer window means the quantized
  two-shift (Q = 2) Doppler model must explain more off-grid rotation, so
  instantaneous fidelity is traded for horizon coverage. N₄ = 8 crosses
  above the others from s ≈ 2 — choose N₄ by the expected feedback period.

**Why it looks like this (right) — investigated.** R18's aging curve is
only marginally above R16's (0.797 vs 0.792 at d = 3), which at first looks
like the prediction gain vanished. It hasn't — the harness applies reports
*delay-obliviously*: R18's interval-s precoder is scored against interval
s+d, so a constant phase-staleness of d intervals remains, identical to the
staleness R16 suffers; R18 only fixes the intra-window evolution, a small
term by comparison. A delay-aware gNB would instead start reading the
predicted sequence at offset d (left panel's per-interval view), recovering
the full gain for d < N₄. The two panels deliberately bracket the
deployment assumptions: delay-aware (left) vs delay-oblivious (right).
Type I barely ages (0.642 → 0.629) — a wideband beam is too coarse to
notice phase drift — and R17 ages like the other held Type II reports.

**Fix landed (S4).** `evaluate(..., delay_aware=True)` implements the
delay-aware gNB described above: scoring interval j applies the *predicted*
interval d + j (clamped to the report's last interval). The right panel now
carries a dotted "R18 … (delay-aware)" curve sitting clearly above the
oblivious one (0.840 vs 0.797 at d = 3, 0.69 vs 0.64 at d = 6) — the
prediction gain the left panel showed by hand, now harness-level. Locked in
`tests/test_restrictions_and_harness.py::TestHarnessKnobs::test_delay_aware_recovers_r18_prediction`.

**Config.** `scripts/fig_05_mobility.py`, 60 drops, rank 1,
max_doppler = 1 shift over an 8-interval period. Pair with fig_04's right
panel for the bits side of the trade: R18's flat horizon costs one report.
