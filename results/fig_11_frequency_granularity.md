# fig_11_frequency_granularity.png — fidelity and cost vs N₃

**What it shows.** N₃ swept 4 → 32 with the physical selectivity held fixed
(ray delays up to 37.5% of the band), so only the *reporting granularity*
changes. Left: SGCS. Right: measured bits per report (log). R16 shown at
two compression factors (pc4: p_v = 1/4, pc6: p_v = 1/2).

**Why it looks like this.**

* **R15 is flat in fidelity (0.914 → 0.899) and linear in cost
  (118 → 707 b):** it quantizes every subband independently, so finer
  granularity buys nothing on a channel whose delay structure it already
  tracks — it just re-pays the i₂ block N₃ times. The mild fidelity *drop*
  with N₃ comes from the wideband amplitudes (i₁,₄) being shared across
  ever more diverse subbands while only the phases adapt per subband
  (subband amplitude is off by default).
* **R16 climbs in fidelity and stays cheap:** M_v = ⌈p_v·N₃⌉ taps grow in
  proportion to the (fixed-fraction) delay spread, so the DFT-tap model
  matches the channel better as N₃ grows: pc4 0.742 → 0.928, crossing R15
  at N₃ ≈ 12 with one third of R15's bits (125 vs 287 b). pc6 is above R15
  from N₃ = 8 onward. The bit curves grow ~linearly in M_v but with a slope
  ~4× (pc4) / ~2× (pc6) shallower than R15's; N₃ = 24, 32 exercise the
  two-level (i₁,₅/i₁,₆) tap-indication path without a visible kink.
* **Type I is flat at 0.64 and nearly free:** a wideband beam plus
  per-subband co-phase has no frequency-domain payload to scale.

**Notable / unexpected — explained.** R16 pc4 *loses* to R15 at N₃ = 4
(0.742 vs 0.914): M_v = ⌈4/4⌉ = 1 single delay tap must explain a channel
whose delay spread is 1.5 taps — the compression model is under-provisioned,
while R15 happily brute-forces the 4 subbands. The general rule: R16's
advantage requires M_v to reach the channel's effective delay support;
below that, per-subband reporting is more robust. This is the
frequency-domain mirror of fig_08's sparsity panel (budgets must cover the
channel's true dimensionality) and the measured counterpart of fig_04's
left panel (which shows the same N₃ scaling from the bit formulas alone).

**Config.** `scripts/fig_11_frequency_granularity.py`, 60 drops, rank 1,
(4,2) array, max ray delay = 0.375·N₃ taps so the per-subband channel
statistics are N₃-invariant.
