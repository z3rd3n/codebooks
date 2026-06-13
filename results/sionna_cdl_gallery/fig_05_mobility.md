# fig_05_mobility.png — CSI aging vs the R18 predicted PMI (CDL-C, 30 km/h)

**What it shows.** A **time-evolving** CDL-C channel (UE at 30 km/h, 2 ms slot
intervals — real Doppler from UE motion, not a synthetic knob). Left: per-interval
SGCS of one report over an 8-interval horizon — R15/R16 hold their precoder (it
ages) while R18 with N₄∈{2,4,8} reports a *predicted* precoder per future
interval. Right: harness CSI aging — mean SGCS when the report is applied
`feedback_delay` intervals late, plus a dotted delay-aware R18 curve.

**Why it looks like this.**

* **Held precoders decay steeply.** R15/R16 fall from 0.95 at interval 0 to ~0.29
  by interval 7: at 30 km/h the channel decorrelates within a few intervals, so a
  single report goes stale fast.
* **R18 predicts as far as its window reaches.** N₄ = 8 holds SGCS ≈ 0.9 across
  the *entire* horizon; N₄ = 4 tracks through interval ~4 then decays; N₄ = 2
  only briefly beats the held curves. The Doppler-domain DFT basis extrapolates
  the precoder forward — the larger N₄, the longer the reach (at the bit cost in
  fig_03/fig_04).
* **The aging panel confirms it at the harness level.** Every scheme loses SGCS
  with feedback delay, but **delay-aware R18** (the gNB applies the predicted
  interval d+j instead of replaying interval j) sits above plain R18 at every
  delay (e.g. 0.89 vs 0.86 at delay 1, 0.73 vs 0.61 at delay 3) — the prediction
  gain the left panel shows by hand.

**CDL vs synthetic.** This is the figure that benefits most from CDL: the
synthetic version injects an off-grid Doppler shift by hand, whereas here the
aging is a genuine consequence of UE motion through the 38.901 cluster geometry.
The prediction-beats-holding conclusion is the same but more convincing. (The
suptitle text "off-grid Doppler" is inherited from the shared plotting code; on
CDL the Doppler is physical UE mobility.)

**Config.** `scripts/cdl_fig_05_mobility.py`, CDL-C, 30 km/h, 2 ms intervals,
8-interval horizon, (4,2) array, N₃ = 8, 60 drops.
