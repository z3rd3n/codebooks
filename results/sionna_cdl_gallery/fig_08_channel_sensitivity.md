# fig_08_channel_sensitivity.png — robustness to richness and estimation noise (CDL-C)

**What it shows.** Standard scheme set, rank 1. Left: SGCS vs "number of multipath
rays". Right: SGCS vs measurement SNR (complex Gaussian estimation noise added to
the channel the UE sees; scoring uses the true channel), with a dashed
"R16 (4-slot meas.)" curve giving R16 the 4-interval observation window R18
enjoys.

**Why it looks like this.**

* **Left panel is flat — by design, not a bug.** CDL has a *fixed* cluster
  structure: there is no "number of rays" knob to sweep, so every x-value uses the
  identical CDL-C channel and each scheme draws a perfectly horizontal line
  (Type I 0.585, R15 0.943, R16 0.953, R17 0.896, R18 0.958). The synthetic
  figure's point — that all codebooks degrade as the channel densifies — has no
  CDL analog at fixed model. **The real channel-richness story for CDL is the
  CDL-A…E model sweep in `results/sionna_cdl/cdl_models.png`**, where Type I
  swings from 0.30 (rich NLOS CDL-B) to 0.87 (near-LoS CDL-D).
* **Right panel is the meaningful one and maps cleanly.** All schemes lose SGCS as
  estimation noise rises; the Type II family degrades fastest at low measurement
  SNR (it fits more parameters to a noisier channel) — at −10 dB R15 drops to 0.17
  while Type I, with almost nothing to corrupt, holds 0.33.
* **The 4-slot fairness control.** Giving R16 the same 4-interval observation
  window R18 uses (noise averages over slots) lifts it from 0.19 → **0.49** at
  −10 dB and keeps it on par with R18 throughout — i.e. R18's apparent
  noise-robustness is mostly its longer observation window, not the Doppler basis.

**CDL vs synthetic.** Only the right panel is informative on CDL; the left is
intentionally flat and cross-referenced to the model sweep.

**Config.** `scripts/cdl_fig_08_channel_sensitivity.py`, CDL-C, (4,2) array,
N₃ = 8, rank 1, 60 drops.
