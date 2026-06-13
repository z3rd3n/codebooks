# fig_11_frequency_granularity.png — fidelity and cost vs N₃ (CDL-C)

**What it shows.** N₃ swept 4…32 on CDL-C (the physical selectivity is fixed by
the model's 100 ns delay spread; only the reporting granularity changes). Left:
mean SGCS vs N₃ for Type I, R15, and R16 at p_v = 1/4 and 1/2. Right: measured
bits per report vs N₃ (log).

**Why it looks like this.**

* **R15 holds fidelity flat and pays linearly.** SGCS ≈ 0.947 at every N₃ because
  it quantizes each subband independently — but bits climb **linearly** 116 → 688
  as N₃ grows, the per-subband i2 cost.
* **R16 *gains* fidelity with N₃, cheaply.** Its M_v = ⌈p_v·N₃⌉ DFT taps resolve
  the delay structure better as the grid refines (pc4 SGCS 0.855 → 0.958 from
  N₃ = 4 to 32; pc6 0.925 → 0.958), while bits grow only ~logarithmically
  (47 → 238) — the tap indicators, not per-subband coefficients. N₃ = 24, 32
  exercise the R16 two-level (i15/i16) tap-indication path.
* **Type I is the floor** (flat 0.638): a single wideband beam plus per-subband
  co-phase cannot exploit finer granularity.

**CDL vs synthetic.** On the synthetic channel the ray delays are scaled to a
fixed fraction of the band so that selectivity is held constant by construction;
on CDL the delay spread is a genuine physical 100 ns and N₃ simply changes how
finely the codebook reports it. The R15-flat / R16-rises-cheaply contrast — the
core message — is reproduced, with R16 reaching slightly higher fidelity here
because CDL-C's delay clustering is well matched to a DFT-tap basis.

**Config.** `scripts/cdl_fig_11_frequency_granularity.py`, CDL-C (100 ns delay
spread), (4,2) array, rank 1, SNR 10 dB, 60 drops.
