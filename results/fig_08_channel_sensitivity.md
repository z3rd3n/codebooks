# fig_08_channel_sensitivity.png — robustness to sparsity and estimation noise

**What it shows.** Standard scheme set, rank 1, SGCS. Left: vs the number
of multipath rays (1 → 12). Right: vs the measurement SNR (complex Gaussian
estimation noise added to the channel the UE sees; scoring uses the true
channel).

**Why it looks like this (left).**

* Every codebook degrades as the channel densifies — all of them are
  sparsity-coded (L beams, M_v taps, K₀ coefficients) and a 12-ray channel
  simply has more significant dimensions than the budgets cover. The decay
  *rates* differ: Type I falls 0.93 → 0.45 (a single beam explains an ever
  smaller energy fraction), the Type II families fall gently
  (R16 0.99 → 0.80), staying ~0.35 above Type I throughout.
* R18 sits slightly above R16 at every point (e.g. 0.881 vs 0.851 at
  8 rays) — its spec coefficient budget K₀ = ⌈2βLM_vQ⌉ carries the Q = 2
  factor, and on a static channel the unused Doppler bin donates its half
  of the budget to more delay taps (see fig_10 notes; paper eq. for K₀ in
  the R18 section).
* R17 tracks between Type I and the Type IIs: its K₁ = 8-port window in
  the PEB domain covers more than one beam but less than a tuned L = 4
  combination on these 4-ray-domain drops.

**Why it looks like this (right) — investigated.** Below ~0 dB measurement
SNR the ranking *inverts*: Type I (0.30 at −10 dB) beats R15/R16
(0.18/0.19). Coarse codebooks are noise-robust — picking the argmax beam is
a 1-of-256 decision that noise must flip to hurt, while Type II's
least-squares coefficients fit the noise directly (classic
estimation-variance vs quantization-bias trade). The crossover back to the
clean ordering happens by ~5 dB, and above 10 dB all curves are within
~0.01 of their noiseless values — Type II needs decent CSI-RS SNR to be
worth its bits.
**R18's surprising lead at −10/−5 dB (0.32/0.61) is not a codebook effect:**
it measures N₄ = 4 slots, and with i.i.d. per-slot noise its selection
implicitly averages ~4× the observations. Probe: at −5 dB, R16 given the
same 4-slot average scores 0.79 vs 0.46 from one slot — the gain is the
longer measurement window, available to any scheme that observes 4 slots.

**Config.** `scripts/fig_08_channel_sensitivity.py`, 60 drops; right panel
at 4 rays, rightmost x-position = noiseless.
