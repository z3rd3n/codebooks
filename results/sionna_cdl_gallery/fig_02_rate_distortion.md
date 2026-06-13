# fig_02_rate_distortion.png — fidelity vs feedback overhead (CDL-C)

**What it shows.** Every configuration knob of every family swept on the same
CDL-C drops: Type I modes 1–2; R15 (L∈{2,3,4} × N_PSK∈{4,8} × subband
amplitude); R16 paramCombinations 1–8; R17 paramCombinations 1–8 (via the DFT
PEB). Left: mean SGCS vs feedback bits; right: SE@10 dB vs bits. The Pareto
frontier (fewest bits per fidelity level) is dotted — the plane a learned CSI
scheme must beat.

**Why it looks like this.**

* **R16 owns the frontier.** Its points span bits 33→155 at SGCS 0.73→0.95;
  for almost every fidelity level it is the cheapest family, because its
  frequency-domain DFT-tap basis (M_v taps + a non-zero-coefficient bitmap)
  encodes the channel's delay structure compactly instead of re-quantizing every
  subband. R15 reaches the same top fidelity (0.95) but only at 2–3× the bits
  (up to 223) — its per-subband i2 is the cost.
* **Type I is the cheap, low-fidelity corner** (23–37 bits, SGCS 0.64, flat):
  it has only two configuration modes and a single wideband beam.
* **R17 is a tight, cheap cluster** (60–154 bits, SGCS 0.87–0.93): evaluated
  in the beam domain its free port selection is efficient, but on CDL-C it tops
  out a touch below the antenna-domain Type II families (which see the full
  spatial channel directly).
* **The SE panel is compressed** (all families 7.3–7.8 at 10 dB) for the same
  `log₂` reason as fig_01 — the fidelity panel is the discriminating one.

**CDL vs synthetic.** CDL-C lifts the whole frontier (top SGCS 0.95 vs ~0.93)
and tightens the family clusters, but the *ordering* and the R16-owns-the-knee
conclusion are identical to the synthetic plane.

**Config.** `scripts/cdl_fig_02_rate_distortion.py`, CDL-C, 80 drops, rank 1,
SNR 10 dB; one marker per configuration, R17 via PEB.
