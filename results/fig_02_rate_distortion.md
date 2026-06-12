# fig_02_rate_distortion.png — the rate–distortion plane (fidelity vs bits)

**What it shows.** Every configuration knob of every family evaluated on the
same drops, placed on the (feedback bits, fidelity) plane: Type I modes 1–2;
R15 L×N_PSK×subband-amplitude grid (12 points); R16 paramCombinations 1–8;
R17 paramCombinations 1–8 via the DFT PEB. Left: SGCS; right: SE@10 dB.
The dotted line is the Pareto frontier — the curve a learned CSI feedback
scheme must beat to claim a win over 3GPP.

**Why it looks like this.**

* **The frontier is traced almost entirely by R16** (yellow triangles), with
  Type I holding the ultra-cheap end (~25–45 bits, SGCS 0.65–0.75). This is
  the cleanest quantitative statement of the R16 design goal: delay-domain
  compression dominates R15's per-subband reporting at every operating
  point — R16 pc6 reaches SGCS 0.924 at 156 bits while R15's best
  (L=4, 8-PSK, SA) needs ~250 bits for 0.913.
* **R15 points (orange squares) sit strictly inside the frontier**: its i₂
  payload scales with N₃ = 8 subbands regardless of how compressible the
  channel is. Its knobs move it along a shallower trade: subband amplitude
  costs ~56 bits for ~+0.01 SGCS here (the channel is delay-sparse, so
  per-subband amplitude adds little).
* **R17 (purple diamonds) is dominated on this channel** (best 0.835 at
  182 bits). The default 4-ray channel spreads beam-domain energy over more
  ports than its K₁ window covers; R17's regime is *very* sparse channels
  (see fig_09, where it wins at 82 bits on 2-ray drops). Port selection
  trades search freedom for cheap indexing — that only pays when the PEB
  concentrates energy on few ports.
* The right panel compresses the same ordering into a ~0.45 b/s/Hz SE band
  (see fig_01 for why SU-SE differences are small at rank 1).

**Notable.** Within R16, the dominant knob is p_v (M_v taps): pc4→pc6
(p ¼→½) buys +0.03 SGCS for ~70 bits, while β (the K₀ coefficient budget)
moves points much less on this 4-ray channel — the budget only binds on
denser channels (fig_08 left).

**Config.** `scripts/fig_02_rate_distortion.py`, 80 drops, rank 1, paired
seeds. R18 is intentionally absent: on a static channel its report is an
R16 report plus Doppler bits (it lives on the mobility axis, figs 04/05).
