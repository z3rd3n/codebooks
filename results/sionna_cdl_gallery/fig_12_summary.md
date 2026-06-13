# fig_12_summary.png — scorecard radar across all axes (CDL-C)

**What it shows.** Five min-max-normalized axes (1 = best among the five families
on this channel): SE@10 dB rank 1, SE@10 dB rank 2, SGCS rank 1, compactness
(fewest bits / own bits), and mobility (mean SGCS on future intervals 1–3 of one
report under CDL Doppler). The raw numbers are in `fig_12_summary_table.md` and
the JSON.

**Why it looks like this.**

* **R18 is the largest envelope.** It ties or leads on SE (rank 1 7.81, rank 2
  12.0), SGCS (0.96), and **owns the mobility axis** (0.92 vs 0.63–0.69 for the
  others) — the only family that predicts. Its one weak vertex is compactness
  (the most bits, 207), the cost of the Doppler block.
* **R16 is the all-round efficient shape:** essentially R18's fidelity (0.95
  SGCS, 7.81 SE) at fewer bits (156 → compactness 0.15), with mobility limited to
  holding (0.69). On a static deployment it is the practical pick.
* **R15 overlaps R16 on fidelity/SE but is less compact** (197 bits) — its
  per-subband encoding buys no extra fidelity here.
* **R17 is a balanced mid-size shape** — good fidelity (0.90) at low cost
  (compactness 0.28, second only to Type I), the value of port selection.
* **Type I is a spike on compactness only** (1.0 — by far the fewest bits, 23)
  and the floor on every fidelity/mobility axis: the cheap-but-coarse extreme.

**CDL vs synthetic.** The shapes are the same as the synthetic scorecard; CDL-C
pushes the Type II/R18 fidelity vertices closer to 1 and, because the mobility
axis is now driven by real UE motion, R18's mobility lead is more pronounced.

**Config.** `scripts/cdl_fig_12_summary.py`, CDL-C, (4,2) array, N₃ = 8, 60 drops;
mobility axis at 30 km/h over a 4-interval horizon.
