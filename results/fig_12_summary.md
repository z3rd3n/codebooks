# fig_12_summary.png — the five-family scorecard

**What it shows.** Five axes, min–max normalized across the families (outer
ring = best of the five on that axis): SE@10 dB at ranks 1 and 2, SGCS,
compactness (fewest bits / own bits), and mobility (mean SGCS on future
intervals 1–3 of one report under Doppler). Raw numbers:
`fig_12_summary_table.md` / `.json`. This is the paper's qualitative
comparison table ("Discussion of 5G Codebooks"), quantified on one channel.

**Why the shapes look like this.**

* **Type I (blue) is a spike, not a polygon:** maximal compactness (23 b),
  minimal everything else (SGCS 0.64, mobility 0.63). It is the
  low-complexity baseline the paper says every 5G phone must support —
  nothing more.
* **R15 Type II (orange) is R16's shape shifted inward on compactness**
  (201 b vs 156 b for *less* fidelity, 0.904 vs 0.923): R16 dominates it
  pointwise here, which is the delay-domain-compression story of
  figs 02/03/04 in one glance. R15's remaining role is hardware support
  (it predates R16) rather than a distinct trade.
* **R16 (yellow) is the balanced all-rounder:** near-max on both SE axes
  and SGCS at mid compactness; its only weak axis is mobility (0.735) — a
  held report ages like any other.
* **R17 (purple) is the budget vertex:** 2.4× cheaper than R16 (82 b) at
  intermediate fidelity (0.729). Two caveats keep its polygon small here:
  this channel (4 rays) is denser than its design regime (it wins
  outright on 2-ray beam-domain drops, fig_09), and its α-scaling makes it
  the only family whose fidelity *grows* with the array (fig_10).
* **R18 (green) is R16 plus the mobility vertex** (0.917 vs ≤0.735 for
  everyone else — the single largest normalized gap in the chart) at the
  worst compactness (239 b *per report*; per covered interval it is
  actually the cheapest Type II, fig_04 right — the radar's static
  per-report accounting understates it). Its slight SGCS/SE edge over R16
  on this static channel is the Q-doubled K₀ budget (fig_10 notes).

**Reading guide / caveats.** Min–max normalization makes axes relative:
the SE axes span only 0.5 b/s/Hz (fig_01 explains the compression), while
the SGCS axis spans 0.29 — visually equal rings, very different physical
spreads. Compactness uses per-report bits, which flatters R15/R16 vs R18
under mobility (use fig_04's right panel for the time-coverage view) and
flatters Type I always. No single channel is neutral: this one (sparse,
static-with-Doppler-probe, antenna-domain + PEB for R17) was chosen to
give every family its intended measurement, but the per-figure analyses
are the trustworthy detail; the radar is the executive summary.

**Config.** `scripts/fig_12_summary.py`, 60 drops, (4,2) array, N₃ = 8.
