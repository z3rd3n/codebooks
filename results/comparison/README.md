# Codebook comparison — the five-family story in six figures

A presentation-grade comparison of the implemented 3GPP NR PMI codebook families, rendered by
`scripts/figures/comparison_gallery.py` from the *same JSON data* as the main gallery
(`results/fig_*.json`, regenerated 2026-07-02 with UCI-exact overhead accounting) — no separate
simulation, so every number here matches the main gallery bit-for-bit.

**Families and fixed colors** (used identically in every figure):
<span style="color:#1f77b4">**R15 Type I**</span> (blue) ·
<span style="color:#d62728">**R15 Type II**</span> (red) ·
<span style="color:#e6a817">**R16 eType II**</span> (amber) ·
<span style="color:#9467bd">**R17 FeType II PS**</span> (purple) ·
<span style="color:#2ca02c">**R18 eType II Doppler**</span> (green).
Grey dashed = the full-CSI eigen bound (unbeatable reference).

Baseline channel unless stated: synthetic sparse 4-ray, (4,2) dual-pol array (P = 16), N₃ = 8.
The same comparisons on realistic 38.901 CDL-C channels live in `results/sionna_cdl_gallery/`
(orderings identical; Type II fidelity ceilings shift up).

---

## 1 · `cmp_1_se_vs_snr.png` — how much rate each codebook actually delivers

**What it shows.** Single-user spectral efficiency vs SNR at ranks 1 and 2, against the per-drop
eigen upper bound.

**How to read it.** Vertical distance to the grey dashed line is the *price of quantized
feedback*. Parallel curves = a constant dB offset; diverging curves = a structural deficit.

**Takeaways.**
* All Type II variants (R15/R16/R18) ride within ~0.1–0.15 b/s/Hz of the bound — beam-domain
  linear combination with per-subband (or per-tap) coefficients is essentially sufficient on a
  sparse channel.
* Type I pays a fixed ~0.5 b/s/Hz at rank 1 (one DFT beam + 2-bit co-phase cannot steer between
  grid points), and roughly **double** that at rank 2: its second layer is locked to a fixed
  orthogonal partner beam, so the pair spans the channel's 2-D subspace poorly
  (column SGCS 0.26; subspace SGCS 0.52 — see `fig_01_se_vs_snr.md`).
* R17 sits between: its fidelity is capped by the gNB's beamformed-port basis (evaluated through
  a unitary DFT PEB here), not by its own quantizer.

## 2 · `cmp_2_rate_distortion.png` — the plane an ML scheme has to beat

**What it shows.** Every spec configuration knob (L, N_PSK, subband amplitude, all R16/R17
paramCombinations) as one marker in the fidelity-vs-bits plane; per-family running-best lines;
the black dotted global Pareto frontier.

**How to read it.** Up and to the left is better. The shaded region above the frontier is
*unclaimed territory*: any learned CSI encoder that lands there beats every standardized
configuration at that budget.

**Takeaways.**
* **R16 owns the entire frontier** from ~40 bits up: frequency-domain compression (M_v taps
  instead of N₃ subbands) dominates R15 at every budget — the R15 markers sit strictly right of
  the R16 curve at equal SGCS.
* R15 Type II configurations cluster at 2–6× the bits of the R16 configuration with equal
  fidelity.
* R17's markers are Pareto-dominated *on this antenna-domain channel* — its budget lives in the
  port-selection index. On beam-domain channels (its design regime, fig_09) it wins instead:
  0.85 SGCS at 82 bits.
* Type I is the 23–37 bit floor: nothing standardized is cheaper, and nothing there is better.

## 3 · `cmp_3_overhead_anatomy.png` — where the bits actually go

**What it shows.** The rank-2, N₃ = 18 feedback budget of each family, split by what the bits
encode (spatial basis / delay basis / Doppler basis / selection structure / amplitudes / phases),
with per-family totals. Element-exact splits (TS 38.214 PMI fields + 38.212 widths) are in
`fig_03_overhead_breakdown.json`.

**Takeaways.**
* R15 Type II's 923 bits are ~94% per-subband amplitudes+phases (i₂,₁/i₂,₂ = 864 bits at
  N₃ = 18): the payload that FD compression exists to remove.
* R16 cuts the total to 430 bits but its single biggest item becomes the *bitmap* (i₁,₇ = 144 b,
  ~33%) — structure, not values.
* R18 = R16 with the bitmap (and coefficient space) doubled by Q = 2 Doppler bins plus a 4-bit
  shift index: 580 bits to cover **four** future intervals, vs 4 × 923 for re-reported R15.
* R17 is the cheapest Type II report (155 b) because reciprocity moved the spatial job to the
  gNB: it reports no q₁/q₂/beam combination at α = 1, just taps and coefficients.

## 4 · `cmp_4_scaling_laws.png` — how cost grows with each dimension

**What it shows.** Total report bits vs frequency granularity N₃, beam count L, and time coverage
N₄, from the spec bit compositions (K^NZ = K₀ budget convention, rank 2, (16,1) array).

**Takeaways.**
* **N₃**: R15 is linear (~38 b/unit — every subband re-quantized); R16/R18 sub-linear (only
  M_v = ⌈N₃/4⌉ taps and the bitmap grow). The R15:R16 ratio grows from 1.5× at N₃ = 4 to 3.2× at
  N₃ = 72 — wideband deployments are where eType II earns its name.
* **L**: everyone is linear in L. Spatial-domain cost is never compressed by any release — L is
  the one knob that always costs.
* **N₄**: the R18 punchline. R15/R16 must re-report per interval (×N₄); R18's one predicted
  report grows only by the ⌈log₂(N₄−1)⌉-bit shift indices: 257 → 345 bits from N₄ = 1 to 8,
  vs 5960 for R15. Per covered interval, R18 is the cheapest Type II report that exists.

## 5 · `cmp_5_mobility.png` — what "predicted PMI" buys, and what it doesn't

**What it shows.** Left: SGCS at each slot interval after a *single* report on a channel with
off-grid Doppler; R18 windows shaded. Right: the classic CSI-aging experiment (report applied d
intervals late), including R18 with a delay-aware gNB.

**Takeaways.**
* A held R15/R16 precoder decays from ~0.92 to ~0.60 SGCS over 7 intervals. R18's per-interval
  precoders stay ≥ 0.86 *inside* the N₄-window they were predicted for — and decay just like
  everyone else beyond it. Match N₄ to the CSI period, not longer.
* Aging (right): if the gNB just replays a stale report, R18 decays *identically* to R16 — the
  phase error from applying interval j at time j+d is the same. The prediction gain only
  materializes when the gNB applies the *predicted* interval j+d (dotted curve: +0.09 SGCS at
  d = 4). Doppler codebooks are a gNB-scheduling feature as much as a UE-feedback feature.
* Type I barely ages (nothing sharp to mis-steer) but from a floor of 0.64.

## 6 · `cmp_6_scorecard.png` — the one-slide summary

**What it shows.** Min–max normalized heatmap of the five families over SE (ranks 1–2), rank-1
SGCS, mobility SGCS, and bits per report (inverted — fewer is greener), raw values printed in
each cell.

**How to choose.**

| If you need… | Take | Because |
|---|---|---|
| a floor / control baseline | **Type I** | 23 bits, ~0.5 b/s/Hz below bound, unbeatable robustness to estimation noise (fig_08) |
| best fidelity per bit, static users | **R16 eType II** | owns the rate–distortion frontier at every budget ≥ ~40 b |
| minimum overhead with gNB reciprocity | **R17 FeType II PS** | 82–155 b; wins on beam-domain channels, loses off its design regime |
| mobility / long CSI periodicity | **R18 Doppler** | only family whose fidelity survives its report window; cheapest per covered interval |
| a spec-simple wideband-amplitude scheme | **R15 Type II** | historically first, strictly dominated by R16 on cost — use as ablation, not as champion |

**Caveats.** Min–max normalization exaggerates small spreads (the SE columns span < 0.6 b/s/Hz);
read the printed values, not just the colors. Single channel family per figure — R17's column
would look very different on a beam-domain channel (fig_09), and all Type II columns rise on
CDL-C (`sionna_cdl_gallery/fig_12_summary_table.md`).

---

### Reproduce

```bash
python scripts/figures/make_all_figures.py     # refresh the underlying JSONs (Monte Carlo)
python scripts/figures/comparison_gallery.py   # re-render these six figures from the JSONs
```
