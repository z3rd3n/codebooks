# Figure suite: comparing all codebook families in every measured sense

## Context

The framework (R15–R18, 7 variants) exposes four metric groups — spectral
efficiency (`metrics.se`), fidelity (SGCS/NMSE, `metrics.similarity`),
feedback overhead (`metrics.overhead` + serialization-honest
`overhead_bits()`), and the harness realism knobs (CSI aging, measurement
noise, MU-MIMO). The existing scripts only reproduce the paper's two
figures (f1, f2). This plan adds a self-contained gallery of comparison
figures so that (a) the 3GPP baselines are characterized along *every*
axis the harness measures, and (b) a future ML scheme can be dropped into
any of these scripts and inherit the full comparison for free.

## Conventions shared by all scripts (`scripts/figlib.py`)

* Default benchmark: `AntennaConfig.standard(4, 2)` (P = 16 ports), N3 = 8,
  `RandomRayChannel` (sparse multipath, n_paths = 4, max_delay = 3), rank 1,
  paired seeds across schemes (same drops for every curve).
* Standard scheme set at matched L = 4: Type I (mode 1), R15 Type II (L=4),
  R16 eType II (paramCombination 6: L=4, p=1/2, beta=1/2), R17 FeType II
  (paramCombination 5: alpha=1/2 -> K1=8, L=4, M=2), R18 Doppler
  (paramCombination 7: L=4, N4=4).
* Port-selection codebooks are *beam-domain* schemes (the gNB applies a PEB
  to the CSI-RS first). They are evaluated through a unitary per-polarization
  DFT PEB wrapper (`BeamDomainChannel`); unitarity makes SE/SGCS directly
  comparable with antenna-domain schemes on the same physical drops.
  Headline figures mark these schemes "(via PEB)".
* Every figure writes `results/<name>.png` plus `results/<name>.json` with
  the exact plotted numbers (auditable, diffable).
* Common CLI: `--drops`, `--seed`, `--out`, `--fast` (smoke-test sizes).
* One fixed color per codebook family across all figures; the eigen
  upper bound is a grey dashed line.

## The figures

1. **fig_01_se_vs_snr** — SE vs SNR for the standard set + eigen upper
   bound, panels for rank 1 and rank 2. The headline "who is better and by
   how much" figure (generalizes paper f1 to all families, dual-pol).
2. **fig_02_rate_distortion** — SGCS vs feedback bits and SE@10 dB vs bits,
   sweeping each family's knobs (Type I modes; R15 L x N_PSK x subband
   amplitude; R16 paramCombinations 1–8; R17 combos 1–8 via PEB). Pareto
   frontier highlighted. The plane an ML scheme must beat — the single most
   important figure for the project's purpose.
3. **fig_03_overhead_breakdown** — stacked per-element bits (i11, i12, i17,
   i23, ...) from actual reported PMIs at the rank-2 / N3=18 configuration,
   grouped into spatial basis / delay basis / Doppler basis / selection
   structure / amplitudes / phases. Shows *where* each generation spends
   its bits and why R16+ compresses (i2 dominates R15; bases+bitmap
   dominate R16+).
4. **fig_04_overhead_scaling** — overhead formula scaling laws (Tables
   bit1/bit2): bits vs N3 (R15 linear, R16/R18 ~log), bits vs L
   (combinatorial + linear growth), bits to cover N4 intervals (R15/R16
   re-report linearly, R18 single predicted report). Log-scale panels;
   paper-f2 configuration (16,1), v=2.
5. **fig_05_mobility** — the R18 story: (a) per-future-interval SGCS of one
   report under Doppler (R15/R16 held vs R18 predicted, N4 in {2,4,8});
   (b) SGCS vs feedback delay (harness `feedback_delay_slots`) for the
   standard set. Quantifies CSI aging and prediction gain.
6. **fig_06_mu_mimo** — MU-MIMO ZF sum rate from reported PMIs
   (`evaluate_mu`, paper eq. 2): (a) sum rate vs SNR at K=4 users vs the
   full-CSI ZF reference; (b) sum rate vs K at 15 dB. Shows the Type II
   families' MU advantage that SU evaluation undersells.
7. **fig_07_rank_adaptation** — (a) SE vs SNR for R16 at fixed ranks 1–4
   plus the auto-RI envelope (`eval.select_rank`); (b) auto-RI rank
   distribution vs SNR. Multi-layer behavior and where extra layers pay.
8. **fig_08_channel_sensitivity** — robustness: (a) SGCS vs number of
   multipath rays (sparse -> dense; Type II's design regime and the K0
   budget saturation); (b) SGCS vs measurement SNR (estimation noise via
   the harness knob). Standard set.
9. **fig_09_port_selection** — applicability boundary: grouped bars of
   SGCS for regular vs PS variants (R15/R15-PS/R16/R16-PS/R17 + Type I) on
   (a) the antenna-domain channel and (b) the beam-domain (post-PEB) view
   of the same drops. Quantifies the "Applicable Scenarios" section.
10. **fig_10_array_scaling** — SE@10 dB, SGCS, and bits vs array size for
    supported (N1,N2) in {(2,2),(4,2),(6,2),(8,2)} (P = 8..32): Type I's
    gap grows with N, Type II overhead growth vs fidelity.
11. **fig_11_frequency_granularity** — SGCS and measured bits vs N3
    (4..32) with the physical delay spread held proportional: R15's
    per-subband i2 grows linearly while R16's M_v-tap compression stays
    flat-ish at equal fidelity; exercises the N3>19 two-level path.
12. **fig_12_summary** — scorecard: min–max-normalized radar over
    {SE@10 rank 1, SE@10 rank 2, SGCS, overhead compactness, mobility
    robustness} for the five families + a raw-numbers markdown/JSON table.

## Runner

`scripts/make_all_figures.py` executes every `fig_*.py` in order with a
shared `--fast/--drops/--seed` passthrough and prints a timing/status
summary. Full-quality regeneration of the whole gallery is a single
command; `--fast` is the CI smoke mode.

## Non-goals

* No Sionna dependency in the gallery (NumPy `RandomRayChannel` only) —
  the 38.901 integration stays in `pytest -m sionna`.
* R19, Type I ranks 3–8, exact paper-f2 bar heights: out of scope as
  documented in the README.
