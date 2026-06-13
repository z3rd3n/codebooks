# Repro: review Fig. 5 -- eType-II relative gain vs feedback overhead

Source: Qin & Yin, *A Review of Codebooks for CSI Feedback in 5G New Radio and Beyond*, **arXiv 2302.09222**, Fig. 5.

## Operating point

- Array: `AntennaConfig.standard(8, 2)` -> **P = 32** CSI-RS ports (dual-pol (N1,N2)=(8,2)).
- N3 = 13 PMI frequency units; **MU-MIMO**, K = 4 users, **rank 1** per user; reference SNR = 15 dB; 40 Monte-Carlo drops (seed 0, paired across schemes).
- Channel: `RandomRayChannel` (sparse multipath, 4 rays) -- Type II's design regime.

## Machinery reused (no new evaluator)

- Sum rate: `evaluate_mu` (ZF across reported rank-1 directions) -- the same evaluator behind `fig_06_mu_mimo.py`.
- Overhead (x): `evaluate(...).overhead_bits`, i.e. mean `scheme.total_overhead_bits(pmi)` over the same drops.
- Relative gain (y): `100 * sum_rate(scheme) / sum_rate(R15 Type I)`.
- R16 sweep generalizes the `fig_02_rate_distortion.py` paramCombination loop; `Mv = ceil(p_v * N3 / R)` is derived, so each point is labeled with the real (L, Mv) it yields rather than the paper's hard-coded grid.

## Result

| family | config | overhead [bits] | rel. gain [%] | on Pareto frontier |
|---|---|---:|---:|:--:|
| R15 Type I | rank-1 baseline | 34 | 100.0 | **yes** |
| R15 Type II | L=2 | 178 | 104.2 |  |
| R15 Type II | L=3 | 240 | 107.4 |  |
| R15 Type II | L=4 | 348 | 111.6 |  |
| R16 eType II | pc1 (L=2,Mv=4) | 62 | 105.1 | **yes** |
| R16 eType II | pc2 (L=2,Mv=4) | 85 | 106.2 | **yes** |
| R16 eType II | pc3 (L=4,Mv=4) | 111 | 111.5 | **yes** |
| R16 eType II | pc4 (L=4,Mv=4) | 158 | 114.9 | **yes** |
| R16 eType II | pc5 (L=4,Mv=4) | 179 | 115.4 | **yes** |
| R16 eType II | pc6 (L=4,Mv=7) | 224 | 115.6 | **yes** |
| R16 eType II | pc7 (L=6,Mv=4) | 224 | 119.4 | **yes** |
| R16 eType II | pc8 (L=6,Mv=4) | 249 | 119.6 | **yes** |

**Ordering reproduced:** R15 Type I is the 100% floor; R15 Type II rises with L but pays a steep overhead; R16 eType II dominates at higher overhead and its paramCombination knobs trace the upper-left frontier (more gain per bit) -- matching the paper's qualitative Fig. 5.

## Differences from the paper (trend, not bit-exact -- by design)

- **Channel model.** We use `RandomRayChannel` (a stochastic ray channel), not the paper's system-level simulator with a spatially consistent channel and a deployment layout.
- **Scheduler / load.** The paper reports gains at resource utilization RU ~ 70% under a proportional-fair scheduler; we use a fixed K = 4-user ZF sum rate with an equal power split and no scheduler.
- **Normalization.** Both normalize to R15 Type I, but our reference is the rank-1 ZF MU sum rate of Type I on the same drops, so absolute percentages differ from the paper's throughput-gain percentages.
- **(L, Mv) grid.** The paper sweeps (L, Mv) directly; the spec parameterizes R16 by paramCombination (Mv derived from p_v and N3), so we sweep the spec table and annotate the realized (L, Mv).

These are intentional: the goal is to reproduce the *ordering, the Pareto-frontier shape, and the relative spread*, not the absolute throughput-gain values, which depend on a system simulator we do not run.
