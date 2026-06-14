# Repro: review Fig. 5 -- eType-II relative gain vs feedback overhead

Source: Qin & Yin, *A Review of Codebooks for CSI Feedback in 5G New Radio and Beyond*, **arXiv 2302.09222**, Fig. 5.

## Operating point

- Array: `AntennaConfig.standard(8, 2)` -> **P = 32** CSI-RS ports (dual-pol (N1,N2)=(8,2)).
- N3 = 13 PMI frequency units; **MU-MIMO**, K = 4 users, **rank 1** per user; reference SNR = 15 dB; 40 Monte-Carlo drops (seed 0, paired across schemes).
- Channel: Sionna 3GPP TR 38.901 CDL-C.

## Machinery reused (no new evaluator)

- Sum rate: `evaluate_mu` (ZF across reported rank-1 directions) -- the same evaluator behind `fig_06_mu_mimo.py`.
- Overhead (x): `evaluate(...).overhead_bits`, i.e. mean `scheme.total_overhead_bits(pmi)` over the same drops.
- Relative gain (y): `100 * sum_rate(scheme) / sum_rate(R15 Type I)`.
- **beta sweep (the paper's knob).** Each R16 curve holds (L, M_v) fixed and varies beta in {1/8, 1/4, 1/2, 3/4} via explicit `R16ParamCombo` objects (the `combo=` override on `R16Type2Codebook`), since the standardized `R16_PARAM_COMBOS` table bundles (L, p_v, beta) into eight rows and has no beta = 1/8. `M_v = ceil(p_v * N3 / R)`, so N3 = 13 gives M_v in {4, 7}.

## Result

| family | config | overhead [bits] | rel. gain [%] |
|---|---|---:|---:|
| R15 Type I | rank-1 baseline | 34 | 100.0 |
| R15 Type II | L=2 | 178 | 184.3 |
| R15 Type II | L=3 | 239 | 227.6 |
| R15 Type II | L=4 | 348 | 261.7 |
| R16 eType II | (L,Mv)=(2,4), beta=1/8 | 48 | 116.5 |
| R16 eType II | (L,Mv)=(2,4), beta=1/4 | 62 | 166.1 |
| R16 eType II | (L,Mv)=(2,4), beta=1/2 | 86 | 185.0 |
| R16 eType II | (L,Mv)=(2,4), beta=3/4 | 97 | 187.2 |
| R16 eType II | (L,Mv)=(2,7), beta=1/8 | 76 | 166.1 |
| R16 eType II | (L,Mv)=(2,7), beta=1/4 | 95 | 183.6 |
| R16 eType II | (L,Mv)=(2,7), beta=1/2 | 124 | 188.5 |
| R16 eType II | (L,Mv)=(2,7), beta=3/4 | 131 | 189.0 |
| R16 eType II | (L,Mv)=(4,4), beta=1/8 | 83 | 173.8 |
| R16 eType II | (L,Mv)=(4,4), beta=1/4 | 111 | 232.3 |
| R16 eType II | (L,Mv)=(4,4), beta=1/2 | 162 | 268.2 |
| R16 eType II | (L,Mv)=(4,4), beta=3/4 | 187 | 273.9 |
| R16 eType II | (L,Mv)=(4,7), beta=1/8 | 130 | 223.9 |
| R16 eType II | (L,Mv)=(4,7), beta=1/4 | 177 | 263.0 |
| R16 eType II | (L,Mv)=(4,7), beta=1/2 | 233 | 276.5 |
| R16 eType II | (L,Mv)=(4,7), beta=3/4 | 241 | 277.4 |

**Ordering reproduced:** R15 Type I is the 100% floor; R15 Type II rises with L but pays a steep overhead; each R16 eType II (L, M_v) curve climbs up-and-right as beta grows (more reported coefficients = more bits and more gain), and the (4, *) curves dominate the (2, *) curves -- the per-(L,M_v) curve family of the paper's Fig. 5.

## Differences from the paper (trend, not bit-exact -- by design)

- **Channel model.** We use Sionna 3GPP TR 38.901 CDL-C (a link-level channel), not the paper's system-level simulator with a spatially consistent channel and a deployment layout.
- **Scheduler / load.** The paper reports gains at resource utilization RU ~ 70% under a proportional-fair scheduler; we use a fixed K = 4-user ZF sum rate with an equal power split and no scheduler.
- **Normalization.** Both normalize to R15 Type I, but our reference is the rank-1 ZF MU sum rate of Type I on the same drops, so absolute percentages differ from the paper's throughput-gain percentages.

These are intentional: the goal is to reproduce the *ordering, the per-(L,M_v) curve shape, and the relative spread*, not the absolute throughput-gain values, which depend on a system simulator we do not run.
