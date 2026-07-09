# GLIMPSE comparison gallery

Figures comparing **GLIMPSE** (one-sided learned CSI feedback, `docs/ml/glimpse.md`)
against the 3GPP R15–R19 codebook families. Every scheme is scored on the
**same** frozen Sionna 38.901 CDL drops through `nr_csi.eval.evaluate` —
identical channels, SGCS/SE metric, and honest bit accounting as the codebook
figures in `results/`. Regenerate:

```bash
.venv/bin/python scripts/ml/eval_glimpse.py --model models/glimpse_p32 \
    --cdl A,B,C,D,E --drops 200 --ablation --out results/ml
.venv/bin/python scripts/ml/eval_glimpse.py --model models/glimpse_p32 \
    --cdl C --rank 2 --drops 200 --out results/ml
.venv/bin/python scripts/ml/make_glimpse_figures.py --results results/ml --cdl C
```

| Figure | What it shows |
|---|---|
| `fig_glimpse_frontier.png` | **The headline.** SGCS and SE@10 dB vs feedback bits: every codebook configuration + the GLIMPSE (learned/OMP/LS) grids, the codebook Pareto frontier, and the eigen upper bound. GLIMPSE's learned frontier sits up-and-left of the codebook frontier. |
| `fig_glimpse_gain.png` | Feedback bits to reach a target SGCS: GLIMPSE vs the best codebook at matched fidelity — the overhead-reduction bars. |
| `fig_glimpse_decoders.png` | The one-sided story: the **same** UE report decoded three ways (learned / OMP / least-squares). The learned decoder is a pure gNB-side upgrade over classical compressed sensing; LS is the prior-free floor. |
| `fig_glimpse_models.png` | Cross-profile generalization: SGCS at matched bits across CDL-A…E. The KLT basis + decoder are fitted on the CDL-A/B/C mix; CDL-D/E are near-LoS, outside the training distribution. |

Raw numbers are in the paired `frontier_cdl*.json`, `glimpse_gain_summary.json`,
and `glimpse_generalization.json`.

## Headline numbers (CDL-C, (4,4) P=32, N3=8, rank 1, 200 drops)

**Overhead at matched fidelity — GLIMPSE vs the cheapest codebook that reaches it:**

| target SGCS | codebook bits | GLIMPSE bits | reduction |
|---|---|---|---|
| 0.80 | 49 | 24 | **−51%** |
| 0.85 | 90 | 36 | **−60%** |
| 0.90 | 116 | 60 | **−48%** |
| 0.92 | 162 | 60 | **−63%** |

GLIMPSE reaches SGCS **0.98** at 192 bits (SE within 0.02 of the eigen bound),
a fidelity no codebook attains at any overhead on this channel; and it operates
below 40 bits, where the codebooks have no configurations.

**Ablation (SGCS @ 96 bits):** KLT + water-fill + learned **0.952** ≈ KLT +
water-fill + LS **0.952** (the learned decoder is optional on CDL) ≫ KLT +
uniform-B **0.936** ≫ random basis **0.07** (the KLT basis is the whole gain).

**Generalization:** GLIMPSE wins on the in-distribution rich-scattering
profiles (CDL-A/B/C) and *loses* to the fixed-DFT codebooks on the
out-of-distribution near-LoS profiles (CDL-D/E) — the honest cost of matching a
distribution. See `docs/ml/glimpse.md` §4.4.

**UE complexity:** the encoder is one `m×D` matrix multiply — 8× (P=16) to 64×
(P=128) cheaper than the R16 beam search it replaces, growing with array size.
