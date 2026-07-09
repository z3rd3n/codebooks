# PRISM: One-Sided CSI Feedback via a Published Mixture of KLT Sketches

**Precoder Reporting via an Indexed Sketch Mixture** — the successor to
GLIMPSE (`docs/ml/glimpse.md`), designed to remove its one honest failure
mode while keeping (and strengthening) its deployability story.

> **Abstract.** GLIMPSE reports a fixed KLT sketch of the eigen-precoder and
> beats the 3GPP Type II codebooks by 30–60% overhead at matched fidelity on
> the channel distribution it was fitted to — and *loses* to the codebooks on
> out-of-distribution near-LoS profiles (CDL-D/E), the bias–variance cost of
> matching a single distribution. PRISM removes that failure by replacing the
> single published basis with a **published mixture of K KLT bases**, fitted
> offline by a Lloyd-style K-subspaces alternation on a training mix that
> covers every deployment regime. Per report, the UE picks the basis whose
> m-row sketch captures the most energy — an argmax over K small matrix
> multiplies, still **zero learned parameters** — and signals the choice with
> ⌈log₂K⌉ bits (2 bits at K=4). Reconstruction stays **pure least squares**:
> no neural network on either side, in any configuration. On identical frozen
> Sionna 38.901 CDL banks, PRISM K=4 beats the best codebook *and* GLIMPSE on
> **all five** CDL profiles simultaneously (at ~144 bits: 0.987/0.933/0.984/
> 0.990/0.982 vs GLIMPSE's 0.822 on CDL-E), needs 47–62% fewer bits than the
> cheapest codebook at matched fidelity on both rich-scattering and near-LoS
> channels, and a variant fitted *without* profile E still scores 0.972 on E —
> the mixture covers *regimes*, not memorized profiles. The K-subspaces fit
> discovers the LoS/NLoS split unsupervised (cluster purities 95–100%).

## 1. The problem PRISM solves

GLIMPSE's cross-profile table (`docs/ml/glimpse.md` §4.4) is the motivation:

| profile | GLIMPSE | best codebook | verdict |
|---|---|---|---|
| CDL-A/B/C (rich scattering, in-fit) | 0.979/0.915/0.972 | 0.973/0.869/0.932 | GLIMPSE wins |
| CDL-D/E (near-LoS, out-of-fit) | 0.922/0.822 | 0.963/0.929 | **codebooks win** |

A single KLT basis is the optimal linear sketch for **one** second-moment
matrix. A deployment's channel population is a **mixture** (LoS/NLoS regimes,
delay-spread classes), and the KLT of a pooled mixture is optimal for none of
its members — fitting one basis on everything pays an **averaging penalty**
(measured in §4: the pooled basis is *worse than GLIMPSE* on CDL-B, 0.868 vs
0.915). The correct statistical object is one basis per regime.

## 2. Method

Publish K bases `{(A_k, σ_k)}` instead of one. Everything else — the
angle–delay transform, the standardization, the reverse-water-filled
Lloyd–Max quantizer, the least-squares inverse, the rateless prefix property
within a basis — is unchanged from GLIMPSE (and reuses its code:
`nr_csi.ml.quantizer`, transforms via `GlimpseCodec`).

**UE (still zero learned parameters).** Compute the angle–delay eigen vector
`g` (as GLIMPSE); compute `k* = argmax_k ||A_k[:m] g||²` (K m×D matrix
multiplies + argmax; the winner's sketch is already computed); standardize by
`σ_{k*}`, quantize under basis k*'s published water-filled allocation; report
`⌈log₂K⌉` index bits + `2mB` payload bits per layer. A two-stage variant
selects on an `m_sel`-row prefix (energy captured by the top rows dominates
the comparison) and computes only the winner's full sketch:
`K·m_sel·D + m·D` MACs.

**Fitting (offline, one covariance-level estimation).**
`nr_csi.prism.mixture.fit_mixture`: Lloyd/K-subspaces alternation —
assign each training vector to the basis capturing the most of its energy at
`m_ref=16`, refit each cluster's KLT, repeat; multiple restarts. Plain
random-assignment initialization is **degenerate** (every random half shares
the pooled covariance, so all components start identical); we seed
k-means++-style: component 0 from a random subset, each next component from
the samples the current dictionary captures *worst* (residual-greedy).
The objective (mean captured energy at m_ref) is monotone → converges.
K=1 degenerates to a single pooled KLT — the "broad GLIMPSE" ablation.

**gNB.** `ĝ = A_{k*}^H y` — least squares under the signalled basis. PRISM is
deliberately **fully linear**: GLIMPSE already measured that a learned decoder
adds nothing in-distribution once the basis is right (LS = learned to three
decimals), so PRISM ships with no neural network anywhere.

This is classical **universal transform coding** (Effros–Chou: a codebook of
KLTs with per-block selection) and **mixture-PCA / K-subspaces clustering**
(Tipping–Bishop MPPCA; Vidal), composed for the one-sided CSI constraint: the
mixture is a *published spec constant*, exactly the kind of object 3GPP
already standardizes — a codebook, here a codebook *of transforms*.

## 3. Fit diagnostics (60k A/B/C + 24k D/E train rows, m_ref=16)

Captured-energy fraction (of 8.0 total; `scripts/prism/fit_prism.py`):

| codec | A/B/C | D | E |
|---|---|---|---|
| GLIMPSE basis (A/B/C fit) | 0.925 | 0.909 | **0.727** |
| PRISM K=1 (pooled A–E) | 0.909 ↓ | 0.946 | 0.854 |
| **PRISM K=4** | **0.945** | **0.986** | **0.982** |
| PRISM K=8 | 0.952 | 0.989 | 0.982 |
| PRISM K=4 no-E (never saw E) | 0.948 | 0.986 | 0.976 |

The K=4 clusters discover the regime structure **unsupervised**: cluster 1 is
99% CDL-D, cluster 3 is 95% CDL-E, clusters 0/2 split A/B/C into two
rich-scattering sub-regimes. K=2 does *not* separate LoS from NLoS (both
clusters stay ABC-dominated) — the regime split emerges at K=4; K=8 adds
little. K=4 (2 index bits) is the headline.

## 4. Results (frozen Sionna banks, seed 777, same drops as results/ml)

Bank determinism is *verified* before every eval: `eval_prism.py` re-scores a
stored GLIMPSE row on the rebuilt bank and asserts SGCS equality to 1e-9.

**Cross-profile SGCS at ~144 bits** (`fig_prism_profiles.png`):

| profile | best codebook | GLIMPSE | PRISM K1 (pooled) | **PRISM K4** |
|---|---|---|---|---|
| CDL-A | 0.973 | 0.979 | 0.974 | **0.987** |
| CDL-B | 0.869 | 0.915 | 0.868 | **0.933** |
| CDL-C | 0.932 | 0.972 | 0.955 | **0.984** |
| CDL-D | 0.963 | 0.922 | 0.970 | **0.990** |
| CDL-E | 0.929 | 0.822 | 0.938 | **0.982** |

PRISM K4 wins **everywhere**, including +0.008–0.018 over GLIMPSE on the
in-distribution profiles (the mixture also splits rich scattering more
finely). Worst-profile fidelity: PRISM 0.933 vs GLIMPSE 0.822 vs codebook
0.869. The pooled K=1 row is the measured averaging penalty: below GLIMPSE on
A/B/C (0.868 on B!) — broader data alone does not fix OOD; the mixture does.

**Overhead at matched fidelity** (bits, cheapest config reaching target):

| target | CDL-C: cb / PRISM | CDL-D: cb / PRISM | CDL-E: cb / PRISM |
|---|---|---|---|
| 0.90 | 116 / 62 (−47%) | 48 / 34 (−29%) | 94 / 50 (−47%) |
| 0.92 | 162 / 62 (−62%) | 65 / 34 (−48%) | 117 / 62 (−47%) |
| 0.95 | – / 98 | 87 / 50 (−43%) | – / 98 |
| 0.97 | – / 122 | – / 74 | – / 122 |

("–" = no codebook configuration reaches the target at any overhead.)
On CDL-E the best codebook saturates at 0.935 (225 b); PRISM reaches 0.997.

**Honest caveats, measured.**
- At *low* targets (≤0.85) on near-LoS profiles the 24-bit Type I report is
  unbeatable — a specular channel *is* a DFT beam; PRISM needs ~34 bits
  there (the mixture's dense sketch cannot undercut a 1-beam index).
- The no-E variant scores 0.972 on unseen CDL-E (vs 0.982 with E in the fit,
  codebook 0.929): coverage is at the *regime* level, so an unseen profile
  inside a covered regime generalizes — but a genuinely novel regime
  (neither rich-scattering nor near-LoS) would still degrade; the
  distribution-blind random-basis fallback and the codebooks remain the
  safety net, unchanged from GLIMPSE.
- Same per-geometry fitting caveat as GLIMPSE ((N₁,N₂,N₃)-specific), and the
  same no-prediction caveat (reports the eigen precoder; Doppler/aging is
  orthogonal).

## 5. Complexity (complex MACs, shared eigen step excluded, K=4, m=16)

| array | P | PRISM direct | PRISM 2-stage (m_sel=4) | R16 select | speedup |
|---|---|---|---|---|---|
| (4,2) | 16 | 8,228 | 4,132 | 17,440 | 2.1× / 4.2× |
| (4,4) | 32 | 16,420 | 8,228 | 67,104 | 4.1× / 8.2× |
| (16,2) | 64 | 32,804 | 16,420 | 264,736 | 8.1× / 16.1× |
| (16,4) | 128 | 65,572 | 32,804 | 1,053,216 | 16.1× / 32.1× |

K× GLIMPSE's cost, still a small fraction of the Type II search, still
O(P) vs the search's O(P²). UE stores K constant matrices + one Lloyd–Max
table; zero learned parameters. gNB: one matrix multiply. No neural network
exists anywhere in the pipeline.

## 6. Reproducing

```bash
# 1. Near-LoS raw data (A/B/C dataset + GLIMPSE targets already exist)
.venv/bin/python scripts/dataset/generate_cdl_dataset.py \
    --configs 4x4 --n-samples 30000 --profiles D,E \
    --freq-res 64 --fft-size 256 --seed 7 --out data/cdl_p32_de
.venv/bin/python scripts/ml/prepare_targets.py data/cdl_p32_de --n3 8 \
    --out data/ml/targets_p32_de

# 2. Fit the mixture codecs (K sweep + no-E OOD variant)
.venv/bin/python scripts/prism/fit_prism.py --out models

# 3. Evaluate on the frozen banks (verifies bank determinism first)
.venv/bin/python scripts/prism/eval_prism.py --out results/prism
#    ... and on the deployment-level MIXED bank (40 drops x CDL-A..E)
.venv/bin/python scripts/prism/eval_prism_mixed.py --out results/prism

# 4. Figures (internal comparison incl. GLIMPSE / paper figures, PRISM vs codebooks)
.venv/bin/python scripts/prism/make_prism_figures.py --results results/prism
.venv/bin/python scripts/prism/make_paper_figures.py --results results/prism
```

**Mixed-bank headline** (`frontier_mixed.json`, 200 drops): PRISM K4 0.977 vs
best codebook 0.937 at ~144 b; −50/−56/−57% bits at SGCS 0.85/0.90/0.92; no
codebook reaches 0.94+ while PRISM reaches 0.995. K sweep @98 b: K1 0.903
(only *tracks* the codebook Pareto — the mixture, not the KLT alone, is the
gap), K2 0.915, K4 0.953, K8 0.955. The camera-ready paper
(`docs/prism/paper/`) reports PRISM vs the 3GPP codebooks only, mixed-first.

Source: `src/nr_csi/prism/` (`mixture.py`, `scheme.py`); tests:
`tests/prism/`. GLIMPSE (`src/nr_csi/ml/`) is untouched and remains the
K=1/single-fit special case.
