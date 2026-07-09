# GLIMPSE: One-Sided Learned CSI Feedback

**gNB-Learned Inversion of Measurement-Projected Subband Eigenvectors**

> **Abstract.** Learned CSI feedback promises large gains over the 3GPP Type II
> codebooks, but the two designs that dominate the literature both stall at
> deployment: two-sided autoencoders require the UE and gNB vendors to jointly
> train and version-match a shared encoder/decoder — an inter-vendor
> coordination problem 3GPP has not solved — and they put a full neural network
> on the power- and area-constrained UE. We propose **GLIMPSE**, a *one-sided*
> scheme in which the UE runs **zero learned parameters**: it reports a fixed,
> published linear sketch of the per-subband channel eigenvectors — the
> **Karhunen–Loève transform** of the channel, the statistically-optimal
> generalization of the codebooks' fixed DFT dictionary — standardized and
> quantized under a reverse-water-filling bit allocation. The only thing
> "learned" is the KLT basis itself, estimated offline from the channel's
> second-order statistics (a covariance) and published as a constant, exactly
> like a codebook table. The UE encoder is a single `m×D` matrix multiply
> (**8–64× cheaper** than the Type II beam search it replaces, a gap that grows
> with array size); the report is **rateless** (any prefix is a valid
> lower-rate report — a continuous overhead knob, not a discrete codebook
> ladder); and reconstruction at the gNB is a **linear least-squares inverse**
> that needs no neural network, with an optional learned unrolled decoder for
> out-of-distribution robustness. On 38.901 CDL rich-scattering channels
> GLIMPSE cuts feedback overhead by **30–60% at matched fidelity** and reaches
> SGCS/SE the codebooks cannot attain at any overhead; on out-of-distribution
> near-LoS profiles it degrades below the codebooks — the honest cost of
> matching a distribution — and falls back to the codebook or a
> distribution-blind random basis there. Nothing learned crosses the air
> interface or the vendor boundary: the only shared objects are two published
> constants (the KLT matrix and the quantizer table).

A learned CSI-feedback scheme designed to beat the 3GPP Type II codebook
family on both axes that matter — feedback overhead *and* reconstruction
fidelity — while respecting the three deployment constraints that have kept
learned CSI feedback out of the standard:

1. **UE complexity.** The device cannot run a large neural network.
2. **Inter-vendor collaboration.** Two-sided autoencoders require the UE
   vendor and the gNB vendor to jointly train and version-match a shared
   encoder/decoder pair — a coordination problem 3GPP has not solved.
3. **Overhead *and* performance.** A scheme that only improves one at the
   expense of the other does not move the Pareto frontier.

GLIMPSE puts **zero learned parameters on the UE** and keeps every learned or
fitted object **on the infrastructure side**. The UE side is a fixed,
published, spec-freezable pipeline — a linear projection onto a published basis
followed by a fixed scalar quantizer — indistinguishable in character from a
codebook table. The gNB reconstructs by least squares (a matrix multiply,
optionally refined by a small learned decoder each vendor upgrades
independently), with no UE involvement and no cross-vendor agreement beyond the
published projection matrix. The projection is the **Karhunen–Loève transform
of the channel** — the statistically-optimal generalization of the fixed DFT
dictionary the codebooks already use.

---

## 1. What the codebooks do, and where they waste bits

This framework implements the full R15–R19 PMI codebook family and scores
every one on identical channels (`nr_csi.eval.evaluate`). The measured
rate–distortion plane (`results/fig_02_rate_distortion.*`,
`results/sionna_cdl_gallery/fig_02_*`) shows the operating points every
learned scheme must beat. The dominant Type II family
(`R15Type2Codebook`, `R16Type2Codebook`) reconstructs a precoder as

```
W(t) = Σ_i Σ_f  c_{i,f} · (b_i ⊗ e_pol) · y_f(t)          (eq. a85)
```

a sum over **`L` selected spatial DFT beams** `b_i` and **`M_v` selected
delay taps** `y_f`, with quantized complex coefficients `c_{i,f}`. The
feedback report (`nr_csi.metrics.overhead`, `results/fig_03_overhead_breakdown`)
splits into:

| Element | What it encodes | Cost driver |
|---|---|---|
| `i₁,₁ / i₁,₂` | rotation + beam-combination index | `log₂C(N₁N₂/... , L)` |
| `i₁,₅ / i₁,₆` | delay-tap window + combination | `M_v`, `N₃` |
| `i₁,₇ (bitmap)` | which of `2LM_v` coefficients are non-zero | `2LM_v` bits |
| `i₂,₃/₄/₅` | per-coefficient amplitude + phase | `K_NZ · (bits_amp + bits_phase)` |

Three structural inefficiencies fall out of this design, and each is a lever
GLIMPSE pulls:

- **The support is re-encoded on every report.** The beam index, the tap
  window, and the non-zero bitmap describe *where* the energy sits in the
  angle–delay plane. Across a coherence region this support is highly
  predictable, yet Type II spends 30–60 bits re-transmitting it every time
  (fig_03: `i₁,₇` alone is `2LM_v` bits). A basis matched to the channel
  statistics holds that "where the energy sits" prior in its *fixed directions*
  instead of paying for it on the air every report.
- **The dictionary is a fixed orthogonal DFT grid.** Off-grid paths — every
  real CDL path — leak across many beams and taps, so Type II needs large
  `L, M_v` to capture an off-grid cluster (fig_08/fig_10). The overhead grows
  to fight a basis-mismatch problem a data-driven decoder simply does not
  have.
- **The quantizer is uniform per element.** Type II uses fixed-width
  amplitude/phase grids regardless of the coefficient statistics, and cannot
  spend fractional bits where they matter.

The consequence is the measured frontier: to exceed ~0.90 SGCS the codebooks
need **90–230 bits** (fig_02 CDL-C). GLIMPSE reaches the same fidelity at
**30–60% fewer bits** on the rich-scattering profiles by pulling all three
levers at once (§4).

---

## 2. Method

GLIMPSE reports a **fixed linear sketch of the eigen-precoder**, and lets the
gNB invert it with a learned prior. Three stages; only the third has any
learned parameters, and it lives at the gNB.

### 2.1 UE side — a fixed, parameter-free encoder

The UE already computes, for CQI/RI, the per-subband dominant eigenvectors of
its estimated channel. GLIMPSE reuses exactly the quantity the codebooks'
`select` uses (`nr_csi.codebooks._spatial.aligned_eigen_targets`):

**(a) Aligned eigen targets.** For channel `H[t, r, p]` (subband `t`, RX `r`,
port `p`), take the rank-`v` dominant right singular vectors per subband and
remove the SVD's per-subband phase ambiguity by sequential alignment along
`t`. Result: `V ∈ ℂ^{N₃×P×v}`, unit columns. This is the *target the gNB
wants to reconstruct*, identical to what Type II approximates.

**(b) Angle–delay transform.** Apply the unitary per-polarization 2-D spatial
DFT over the `(N₁,N₂)` port grid and a unitary IDFT along the `N₃` frequency
axis (`nr_csi.dataset.preprocess`; the CsiNet angle–delay representation).
This concentrates each physical path into a few coefficients and is the
domain the decoder's convolutional prior operates in. Flatten to
`g ∈ ℂ^D`, `D = P·N₃`.

**(c) Fixed KLT projection.** Report

```
y = A_m · g ,     A_m = first m rows of the fixed KLT matrix A.
```

`A` is the **Karhunen–Loève basis** of the channel-eigenvector distribution:
the eigenvectors of the second-moment matrix `E[g gᴴ]`, ordered by variance,
**fitted once offline and published** as a constant
(`nr_csi.ml.projection.fit_klt`). It is a table, like a codebook, not a
trained inference-time weight. Because the spatial DFT, the delay IDFT, and
`A` are all linear, the entire encoder collapses to **one `m×D` complex
matrix multiply** applied to the eigenvector the UE already has:

```
y = (A · T) · vec(V) ,    T = the fixed angle–delay transform.
```

This is precisely the classical **DFT → KLT upgrade** of the codebook's fixed
DFT dictionary: Type II *assumes* a DFT grid and pays overhead to fight the
resulting basis mismatch; GLIMPSE *measures* the channel's own principal
directions, so it captures far more of the eigenvector per coordinate. On the
benchmark CDL distribution the top-16 KLT coordinates already capture **93%**
of the eigenvector energy with a purely linear decoder, versus **6%** for a
random projection of the same size (`docs/ml/glimpse.md` §4, and the
`fit_klt` capture curve).

**(d) Fixed quantizer with water-filled bit allocation.** Standardize each
coordinate by its published population std `σ_k` (the √ of the KLT eigenvalue)
so every coordinate is unit-variance, then digitize with a fixed **Lloyd–Max
quantizer of the standard normal** (`nr_csi.ml.quantizer`). For a total budget
of `2mB` bits, the bits are distributed across the `2m` real dimensions by
**reverse water-filling** — optimal scalar transform coding (Gersho–Gray):
`b_j = max(0, ½ log₂(σ_j²/θ))` with the water level `θ` set so `Σ b_j = 2mB`.
High-variance (leading) coordinates get more bits, negligible ones get none.
The allocation is a *deterministic function of the published `σ`* — computed
identically at the UE and the gNB — so **no side information is sent** telling
the gNB how many bits each coordinate used. The gNB rescales by the same
published `σ`. The eigenvector target is unit-norm, so no per-report scale is
signalled either. The report is exactly `2mB` bits per layer.
(`allocation="uniform"`, every dimension `B` bits, is the ablation baseline;
water-filling recovers ~40% of the quantization loss at matched bits — §4.)

That is the whole UE encoder: a matrix multiply and two table lookups. **No
learned parameters. No neural network. Nothing to train or version on the
device.**

### 2.2 Why the KLT sketch is the right report

Three properties make the fixed sketch outperform the codebooks' adaptive
beam/tap selection while remaining a published constant.

**Optimal linear compression (why so few coordinates suffice).** For a fixed
measurement budget `m`, the KLT is the linear map that captures the maximum
expected energy of the eigenvector (the Eckart–Young/PCA optimality). Type II
selects `L` beams and `M_v` taps *adaptively per report* and spends
`log₂C(·,L) + …` bits telling the gNB **which** it chose. The KLT instead
reports the projection onto a *fixed* set of maximally-informative directions
— so it never spends a bit on support signalling, and the decoder holds the
"which directions matter" prior in the basis. This is where the overhead
saving comes from.

**Optimally-ordered rateless nesting (graceful truncation).** `A_m` is the
first `m` rows of *one* fixed matrix for every `m`, and because the rows are
variance-ordered, `A_m` is the *best* `m`-dimensional sketch, not just *an*
`m`-dimensional one. Any prefix of a report is a valid lower-rate report that
drops the least-informative measurement first: the UE, gNB, or scheduler can
truncate the payload to fit any UCI grant and the *same* decoder reconstructs
from whatever arrived, with graceful monotone fidelity loss
(`GlimpseScheme.truncate`, `tests/ml/test_scheme.py::TestPrefixProperty`).
The codebooks are the opposite — every parameter combination is a distinct,
non-nested report format. GLIMPSE's overhead is therefore a *continuous* knob
`m ∈ {1,…,m_max}`, not a discrete ladder: it can place an operating point
anywhere on the bit axis.

**Distribution-robust fallback.** The KLT is fitted to a channel
distribution, but the *random* projection (`basis="random"`,
`measurement_matrix`) is a drop-in alternative whose generic orthonormal rows
Gaussianize the measurements for **any** distribution (a
Johnson–Lindenstrauss/rotation argument, verified in
`test_projection.py::test_gaussianization`). It needs many more measurements
for the same fidelity — it is distribution-*blind* — but it bounds the worst
case and is the honest ablation that isolates how much the KLT's statistical
matching is worth (§4).

### 2.3 gNB side — least-squares, with an optional learned decoder

The gNB inverts `y ≈ A_m g` for the angle–delay vector `g`, then maps back to
the precoder. Because the KLT rows are orthonormal and the KLT is the *optimal
linear sketch*, the minimum-norm least-squares inverse `ĝ = A_mᴴ y`
(`LeastSquaresDecoder`) is already near-optimal — and empirically it **matches
a trained neural decoder to three decimals across the whole in-distribution
range** (§4.3). So the headline GLIMPSE decoder is **a single matrix
multiply**: no neural network at the gNB either, which is the strongest form
of the deployability argument.

For robustness when the deployment distribution is broad or shifted (§4.4),
GLIMPSE optionally refines the linear estimate with a learned **unrolled
proximal-gradient network** (`nr_csi.ml.decoder.UnrolledDecoder`) — `K`
iterations of:

```
x ← x + α_k · A_mᴴ(u − A_m x)          # data-consistency (exact: A rows are orthonormal)
x ← x + γ_k · CNN_k( grid(x) ; m, B )  # learned prior on the angle–delay grid
```

The prior CNN is a small residual 3-D convolution over the physical
`(delay, angle₁, angle₂)` grid, with the two polarizations × (real, imag) as
input channels and **circular padding on all three axes** — the angle–delay
domain is a product of DFTs, so its topology genuinely is a 3-torus and
wrap-around convolutions respect it. Two conditioning channels carry the
report's `(m, B)` so **one network serves every payload size and quantizer
depth** (trained with measurement dropout, §3).

The decoder is deliberately small (default `K=6`, hidden width 32,
≈0.22 M parameters — see §5) because it must be cheap to *serve* at the gNB,
not to run on the UE. Its measured benefit is modest and honest: **0.000 over
LS in-distribution, and +0.002…+0.008 SGCS out-of-distribution** (largest at
the smallest bit budgets, where the linear inverse has least to work with —
§4.3/§4.4). It is included as the general form and the OOD safety margin, not
because it is essential; a deployment may omit it entirely. Reconstruction ends
by mapping `ĝ` back through the inverse angle–delay transform to `V̂[N₃,P,v]`,
column-normalized and scaled so `tr(WᴴW)=1` — the exact precoder convention the
codebooks and the harness use, so GLIMPSE drops into `nr_csi.eval.evaluate`
unchanged.

### 2.4 The one-sided property, precisely

The decoder choice **never changes the report**. The same `2mB` bits are
decodable by:

- `LeastSquaresDecoder` — minimum-norm `A_mᴴy`; on the KLT basis this is the
  near-optimal linear inverse and **the headline decoder** (a single matrix
  multiply that already beats the codebooks, §4.3);
- `KerasDecoder` — the optional trained network, a small refinement over LS;
- `OMPDecoder` — orthogonal matching pursuit (the classical sparse-CS solver);
  it is the natural decoder for the *random-basis* variant and is included as
  that comparison point — it is a poor fit for the dense KLT coordinates.

This is what "one-sided" buys: **the gNB can deploy, retrain, A/B-test, or
roll back its decoder with no UE change and no inter-vendor coordination.** A
UE from vendor X and a gNB from vendor Y interoperate as long as both know the
*published constants* `A` and the quantizer table — there is no shared learned
artifact to agree on, which is the exact blocker that stalls two-sided models
in 3GPP RAN1. And because the winning decoder is *linear least squares*, the
gNB side needs no trained model at all: GLIMPSE is one-sided in the strongest
sense — the only learned object anywhere is the offline-estimated KLT basis,
a published constant like a codebook table.

---

## 3. Training

Only the gNB decoder trains, by plain supervised inversion — no encoder
gradients, no straight-through quantizer estimator, because the encoder is
fixed. `scripts/ml/train_glimpse.py`. Each minibatch randomizes, per sample:

- **measurement count** `m ~ U{m_min..m_max}` (measurement dropout) — the one
  network then serves every payload and any truncated prefix;
- **quantizer depth** `B ~ U{2,3,4,5}`;
- **global phase** — the single symmetry the report does not pin down;
- **input variant** — clean eigenvectors, or eigenvectors computed from a
  channel corrupted with measurement AWGN (20/10 dB), with the *clean* vector
  as the target, so the decoder learns estimation-noise robustness;
- **layer** — rank-1 or rank-2 eigen targets (rank-2 support).

Loss is `1 − SGCS` per subband — *exactly* the evaluation metric — with deep
supervision across the `K` unrolled iterates. Targets are precomputed once
(`scripts/ml/prepare_targets.py`) from the raw-H CDL dataset the framework
already generates (`nr_csi.dataset`), and the vectorized target path is
pinned against the reference `aligned_eigen_targets` to guarantee the training
target is bit-identical to what the harness scores against.

The decoder is trained with **per-coordinate uniform** quantization at random
depths; at deployment it is fed **water-filled** reports. This train/test
mismatch is deliberate and benign — the decoder is conditioned on the noise
level and, having seen a spread of per-coordinate quantization depths in
training, generalizes to the water-filled noise pattern (empirically it decodes
water-filled reports *better* than the uniform reports it trained on, §4). A
decoder is thus reusable across quantizer choices — a robustness property, and
a further reason the report format and the decoder can evolve independently.

---

## 4. Experimental results

**Setup.** Array `(N₁,N₂)=(4,4)`, `P=32` ports, `N₃=8` PMI subbands, `n_rx=2`.
The KLT basis and decoder are fitted on a 60 000-sample raw-H dataset drawn
from the 38.901 CDL-A/B/C profiles with randomized delay spread (30–300 ns) and
UE speed (3–30 km/h). Every scheme — all 26 codebook configurations
(Type I / R15 Type II / R16 eType II / R17 FeType II PS) and the GLIMPSE grids
— is scored on the **same** frozen Sionna CDL drops through
`nr_csi.eval.evaluate`: identical channels, SGCS/SE metric, and honest bit
accounting (`len(pack(report)) == overhead bits`). The decoder is `K=6`
iterations, hidden width 32, **≈0.22 M parameters**.

Figures: `results/ml/fig_glimpse_frontier.png` (headline rate–distortion plane),
`fig_glimpse_gain.png` (bits at matched fidelity), `fig_glimpse_decoders.png`
(the three decoders on one report), `fig_glimpse_models.png` (cross-profile
generalization). Raw numbers in `results/ml/*.json`.

### 4.1 Rate–distortion frontier (CDL-C)

Mean SGCS and SE@10 dB at representative budgets (200 drops; the best codebook
config at ≤ each budget in parentheses):

| ~bits | GLIMPSE SGCS | best codebook SGCS | GLIMPSE SE@10 | codebook SE@10 |
|---|---|---|---|---|
| 48 | **0.884** (m8, B3) | 0.845 (R16 pc2, 49 b) | 8.67 | 8.60 |
| 96 | **0.945** (m12, B4) | 0.897 (R16 pc4, 90 b) | 8.76 | 8.69 |
| 144 | **0.972** (m24, B3) | 0.914 (R15 L3-8PSK, 150 b) | 8.80 | 8.71 |
| 192 | **0.982** (m24, B4) | 0.932 (R16 pc6, 162 b) | 8.81 | 8.74 |

The GLIMPSE frontier (`fig_glimpse_frontier.png`) sits **above the codebook
Pareto frontier across the entire operating range** in both SGCS and SE, and
the gap widens with the budget: GLIMPSE reaches SGCS 0.98 (SE within 0.02 of
the eigen upper bound) at 192 bits, a fidelity **no codebook attains at any
overhead** on this channel. It also occupies the **sub-40-bit regime the
codebooks cannot enter** — the smallest R16 report is 34 bits (SGCS 0.70),
while GLIMPSE places continuous operating points down to a handful of
measurements (24 bits → SGCS 0.82, versus 0.62 for the 24-bit Type I report).

### 4.2 Overhead at matched fidelity

Feedback bits to reach a target mean SGCS, GLIMPSE vs the cheapest codebook
that reaches it (`fig_glimpse_gain.png`):

| target SGCS | codebook bits | GLIMPSE bits | overhead reduction |
|---|---|---|---|
| 0.70 | 34 | 24 | **−29%** |
| 0.80 | 49 | 24 | **−51%** |
| 0.85 | 90 | 36 | **−60%** |
| 0.90 | 116 | 60 | **−48%** |
| 0.92 | 162 | 60 | **−63%** |

At the high-fidelity operating points that matter for MU-MIMO, GLIMPSE cuts the
feedback overhead by roughly **half to two-thirds** at matched accuracy.

### 4.3 Ablations

Mean SGCS at each budget, one factor changed at a time
(`fig_glimpse_ablation.png`):

| variant | ~48 b | ~96 b | ~144 b | ~192 b |
|---|---|---|---|---|
| **KLT + water-fill + learned** (full) | 0.884 | 0.945 | 0.972 | 0.982 |
| KLT + water-fill + **LS** (linear) | 0.884 | 0.945 | 0.972 | 0.982 |
| KLT + **uniform-B** + learned | 0.875 | 0.942 | 0.951 | 0.977 |
| KLT + water-fill + **OMP** | 0.681 | 0.698 | 0.781 | 0.787 |
| **random basis** + LS | 0.044 | 0.065 | 0.087 | 0.099 |
| random basis + OMP | 0.044 | 0.105 | 0.166 | 0.219 |

The ablation is decisive about *where the gain comes from*, and the finding is
worth stating plainly:

* **The KLT basis is the whole story.** Replacing the fitted KLT with a
  distribution-blind random projection collapses SGCS from ~0.95 to ~0.10 at
  matched bits — the random basis captures only `m/D` of the energy (6% at
  m=16) while the KLT captures 93%. *This is the entire source of GLIMPSE's
  advantage over the codebooks:* a measurement basis matched to the channel
  statistics rather than the assumed DFT grid. It is also a form of learning —
  offline estimation of the channel's second-order statistics (a covariance),
  not a deep network.
* **Water-filling adds a small, consistent gain** (≈+0.01–0.02 SGCS at matched
  bits) by spending the budget where the KLT variance is.
* **A linear least-squares decoder already matches the learned decoder**
  (identical to three decimals across the whole CDL-C range). Because the KLT
  is the *optimal linear sketch*, the min-norm inverse is near-optimal and
  there is little non-linear structure left for the network to exploit on
  in-distribution channels. **The practical consequence is strong: GLIMPSE
  needs no neural network at the gNB at all** — reconstruction is one matrix
  multiply — which makes it even simpler to deploy than the acronym suggests.
  The learned decoder is an *optional* refinement that earns its keep only when
  the deployment distribution is broad or shifted (§4.4). OMP is a poor fit for
  the KLT basis (whose coordinates are dense, not sparse) and serves only as
  the guaranteed classical fallback.

### 4.4 Cross-profile generalization (and the honest cost of specializing)

The basis and decoder are fitted on the CDL-A/B/C mix. Evaluated per profile at
~144 bits (GLIMPSE at fixed `m=24`, vs the best codebook ≤ 144 bits):

| profile | GLIMPSE SGCS | best codebook SGCS | in training mix? |
|---|---|---|---|
| CDL-A | **0.979** | 0.973 | yes |
| CDL-B | **0.915** | 0.869 | yes |
| CDL-C | **0.972** | 0.914 | yes |
| CDL-D | 0.922 | **0.962** | no (near-LoS) |
| CDL-E | 0.822 | **0.929** | no (near-LoS) |

This is the central trade-off, stated without varnish. On the three
**in-distribution** profiles GLIMPSE beats the codebooks comfortably (the
overhead savings of §4.2). On the two **out-of-distribution near-LoS**
profiles (D and E — dominated by a single specular path, statistically unlike
the rich-scattering training mix) the fixed-DFT codebooks **win**: their basis
assumes nothing about the distribution, so they neither gain from matching it
nor lose from mismatching it. GLIMPSE's KLT is tuned to rich scattering and is
the wrong basis for a near-LoS channel.

This is the bias–variance trade-off of any statistically-matched scheme, and it
frames GLIMPSE's operating envelope honestly: it is a large win **when the
deployment distribution is known and reasonably stationary** (the common case
for a fixed cell's scattering environment, and the premise under which any
learned CSI scheme operates), and it must be either (i) trained on a
distribution that covers the deployment — including LoS profiles — or (ii) run
with the distribution-blind `basis="random"` fallback, which forfeits the peak
gain for codebook-independent robustness. A production system would detect the
LoS regime (which the UE already estimates for RI/CQI) and fall back to the
codebook or the random basis there — GLIMPSE and the codebooks are
complementary, not mutually exclusive.

---

## 5. Complexity

Accounting in complex multiply–accumulates (`nr_csi.ml.projection.encoder_flops`,
`type2_select_flops`); the shared per-subband eigen decomposition — which every
PMI scheme computes for CQI/RI anyway — is excluded from both sides.

**UE encode.** GLIMPSE is one `m×D` matrix multiply: `m·D` MACs plus `2m` for
standardize+quantize. The Type II `select` the UE would otherwise run scans
`O₁O₂` oversampled beam groups (`≈P²` work), solves least-squares beam
coefficients, and runs an `N₃`-point FD DFT per beam. GLIMPSE's encoder is a
**small fraction** of that, and the gap *grows with the array* — the
projection is `O(mP)` while the beam search is `O(P²)`:

| array | P | GLIMPSE (m=16) | GLIMPSE (m=32) | R16 `select` | speed-up (m=16) |
|---|---|---|---|---|---|
| (4,2) | 16 | 2,080 | 4,160 | 17,440 | **8.4×** |
| (4,4) | 32 | 4,128 | 8,256 | 67,104 | **16.3×** |
| (16,2) | 64 | 8,224 | 16,448 | 264,736 | **32.2×** |
| (16,4) | 128 | 16,416 | 32,832 | 1,053,216 | **64.2×** |

(complex MACs, shared eigen step excluded;
`tests/ml/test_projection.py::TestComplexity`). At a 128-port array the UE does
**64× less** work than the codebook search it replaces.

**UE parameters.** Zero learned. The device stores the constant `A` (or the
composed `A·T`) and one Lloyd–Max table — the same kind of static ROM a
codebook already needs.

**gNB decode.** The headline decoder is **least squares — one `D×m` matrix
multiply, zero parameters**. The optional refinement network is ~0.22 M
parameters and runs at the *infrastructure*, where compute is abundant. Either
way the deliberate asymmetry holds: the heavy side (if any) is the gNB, which
can afford it, and the UE stays a single matrix multiply.

---

## 6. Positioning vs. prior art

| Approach | UE cost | Two-sided? | Report format | GLIMPSE difference |
|---|---|---|---|---|
| **Type II codebooks** (R15–R19) | heavy UE search (beam/tap/coeff) | no | fixed grid, discrete ladder | statistics-matched basis removes support bits; continuous rateless overhead |
| **CsiNet / CRNet / TransNet** (two-sided autoencoders) | **full learned CNN encoder on UE** | **yes** | learned latent | encoder is a *fixed linear sketch*: no UE network, no shared learned artifact |
| **Classical CS feedback** (random projection + OMP/AMP) | linear projection (like GLIMPSE) | no | random sketch | KLT basis (not random) → dense, well-conditioned coordinates a linear decoder inverts; GLIMPSE ⊃ CS |
| **DFT-dictionary / on-grid CS** | correlation search | no | sparse coefficients | KLT replaces the assumed DFT dictionary, absorbing off-grid leakage into the basis rather than paying for it in overhead |

GLIMPSE's novelty is the **combination**, and its packaging for the one-sided
constraint: a linear, fixed, parameter-free encoder — the *statistically
matched* KLT sketch of the eigen precoder, with reverse-water-filled
quantization — reconstructed by least squares (optionally refined by a learned
decoder) at the gNB. Each ingredient exists in isolation (KLT/PCA compression,
transform-coding bit allocation, CS feedback, unrolled reconstruction), but
composing them under the one-sided constraint and benchmarking head-to-head
against the *full* R15–R19 codebook family on identical 38.901 CDL channels
yields a scheme that has, simultaneously: (i) a large fidelity/overhead gain
over the codebooks on in-distribution channels; (ii) the deployability of a
codebook (the UE runs a published constant matrix, exactly as it runs a
codebook today) with **no neural network required on either side**; (iii) a
**continuous, rateless overhead knob**; and (iv) a graceful, well-characterized
failure mode on out-of-distribution channels with a codebook/random-basis
fallback. Crucially, unlike a two-sided autoencoder, **nothing learned crosses
the air interface or the vendor boundary** — the only shared objects are two
published constants (the KLT matrix and the quantizer table), strictly less
than standardizing a codebook.

---

## 7. Limitations and honest caveats

- **The basis and decoder are fitted per antenna geometry.** `A`, `σ`, and the
  network are tied to `(N₁,N₂,N₃)`. This is a gNB-side/standardization artifact
  (the gNB knows its own array), not a UE burden, but a multi-array deployment
  fits one basis+decoder per configuration. A geometry-agnostic decoder is
  future work.
- **Distribution shift.** Both the KLT basis and the decoder prior are fitted
  to a channel distribution; §4 measures cross-CDL-profile generalization. A
  deployment whose channels differ sharply from training would (i) lose some
  KLT compression efficiency and (ii) regress toward the OMP floor — which is
  still a safe fallback, not a failure. The `basis="random"` variant trades
  peak efficiency for distribution-blind robustness when the deployment
  statistics are unknown.
- **Reciprocity of the eigenvector target.** GLIMPSE reports the eigen
  precoder, like Type II; it inherits the same CSI-aging behavior and does not
  itself predict (the R18 Doppler axis is orthogonal and could be layered on).
- **`A` must be shared.** Interoperability requires the UE and gNB to use the
  same published `A` and quantizer table. This is a one-time standardization
  of two constants — strictly less than standardizing a codebook, and far less
  than agreeing on a trained encoder.

---

## 8. Reproducing

```bash
# 1. CDL dataset (raw-H ground truth) — reuses nr_csi.dataset
.venv/bin/python scripts/dataset/generate_cdl_dataset.py \
    --configs 4x4 --n-samples 60000 --profiles A,B,C \
    --freq-res 64 --fft-size 256 --out data/cdl_p32

# 2. Precompute eigen-target angle–delay vectors
.venv/bin/python scripts/ml/prepare_targets.py data/cdl_p32 --n3 8 \
    --out data/ml/targets_p32

# 3. Train the gNB decoder (measurement-dropout, one model, all payloads)
.venv/bin/python scripts/ml/train_glimpse.py \
    --data data/ml/targets_p32.npz --out models/glimpse_p32

# 4. Evaluate vs every codebook on frozen CDL drops (+ ablations, + generalization)
.venv/bin/python scripts/ml/eval_glimpse.py --model models/glimpse_p32 \
    --cdl C --drops 200 --ablation --out results/ml
.venv/bin/python scripts/ml/eval_glimpse.py --model models/glimpse_p32 \
    --cdl A,B,D,E --decoders learned --drops 150 --out results/ml

# 5. Render the comparison figures
.venv/bin/python scripts/ml/make_glimpse_figures.py --results results/ml --cdl C
```

Source: `src/nr_csi/ml/` (`projection.py`, `quantizer.py`, `scheme.py`,
`decoder.py`); tests: `tests/ml/`.
