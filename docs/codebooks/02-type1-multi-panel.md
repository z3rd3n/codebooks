# Chapter 2 — R15 Type I multi-panel codebook

* **Spec:** TS 38.214 §5.2.2.2.2 *"Type I Multi-Panel Codebook"*
  (clause heading verified at line 9251 of `specs/38214-i00.md`; the clause
  spans roughly lines 9251–9441).
* **Code:** [type1_multipanel.py](../../src/nr_csi/codebooks/type1_multipanel.py)
  — class `Type1MultiPanelCodebook`; validation in
  [validate.py](../../src/nr_csi/codebooks/validate.py) (`validate_type1_multipanel`);
  serialization in [serialize.py](../../src/nr_csi/codebooks/serialize.py)
  (`pack_type1_multipanel` / `unpack_type1_multipanel`).
* **Ranks:** 1–4.
* **Compression:** same DFT-beam + co-phasing idea as single-panel
  ([Chapter 1](01-type1-single-panel.md)), plus an extra **inter-panel
  co-phasing** because the array is physically split into $N_g$ panels.
* **Prereq:** Chapters [0](00-foundations.md) and [1](01-type1-single-panel.md).

When the gNB array is assembled from several panels ($N_g \in \{2,4\}$), the
panels are not phase-calibrated against each other. Multi-panel Type I keeps the
single-panel per-panel structure but adds **inter-panel co-phasing** to align the
panels, reported via a new wideband index $i_{1,4}$ and (in Mode 2) refined
per-subband inside $i_2$.

> **Spec/code note on this chapter's notation.** The Markdown export of the spec
> (`specs/38214-i00.md`, lines 9251–9441) has lost every equation and every table
> cell to the Pandoc image/MathML stripping (the cells render blank). The
> equations and table contents transcribed below are reconstructed from TS 38.214
> §5.2.2.2.2 and cross-checked against the implementation. Where the printed spec
> text *is* legible it is quoted verbatim.

---

## 1. Scope, ports, and the panel split

§5.2.2.2.2 applies to **8, 16, or 32 antenna ports** (`{3000,…,3007}`,
`{3000,…,3015}`, `{3000,…,3031}`) with higher-layer parameter
*codebookType* = `'typeI-MultiPanel'`. The number of CSI-RS ports is

$$P_{\text{CSI-RS}} = 2\,N_g\,N_1\,N_2,$$

i.e. two polarizations $\times$ $N_g$ panels $\times$ $N_1 N_2$ ports per
panel-and-polarization. In code, `AntennaConfig.P` returns exactly
$2 N_g N_1 N_2$ ([config.py](../../src/nr_csi/config.py)).

$(N_g, N_1, N_2)$ and the oversampling $(O_1, O_2)$ are configured by the higher
layer parameter *ng-n1-n2*. The supported tuples are fixed by
**Table 5.2.2.2.2-1**.

### Table 5.2.2.2.2-1 — Supported $(N_g, N_1, N_2)$ and $(O_1, O_2)$

| $P_{\text{CSI-RS}}$ | $(N_g, N_1, N_2)$ | $(O_1, O_2)$ |
|---|---|---|
| 8  | $(2,2,1)$ | $(4,1)$ |
| 16 | $(2,4,1)$ | $(4,1)$ |
| 16 | $(4,2,1)$ | $(4,1)$ |
| 16 | $(2,2,2)$ | $(4,4)$ |
| 32 | $(2,8,1)$ | $(4,1)$ |
| 32 | $(4,4,1)$ | $(4,1)$ |
| 32 | $(2,4,2)$ | $(4,4)$ |
| 32 | $(4,2,2)$ | $(4,4)$ |

This is reproduced exactly as the `SUPPORTED_NG_N1N2` dict in
[config.py](../../src/nr_csi/config.py):

```python
SUPPORTED_NG_N1N2 = {
    (2, 2, 1): (4, 1), (2, 4, 1): (4, 1), (4, 2, 1): (4, 1), (2, 2, 2): (4, 4),
    (2, 8, 1): (4, 1), (4, 4, 1): (4, 1), (2, 4, 2): (4, 4), (4, 2, 2): (4, 4),
}
```

Construct with `AntennaConfig.standard(N1, N2, Ng=...)` (strict mode validates the
$(O_1,O_2)$ pairing against this table).

> **codebookMode restriction by $N_g$ (verbatim from the spec, line 9263):**
> *"When [$N_g = 2$], codebookMode shall be set to either '1' or '2'. When
> [$N_g = 4$], codebookMode shall be set to '1'."* The class enforces this:
> `mode == 2` raises unless `Ng == 2`. So **Mode 2 exists only for the four
> $N_g = 2$ configurations**: $(2,2,1)$, $(2,4,1)$, $(2,2,2)$, $(2,8,1)$,
> $(2,4,2)$ — i.e. every $N_g=2$ row above.

The beam grid per panel is the usual oversampled DFT grid of size
$G_1 \times G_2 = (N_1 O_1) \times (N_2 O_2)$ (`AntennaConfig.n_beams`). A spatial
DFT beam $v_{l,m}$ of length $N_1 N_2$ is built by `dft.spatial_beam(a, l, m)`,
identical to single-panel.

---

## 2. PMI structure: $i_1 = \{i_{1,1}, i_{1,2}, i_{1,3}, i_{1,4}\}$ and $i_2$

Each PMI value corresponds to $(i_1, i_2)$ and an RI value, where

$$i_1 = \{i_{1,1},\ i_{1,2},\ i_{1,3},\ i_{1,4}\}.$$

* $i_{1,1}, i_{1,2}$ — **wideband** DFT-beam grid indices (one beam direction for
  the whole band), ranges $i_{1,1}\in\{0,\dots,O_1 N_1-1\}$,
  $i_{1,2}\in\{0,\dots,O_2 N_2-1\}$. As in single-panel, the UE *shall only use
  and shall not report $i_{1,2}$ if $N_2 = 1$* (spec line 9315).
* $i_{1,3}$ — selects the beam **offset** of the companion beam $k_1,k_2$ for
  ranks 2–4 (rank 1 has a single beam, no $i_{1,3}$).
* $i_{1,4}$ — **inter-panel co-phasing** (the new multi-panel index), wideband.
* $i_2$ — **per-subband** co-phasing (polarization, and in Mode 2 also a fine
  per-subband inter-panel refinement). Length $N_3$ (the number of PMI frequency
  units).

The dataclass:

```python
@dataclass
class Type1MPPMI:
    rank: int
    mode: int
    i11: int                 # i_{1,1} beam index dim 1
    i12: int                 # i_{1,2} beam index dim 2
    i14: tuple[int, ...]     # i_{1,4} inter-panel co-phasing (shape per mode)
    i2:  np.ndarray          # per-subband co-phasing (shape depends on mode)
    i13: int | None = None   # i_{1,3} ranks 2-4 beam offset
```

### The $i_{1,4}$ / $i_2$ index structure per mode (spec lines 9303–9309)

The spec defines $i_{1,4}$ and $i_2$ as vectors whose length depends on
codebookMode:

* **codebookMode = 1.** $i_{1,4} = [i_{1,4}^{(1)}\ \dots\ i_{1,4}^{(N_g-1)}]$
  — one QPSK panel phase per panel **other than panel 0** (panel 0 is the
  reference, phase $1$). Each $i_{1,4}^{(p)} \in \{0,1,2,3\}$ (2 bits, QPSK).
  $i_2$ is a single per-subband co-phase index $i_2^{(t)} = n_t$.

* **codebookMode = 2** (only $N_g = 2$). $i_{1,4} = [i_{1,4}^{(1)}\ i_{1,4}^{(2)}]$
  — **two** QPSK indices, each $\in \{0,1,2,3\}$. Per subband $t$, $i_2$ carries a
  **triple** $(n_0, n_1, n_2)$ where $n_0$ is the polarization co-phase and
  $n_1, n_2 \in \{0,1\}$ are finer per-subband panel phases combined with a fixed
  $\pm\pi/4$ construction (see §4).

Code: `_i14_shape()` returns $(N_g-1,)$ in Mode 1 and $(2,)$ in Mode 2;
`_i2_shape()` returns $(N_3,)$ in Mode 1 and $(N_3, 3)$ in Mode 2.

---

## 3. Co-phasing quantities and their ranges

The spec introduces several auxiliary quantities. With $\varphi_n = e^{j\pi n/2}$
the QPSK polarization co-phase and $a_p$ the panel phases, the **number of $i_2$
states** is rank- and mode-dependent. The implementation enumerates them in
`_i2_states(rank)`:

| Rank | Mode 1 ($n_0$ range) | Mode 2 $(n_0, n_1, n_2)$ |
|---|---|---|
| 1 | $n_0 \in \{0,1,2,3\}$ → **4 states** | $n_0\in\{0,1,2,3\},\ n_1,n_2\in\{0,1\}$ → **16 states** |
| 2–4 | $n_0 \in \{0,1\}$ → **2 states** | $n_0\in\{0,1\},\ n_1,n_2\in\{0,1\}$ → **8 states** |

```python
def _i2_states(self, rank):
    if self.mode == 1:
        return [(n,) for n in range(4 if rank == 1 else 2)]
    n0 = range(4 if rank == 1 else 2)
    return list(itertools.product(n0, range(2), range(2)))
```

So per subband, **Mode 1** spends 2 bits at rank 1 and 1 bit at ranks 2–4;
**Mode 2** spends 4 bits at rank 1 ($2 + 1 + 1$) and 3 bits at ranks 2–4
($1 + 1 + 1$). This matches `overhead_bits` (§7) and the serializer's per-state
widths `(2 if rank==1 else 1, 1, 1)`.

> **Spec/code consistency.** Validation (`validate_type1_multipanel`) enforces
> the ranges above: Mode 1 `i2 ∈ [0,3]` at rank 1 else `[0,1]`; Mode 2 the
> polarization column `i2[:,0] ∈ [0,3]` at rank 1 else `[0,1]`, and the panel
> columns `i2[:,1:] ∈ [0,1]` always. $i_{1,4}$ entries are checked `∈ [0,3]`.

---

## 4. Reconstruction (`precoder` → `_w_at` → `_base_vector`)

`precoder` validates the PMI then fills each subband with `_w_at(pmi, t)`. The
core is `_base_vector(l, m, i14, i2_state, family)`, which builds one
length-$P$ column for a given beam $(l,m)$, stacking polarizations within a panel
and then the panels.

Throughout, $v = v_{l,m}$ (length $N_1 N_2$), $\varphi_n = e^{j\pi n/2}$, and the
normalization is $1/\sqrt{P}$ per column (plus a further $1/\sqrt{\text{rank}}$ in
`_w_at`, so $\operatorname{tr}(W^H W) = 1$).

### Mode 1 (any $N_g$)

The panel phases are $a_0 = 1$ and $a_p = e^{j\pi\,i_{1,4}^{(p)}/2}$ for
$p = 1,\dots,N_g-1$ (QPSK). For each panel the column carries a top
(polarization-0) and bottom (polarization-1) block; the bottom block is rotated
by $\varphi_n$ and by a **family sign** $s \in \{+1,-1\}$ that distinguishes the
two orthogonal "layers" inside a beam:

$$\text{panel } p:\quad
\begin{bmatrix} a_p\, v \\ s\,\varphi_n\, a_p\, v \end{bmatrix},
\qquad s = +1\ (\text{family 1}),\ \ -1\ (\text{family 2}).$$

```python
phi_n = np.exp(1j * np.pi * i2[0] / 2)
panel_phases = [1, *(np.exp(1j * np.pi * p / 2) for p in i14)]
for phase in panel_phases:
    blocks.extend([phase * v, (+1 if family == 1 else -1) * phi_n * phase * v])
```

Schematically, a rank-1 Mode-1 precoder is

$$W^{(1)} = \tfrac{1}{\sqrt P}
\big[\, v;\ \varphi_n v;\ a_1 v;\ \varphi_n a_1 v;\ \dots;\ a_{N_g-1} v;\ \varphi_n a_{N_g-1} v \,\big]^{\!\top}.$$

### Mode 2 ($N_g = 2$ only)

Mode 2 fixes a $\pm\pi/4$ rotation and splits the inter-panel phase into a
wideband part ($i_{1,4}$) and a per-subband refinement ($n_1, n_2$ inside $i_2$).
With $\varphi_{n_0} = e^{j\pi n_0/2}$,

$$a_{p,1} = e^{j\pi/4}e^{j\pi\,i_{1,4}^{(1)}/2},\quad
  a_{p,2} = e^{j\pi/4}e^{j\pi\,i_{1,4}^{(2)}/2},$$
$$b_{n,1} = e^{-j\pi/4}e^{j\pi n_1/2},\quad
  b_{n,2} = e^{-j\pi/4}e^{j\pi n_2/2},$$

and the column (for the two-panel, two-polarization stack) is

$$\big[\, v;\ s\,\varphi_{n_0}\, v;\ a_{p,1} b_{n,1}\, v;\ s\, a_{p,2} b_{n,2}\, v \,\big],
\qquad s = \pm 1 \ (\text{family}).$$

```python
n0, n1, n2 = i2
phi_n0 = np.exp(1j * np.pi * n0 / 2)
ap1 = np.exp(1j*np.pi/4) * np.exp(1j*np.pi*i14[0]/2)
ap2 = np.exp(1j*np.pi/4) * np.exp(1j*np.pi*i14[1]/2)
bn1 = np.exp(-1j*np.pi/4) * np.exp(1j*np.pi*n1/2)
bn2 = np.exp(-1j*np.pi/4) * np.exp(1j*np.pi*n2/2)
sign = +1 if family == 1 else -1
blocks = [v, sign*phi_n0*v, ap1*bn1*v, sign*ap2*bn2*v]
```

The $n_1, n_2$ (carried per subband in the $(N_3, 3)$ $i_2$) give finer per-subband
panel alignment than Mode 1's wideband-only $i_{1,4}$.

### Rank stacking (`_w_at`)

Build the base beam $b_1 = (i_{1,1}, i_{1,2})$ and a companion
$b_2 = (i_{1,1}+k_1,\, i_{1,2}+k_2)$, where $(k_1, k_2)$ is the $i_{1,3}$-th
offset (§5). With superscript = family:

| Rank | Columns | Spec codebook table |
|---|---|---|
| 1 | $[\,b_1^{(1)}\,]$ | Table 5.2.2.2.2-3 |
| 2 | $[\,b_1^{(1)},\ b_2^{(2)}\,]$ | Table 5.2.2.2.2-4 |
| 3 | $[\,b_1^{(1)},\ b_2^{(1)},\ b_1^{(2)}\,]$ | Table 5.2.2.2.2-5 |
| 4 | $[\,b_1^{(1)},\ b_2^{(1)},\ b_1^{(2)},\ b_2^{(2)}\,]$ | Table 5.2.2.2.2-6 |

All columns scaled by $1/\sqrt P$ (in `_base_vector`) and a further
$1/\sqrt{\text{rank}}$ (in `_w_at`). The two families (sign $\pm$ on the
polarization/panel block) generate orthogonal columns at higher rank — exactly
the $\pm\varphi$ trick from single-panel, here extended across panels.

---

## 5. Beam-offset tables: $i_{1,3}$ and $(k_1, k_2)$

Rank 1 has no companion beam ($i_{1,3}$ absent; `_offset` returns $(0,0)$).

**Rank 2** reuses the single-panel rank-2 offset table — the spec routes 2-layer
multi-panel reporting through *the same* mapping
$i_2 \to i_{1,3}$ as single-panel, **Table 5.2.2.2.1-3** (spec line 9311). In
code this is `i13_offsets(N1,N2,O1,O2)` from [type1.py](../../src/nr_csi/codebooks/type1.py):

```python
N1>N2>1     -> [(0,0),(O1,0),(0,O2),(2O1,0)]
N1==N2      -> [(0,0),(O1,0),(0,O2),(O1,O2)]
N1>2,N2==1  -> [(0,0),(O1,0),(2O1,0),(3O1,0)]
N1==2,N2==1 -> [(0,0),(O1,0)]
```

**Ranks 3–4** use the multi-panel-specific **Table 5.2.2.2.2-2** ("Mapping of
$i_{1,3}$ to $k_1$ and $k_2$ for 3-layer and 4-layer CSI reporting"). The clause
quoting *"is given in Table 5.2.2.2.1-3"* for rank 3/4 (spec line 9312) actually
points at Table **5.2.2.2.2-2**; the implementation correctly uses the
multi-panel table:

```python
def i13_offsets_multipanel_rank34(N1, N2, O1, O2):
    table = {
        (2, 1): [(O1, 0)],
        (4, 1): [(O1, 0), (2*O1, 0), (3*O1, 0)],
        (8, 1): [(O1, 0), (2*O1, 0), (3*O1, 0), (4*O1, 0)],
        (2, 2): [(O1, 0), (0, O2), (O1, O2)],
        (4, 2): [(O1, 0), (0, O2), (O1, O2), (2*O1, 0)],
    }
    return table[(N1, N2)]
```

The five $(N_1,N_2)$ keys here are exactly the per-panel shapes of the eight
Table 5.2.2.2.2-1 configurations (e.g. $(8,1)$ from the 32-port $(2,8,1)$ config,
$(4,2)$ from $(2,4,2)$/$(4,2,2)$). The number of $i_{1,3}$ values for a given
config is the length of the returned list:

| $(N_1, N_2)$ | rank-2 ($\#i_{1,3}$) | rank-3/4 ($\#i_{1,3}$) |
|---|---|---|
| $(2,1)$ | 2 | 1 |
| $(4,1)$ | 4 | 3 |
| $(8,1)$ | 4 | 4 |
| $(2,2)$ | 4 | 3 |
| $(4,2)$ | 4 | 4 |

`_n_i13(rank)` returns 1 for rank 1, `len(i13_offsets(...))` for rank 2, and
`len(i13_offsets_multipanel_rank34(...))` for ranks 3–4.

---

## 6. PMI selection (UE side, `select`)

Like single-panel, this is a **rate-maximizing enumeration**, but the candidate
space now includes the inter-panel phases $i_{1,4}$:

1. **Guard rails.** Reject `rank` outside $[1,4]$ or prohibited by
   `rank_restriction`; take the last slot $H_t = H[-1]$, shape $(N_3, N_r, P)$.
2. **Enumerate** $(i_{1,1}, i_{1,2}, i_{1,3}, i_{1,4})$ over their valid ranges,
   dropping beam-restricted candidates (`_candidate_allowed` checks both the base
   beam and, for rank > 1, the companion beam). $i_{1,4}$ ranges over
   $4^{N_g-1}$ tuples in Mode 1 (and $4^2 = 16$ in Mode 2, where `_i14_shape()`
   is fixed to length 2 — see §9).
3. **Score** each candidate for every $i_2$ state in `_i2_states(rank)` by
   per-subband log-det rate (`_candidate_rate` →
   $\log_2\det(I_v + \rho\,(H_tW)^H H_t W)$), processed in **batches of 128**
   to bound memory (the candidate count is large once panel phases are included).
4. **Pick** the best wideband $(i_{1,1},i_{1,2},i_{1,3},i_{1,4})$ by
   summed best-per-subband rate, then read off the per-subband $i_2$ states.

The metric is identical to single-panel — `slogdet(I + ρ·gram)` — so the same
nominal `selection_snr_db` governs the choice ($\rho = 10^{\text{SNR}/10}$, UE
side only, not part of the report).

---

## 7. Feedback overhead (`overhead_bits`)

```python
bits = {
  "i11": ceil(log2(G1)),         # i_{1,1}
  "i14": 2 * len(i14),           # 2 bits per reported inter-panel phase
}
if N2 > 1:  bits["i12"] = ceil(log2(G2))   # dropped when N2 == 1
if rank>1:  bits["i13"] = ceil(log2(N_i13))
# i2 is per subband:
mode 1:  bits["i2"] = N3 * (2 if rank==1 else 1)
mode 2:  bits["i2"] = N3 * (4 if rank==1 else 3)
```

Reading this:

* **$i_{1,4}$** adds $2(N_g-1)$ bits in Mode 1 (so 2 bits for $N_g=2$, 6 bits for
  $N_g=4$) or **4 bits** in Mode 2 — the price of inter-panel calibration over a
  single-panel report.
* **$i_2$** is the per-subband term and dominates for large $N_3$; Mode 2 costs
  more per subband (it carries the extra $n_1, n_2$) but buys finer per-subband
  panel alignment.
* $i_{1,2}$ is omitted entirely when $N_2 = 1$ (like single-panel), in both the
  bit count and the serializer (`pack_type1_multipanel` only writes $i_{1,2}$ if
  $N_2 > 1$).

**Worked example** — $(N_g,N_1,N_2) = (4,4,1)$ → $P = 32$, $(O_1,O_2)=(4,1)$,
$G_1 = 16$, $G_2 = 1$, $N_3 = 10$, Mode 1, rank 1:
$i_{1,1} = \lceil\log_2 16\rceil = 4$; $i_{1,2}$ dropped ($N_2=1$);
$i_{1,4} = 2(4-1) = 6$; $i_{1,3}$ absent (rank 1);
$i_2 = 10\cdot 2 = 20$ → **30 bits** total.

---

## 8. codebookSubsetRestriction and ri-Restriction (multi-panel)

> The implemented restriction interfaces mirror single-panel's, but the spec's
> *exact* multi-panel bitmap semantics differ — see the flags below.

* **Beam subset restriction.** The code accepts a `beam_restriction` bitmap of
  $G_1 G_2$ bits indexed `(m % G2) + G2*(l % G1)` (`_beam_allowed`), and `select`
  refuses any beam (base or companion) whose bit is 0. This matches the *intent*
  of *codebookSubsetRestriction* for the spatial beams.

  > 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** The spec's
  > *ng-n1-n2* bit sequence (spec lines 9266–9270) is the *combined* config +
  > subset-restriction bitmap whose bit count *"is given by [a formula]"* and
  > whose bits are *"associated with all precoders based on the quantity"* (beam
  > group). The codebase models the subset restriction as a plain $G_1 G_2$ beam
  > bitmap and does **not** reproduce the spec's exact *ng-n1-n2* group-bitmap
  > construction (the per-group association rule). The functional effect (forbid
  > selected beams) is present; the precise bit layout is approximated.

  > 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** There is no $i_2$
  > co-phasing subset restriction. Single-panel has
  > *typeI-SinglePanel-codebookSubsetRestriction-i2* (a 16-bit map over $i_2$
  > values, used under *reportQuantity* = `'cri-RI-i1-CQI'`; spec lines
  > 8987–8997). No analogous per-$i_2$ (or per-$i_{1,4}$ panel-phase) restriction
  > is offered for multi-panel.

* **Rank restriction.** `rank_restriction` is a **4-bit** bitmap $[r_1,r_2,r_3,r_4]$
  ($r_1$ = LSB, rank-1 layer). `select` raises if the requested rank's bit is 0.
  This matches the spec *ri-Restriction* semantics (spec lines 9271–9273):
  *"When [the bit] is zero, PMI and RI reporting are not allowed to correspond to
  any precoder associated with [that number of] layers."* Multi-panel supports
  ranks 1–4 only, so 4 bits is correct (vs single-panel's 8).

---

## 9. Known limitations / deviations from the spec

* **Mode-2 $i_{1,4}$ enumeration in `select` is exhaustive over $4^2 = 16$**
  tuples (`i14_values = product(range(4), repeat=_i14_shape()[0])` with
  `_i14_shape() == (2,)` in Mode 2). This is consistent with reconstruction,
  which reads two `i14` entries.

* **Per-panel CSI-RS layout / `Ng` interpretation.** The codebook assumes the
  panel-major port ordering used by `_base_vector` (polarizations stacked within
  a panel, panels stacked outermost). The spec's $P$-port ordering is the
  `{3000,…}` enumeration; the mapping is implicit in the implementation and not
  separately exposed.

* **CQI sub-band / wideband split, port indication, and `cri-RI-i1-CQI`
  reporting mode** are out of scope of this codebook class (the class produces a
  precoder per PMI frequency unit only).

  > 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** The
  > *cri-RI-i1-CQI* report quantity (wideband $i_1$ only, with the
  > random-$i_2$ CQI rule) is referenced by the spec for restriction purposes
  > but is not a supported reporting mode here.

---

## 10. Relationship to single-panel

Set $N_g = 1$ and multi-panel *would* collapse to single-panel — but the class
forbids it (`Ng in {2,4}`), since single-panel is its own clause (§5.2.2.2.1) and
supports ranks up to 8. Conceptually: **multi-panel = single-panel per panel + an
inter-panel co-phasing layer** ($i_{1,4}$ wideband, plus a per-subband refinement
$n_1,n_2$ folded into $i_2$ in Mode 2). Everything else — DFT beams, the
$\pm\varphi$ family trick, the rank-2 offset table (5.2.2.2.1-3), the rate-based
enumeration — is shared.

---

**Next:** [Chapter 3 — Type II R15](03-type2-r15.md). Type II abandons the
"single direction" model entirely and reports a *linear combination* of $L$ beams
with per-subband complex coefficients — the foundation of every later release.
