# Chapter 1 — R15 Type I single-panel codebook

* **Spec:** TS 38.214 §5.2.2.2.1 ("Type I Single-Panel Codebook"), Release-18
  (`38214-i00`). Verified: the clause heading sits at line 8918 of
  [specs/38214-i00.md](../../specs/38214-i00.md) and is titled exactly
  *"Type I Single-Panel Codebook"*. The supporting tables are
  5.2.2.2.1-1 … 5.2.2.2.1-12.
* **Code:** [type1.py](../../src/nr_csi/codebooks/type1.py) — class `Type1Codebook`;
  validation in [validate.py](../../src/nr_csi/codebooks/validate.py)
  (`validate_type1`); bit-packing in
  [serialize.py](../../src/nr_csi/codebooks/serialize.py)
  (`pack_type1`/`unpack_type1`).
* **Ranks:** 1–8 (subject to $P$).
* **Compression:** one (or a few) DFT beam(s) + QPSK co-phasing. Scalar
  coefficients only — no per-coefficient amplitude.
* **Prereq:** [Chapter 0 — Foundations](00-foundations.md).

Type I is the cheapest codebook and the structural template for everything else.
The UE reports essentially "which beam, and how to combine the two polarizations."
There is **one beam direction shared across all subbands** ($i_1$ is wideband) and
a **per-subband co-phasing** ($i_2$).

This chapter is a spec-faithful reference for the *whole* clause, so it documents
several standardized features the codebase does not implement. Those are flagged
with the marker below; everything unmarked is implemented.

> 🚩 **Marker convention.** A whole feature standardized but absent from this
> codebase is introduced with
> "🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.**"; a single missing
> option is tagged inline with **[not implemented]**.

---

## 1. The precoder structure at a glance

For a dual-polarized array the precoder is built from one (or a few) length-$N_1N_2$
DFT beam(s) $v_{l,m}$ and a co-phasing scalar $\varphi_n = e^{j\pi n/2}$ (QPSK)
between the two polarization halves. Rank 1 is the canonical case:

$$W^{(1)}_{l,m,n} = \frac{1}{\sqrt{P}} \begin{bmatrix} v_{l,m} \\ \varphi_n\, v_{l,m} \end{bmatrix}.$$

The top block drives polarization 0, the bottom block polarization 1, rotated by
$\varphi_n$. Higher ranks stack more such columns built from the same beam plus
beam *offsets*, with sign/phase patterns chosen to keep the columns orthonormal.
The $1/\sqrt{P}$ (and $1/\sqrt{vP}$ at higher rank) makes $\operatorname{tr}(W^HW)=1$.

The DFT beam $v_{l,m} = u_m \otimes \tilde u_l$ uses the oversampled steering
vectors of [Chapter 0 §5](00-foundations.md#5-dft-bases-spatial-frequency-temporal):

$$\tilde u_l[k] = e^{\,j 2\pi l k /(O_1 N_1)},\quad
  u_m[k] = e^{\,j 2\pi m k /(O_2 N_2)},$$

so the spec's $\varphi_n = e^{j\pi n/2}$ and the beam pair $(l,m)$ are the only
free quantities at rank 1.

---

## 2. The 2-antenna-port special case (Table 5.2.2.2.1-1)

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** The 2-port codebook
> (`{3000,3001}`, $P=2$) and its dedicated restriction bitmap.

For exactly **2 CSI-RS ports** the spec does *not* use the DFT-beam machinery
below. Each PMI value is a single codebook index into **Table 5.2.2.2.1-1**, the
small fixed Householder-style codebook shared with LTE:

| Codebook index $i_2$ | 1 layer $W^{(1)}$ (unnormalized) | 2 layers $W^{(2)}$ (unnormalized) |
|---|---|---|
| 0 | $\begin{bmatrix}1\\1\end{bmatrix}$ | $\tfrac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$ |
| 1 | $\begin{bmatrix}1\\-1\end{bmatrix}$ | $\tfrac{1}{\sqrt2}\begin{bmatrix}1&1\\j&-j\end{bmatrix}$ |
| 2 | $\begin{bmatrix}1\\j\end{bmatrix}$ | — |
| 3 | $\begin{bmatrix}1\\-j\end{bmatrix}$ | — |

(rank-1 vectors carry the $1/\sqrt2$ norm factor). The associated restriction
bitmap is **`twoTX-CodebookSubsetRestriction`**: bits 0–3 gate the four rank-1
indices, bits 4–5 the two rank-2 indices; a zero bit forbids that precoder. None
of this is in the code — `Type1Codebook` requires the UPA path (it computes
$P = 2\,N_1 N_2$ from `AntennaConfig` and there is no $P=2$ table). The smallest
configuration the code accepts is $(N_1,N_2)=(2,1)$, i.e. $P=4$.

---

## 3. Configuration

```python
Type1Codebook(
    antenna,                 # AntennaConfig with Ng == 1
    N3 = 1,                  # number of PMI frequency units
    mode = 1,                # codebookMode 1 or 2
    beam_restriction = None, # G1*G2 bitmap of allowed beams (subset restriction)
    rank_restriction = None, # 8-bit [r0..r7]: which ranks are allowed
    selection_snr_db = 10.0, # nominal SNR the UE uses to score candidates
)
```

* **`mode` (codebookMode).** Mode 1 reports the beam on the coarse oversampled
  grid. Mode 2 reports a coarse beam *plus* a fine offset folded into $i_2$,
  giving finer angular resolution at the cost of more $i_2$ states. Details in
  §6. Mode 2 is only defined for $P \ge 4$ at ranks 1–2; ranks ≥ 3 use the single
  "codebookMode = 1‑2" construction regardless of the configured mode.
* **`beam_restriction`** — `typeI-SinglePanel-codebookSubsetRestriction`, the
  `n1-n2` beam-group bitmap, $G_1 G_2$ bits (`_beam_allowed(l,m)` indexes it as
  `(m % G2) + G2*(l % G1)`). Prohibited beams are never selected and rejected at
  reconstruction. See §10 for the *exact* spec mechanism, which is richer than a
  flat per-beam mask.
* **`rank_restriction`** — `typeI-SinglePanel-ri-Restriction`, 8 bits
  $r_7\ldots r_0$ ($r_0$ = LSB); when $r_{v-1}=0$, RI/PMI for $v$ layers is
  forbidden. `select` refuses a prohibited rank.
* **`selection_snr_db`** sets `selection_rho` $= 10^{\text{SNR}/10}$, the linear
  SNR at which candidate precoders are scored by log-det rate (UE side only; not
  part of the report).

---

## 4. The PMI

```python
@dataclass
class Type1PMI:
    rank: int
    mode: int
    i11: int                 # beam index, dimension 1 (horizontal)
    i12: int                 # beam index, dimension 2 (vertical)
    i2:  np.ndarray          # (N3,) co-phasing index per frequency unit
    i13: int | None = None   # ranks 2-4: beam-offset selector
```

The spec composes the wideband part into a single index $i_1$. For $P>2$ and
ranks $v \le 4$ the PMI is the triple $(i_{1,1}, i_{1,2}, i_2)$ (a fourth index
$i_{1,3}$ appears for $v\in\{2,3,4\}$); for $v\in\{5,6,7,8\}$ it is the pair
$(i_{1,1}, i_{1,2}, i_2)$ with no $i_{1,3}$. Mapping to the code:

* $i_{1,1}, i_{1,2}$ — **wideband** beam grid indices (one beam direction for the
  whole band). Vertical index $i_{1,2}$ is **only reported when $N_2 > 1$**; the
  UE shall use $i_{1,2}=0$ and not report it when $N_2 = 1$.
* $i_{1,3}$ — selects the *offset* of the companion beam(s) for ranks 2–4
  (the second beam is at $(l+k_1, m+k_2)$, or a $\theta$ rotation for $P\ge16$).
* $i_2$ — **per-subband** co-phasing (and, in Mode 2 / low rank, the fine beam
  offset). Length $N_3$.

> The spec also defines $i_2$ as *wideband* when the report is configured for
> wideband PMI only; the codebase always carries a length-$N_3$ $i_2$ array and
> sets $N_3=1$ for the wideband case, which is equivalent.

---

## 5. Beam grids, supported $(N_1,N_2)/(O_1,O_2)$, and index ranges

### 5.1 Supported configurations (Table 5.2.2.2.1-2)

$N_1,N_2$ come from RRC `n1-n2`; $O_1,O_2$ are fixed per row. From
`SUPPORTED_N1N2` in [config.py](../../src/nr_csi/config.py):

| Ports $P$ | $(N_1,N_2)$ | $(O_1,O_2)$ |
|---|---|---|
| 4  | $(2,1)$ | $(4,1)$ |
| 8  | $(2,2)$ | $(4,4)$ |
| 8  | $(4,1)$ | $(4,1)$ |
| 12 | $(3,2)$ | $(4,4)$ |
| 12 | $(6,1)$ | $(4,1)$ |
| 16 | $(4,2)$ | $(4,4)$ |
| 16 | $(8,1)$ | $(4,1)$ |
| 24 | $(4,3)$ | $(4,4)$ |
| 24 | $(6,2)$ | $(4,4)$ |
| 24 | $(12,1)$ | $(4,1)$ |
| 32 | $(4,4)$ | $(4,4)$ |
| 32 | $(8,2)$ | $(4,4)$ |
| 32 | $(16,1)$ | $(4,1)$ |

$P = 2 N_1 N_2$; the oversampled grid is $G_1 \times G_2 = (N_1 O_1)\times(N_2 O_2)$.
1-D arrays ($N_2=1$) always have $O_2=1$. (The R19 large arrays 48/64/128 ports
in `SUPPORTED_N1N2_R19` belong to clause 5.2.2.2.1a, not this clause.)

### 5.2 How many values each index carries (`_n_i11`, `_n_i12`, `_n_i13`, `_n_i2`)

| Index | Count (typical) | Reduced cases |
|---|---|---|
| $i_{1,1}$ | $G_1 = N_1 O_1$ | $G_1/2$ for (mode 2, rank 1–2), (rank 3–4 with $P\ge16$), (rank 7–8, $N_1{=}4,N_2{=}1$) |
| $i_{1,2}$ | $G_2 = N_2 O_2$ | $G_2/2$ for (mode 2, rank 1–2) and (rank 7–8, $N_2{=}2,N_1{>}2$) |
| $i_{1,3}$ | 1 (rank 1), else table size | rank 2: `i13_offsets`; rank 3–4: 4 if $P\ge16$, else `i13_offsets_rank34` |
| $i_2$ | see below | |

The $G_1/2$, $G_2/2$ reductions come straight from the spec: where a beam offset
is folded into $i_2$ (Mode 2 low rank) or is a fixed unit step (some high-rank
constructions), the wideband index only needs to address every *other* grid
point. The high-rank halving of $i_{1,1}$ for $(N_1,N_2)=(4,1)$ ranks 7–8 and of
$i_{1,2}$ for $N_2{=}2,N_1{>}2$ ranks 7–8 matches the layer matrices in §7.4.

**Co-phasing states $N_{i_2}$** (`_n_i2`):

| Rank | Mode 1 | Mode 2 |
|---|---|---|
| 1 | 4 | 16 |
| 2 | 2 | 8 |
| 3–8 | 2 | 2 |

In Mode 1 these are pure co-phasing states: $\varphi_n = e^{j\pi n/2}$ (QPSK,
$n\in\{0,1,2,3\}$) for rank 1, and the BPSK-like $\{1,j\}$ ($n\in\{0,1\}$) for
higher ranks. In Mode 2 at rank 1–2 the count is multiplied by 4 because $i_2$
also carries one of four fine beam offsets (§6).

**Bit-widths** (`overhead_bits`, all $\lceil\log_2(\cdot)\rceil$):

| Field | Width | Reported when |
|---|---|---|
| $i_{1,1}$ | $\lceil\log_2 N_{i_{1,1}}\rceil$ | always |
| $i_{1,2}$ | $\lceil\log_2 N_{i_{1,2}}\rceil$ | $N_2 > 1$ |
| $i_{1,3}$ | $\lceil\log_2 N_{i_{1,3}}\rceil$ | rank $\in\{2,3,4\}$ |
| $i_2$ | $N_3\cdot\lceil\log_2 N_{i_2}\rceil$ | always (per subband) |

---

## 6. Beam-and-phase decoding: Mode 1 vs Mode 2

`_beam_and_phase(pmi, t)` turns $(i_{1,1}, i_{1,2}, i_2[t])$ into the actual
$(l, m, n)$ used to build column(s):

* **Mode 1 (or rank ≥ 3):** the beam is exactly $(i_{1,1}, i_{1,2})$ and the
  co-phase is $n = i_2[t]$.
* **Mode 2, rank 1–2:** $i_2$ is split as

  ```
  offset_index = i2 // n_phases     # 0..3  -> (k1', k2')
  n            = i2 %  n_phases      # co-phase
  ```
  with `n_phases = 4` (rank 1) or `2` (rank 2). The beam becomes

  $$l = 2\,i_{1,1} + k_1',\qquad m = 2\,i_{1,2} + k_2',$$

  where the four offsets are $\{(0,0),(1,0),(0,1),(1,1)\}$ when $N_2>1$, else
  $\{(0,0),(1,0),(2,0),(3,0)\}$. So Mode 2 reports a *coarse* beam in $i_1$
  (hence the $G_1/2,\,G_2/2$ index counts) and refines it per subband through
  $i_2$ — strictly more angular resolution than Mode 1, at the cost of wider
  $i_2$.

This matches the spec's Table 5.2.2.2.1-5/-6 Mode-2 blocks, where the 16 (rank 1)
or 8 (rank 2) $i_2$ states index a $4\times4$ / $4\times2$ grid of
(beam-offset, co-phase) pairs.

---

## 7. Reconstruction by rank (gNB side, `precoder`)

`precoder` validates the PMI (`validate_type1`) then fills
`W[0, t] = _w_at(pmi, t)` for each subband $t$. `_w_at` dispatches on rank.
Throughout, $v_{l,m}$ is `dft.spatial_beam` and $\varphi_n = e^{j\pi n/2}$,
$\theta_p = e^{j\pi p/4}$.

### 7.1 Rank 1 — `_w_rank1` (Table 5.2.2.2.1-5)
$$W^{(1)} = \frac{1}{\sqrt{P}}\begin{bmatrix} v_{l,m}\\ \varphi_n v_{l,m}\end{bmatrix}.$$

### 7.2 Rank 2 — `_w_rank2` (Table 5.2.2.2.1-6, offsets from Table 5.2.2.2.1-3)
Two beams $v_1 = v_{l,m}$ and $v_2 = v_{l+k_1,\,m+k_2}$, where $(k_1,k_2)$ is the
$i_{1,3}$-th entry of `i13_offsets(N1,N2,O1,O2)`:

$$W^{(2)} = \frac{1}{\sqrt{2P}}
\begin{bmatrix} v_1 & v_2 \\ \varphi_n v_1 & -\varphi_n v_2\end{bmatrix}.$$

The opposite signs on the two columns' bottom blocks make the columns orthogonal.
The $i_{1,3}\!\to\!(k_1,k_2)$ table (Table 5.2.2.2.1-3) depends on the array
aspect ratio:

| $(N_1,N_2)$ regime | $i_{1,3}=0$ | $1$ | $2$ | $3$ |
|---|---|---|---|---|
| $N_1>N_2>1$ | $(0,0)$ | $(O_1,0)$ | $(0,O_2)$ | $(2O_1,0)$ |
| $N_1=N_2$   | $(0,0)$ | $(O_1,0)$ | $(0,O_2)$ | $(O_1,O_2)$ |
| $N_1>2,\,N_2=1$ | $(0,0)$ | $(O_1,0)$ | $(2O_1,0)$ | $(3O_1,0)$ |
| $N_1=2,\,N_2=1$ | $(0,0)$ | $(O_1,0)$ | — | — |

So $i_{1,3}$ has 4 values except for $(2,1)$ where it has 2.

### 7.3 Ranks 3–4

Two distinct standardized constructions, switched on port count:

**Small arrays ($P < 16$) — `_w_rank34_small` (Tables 5.2.2.2.1-7/-8, offsets from Table 5.2.2.2.1-4).**
A single beam $v_1$ and a companion $v_2 = v_{l+k_1,m+k_2}$ at the rank-3/4 offset.
The four/three columns alternate $\pm\varphi_n$ between the polarization halves:

$$\text{columns} = \tfrac{1}{\sqrt{vP}}\big[\,[v_1;\varphi v_1],\ [v_2;\varphi v_2],\ [v_1;-\varphi v_1]\,(,\ [v_2;-\varphi v_2])\,\big].$$

The rank-3/4 offset table `i13_offsets_rank34` (Table 5.2.2.2.1-4), $(k_1,k_2)$:

| $(N_1,N_2)$ | $i_{1,3}=0$ | $1$ | $2$ | $3$ |
|---|---|---|---|---|
| $(2,1)$ | $(O_1,0)$ | — | — | — |
| $(4,1)$ | $(O_1,0)$ | $(2O_1,0)$ | $(3O_1,0)$ | — |
| $(6,1)$ | $(O_1,0)$ | $(2O_1,0)$ | $(3O_1,0)$ | $(4O_1,0)$ |
| $(2,2)$ | $(O_1,0)$ | $(0,O_2)$ | $(O_1,O_2)$ | — |
| $(3,2)$ | $(O_1,0)$ | $(0,O_2)$ | $(O_1,O_2)$ | $(2O_1,0)$ |

**Large arrays ($P \ge 16$) — `_w_rank34_large` (Tables 5.2.2.2.1-7/-8, $P\ge16$ branch).**
Here the construction folds the horizontal dimension in half
(`steering(N1//2, O1, l)`), so the half-beam is
$\tilde v = \tilde u^{(N_1/2)}_l \otimes u^{(N_2)}_m$, and uses **two** rotation
scalars: $\varphi_n = e^{j\pi n/2}$ ($n=i_2\in\{0,1\}$) *and* $\theta_p = e^{j\pi p/4}$
where $p = i_{1,3}\in\{0,1,2,3\}$. The columns are the half-beam scaled by the
fixed coefficient matrices (transcribed from the spec, $v=3$ and $v=4$):

$$
\Theta_3 = \begin{bmatrix}
1 & 1 & 1\\
\theta & -\theta & \theta\\
\varphi & \varphi & -\varphi\\
\varphi\theta & -\varphi\theta & -\varphi\theta
\end{bmatrix},\qquad
\Theta_4 = \begin{bmatrix}
1 & 1 & 1 & 1\\
\theta & -\theta & \theta & -\theta\\
\varphi & \varphi & -\varphi & -\varphi\\
\varphi\theta & -\varphi\theta & -\varphi\theta & \varphi\theta
\end{bmatrix},
$$

with $W^{(v)} = \tfrac{1}{\sqrt{vP}}\,(\Theta_v \otimes \tilde v)$ stacking the
four $P/2$-length blocks. This is why $i_{1,1}$ halves to $G_1/2$ and $i_{1,3}$
becomes a 4-valued $\theta$ index for $P\ge16$.

### 7.4 Ranks 5–8 — `_w_rank58` + `_fixed_beams` (Tables 5.2.2.2.1-9 … -12)
A fixed set of 3 (rank 5–6) or 4 (rank 7–8) beams at standard offsets, combined
with hard-coded sign/co-phase patterns (the `top`/`bottom` lists; only $i_2$,
i.e. $n\in\{0,1\}$, is per-subband and there is no $i_{1,3}$). The beam sets
(`_fixed_beams`):

* **Rank 5–6:** $N_2>1\Rightarrow\{(l,m),(l{+}O_1,m),(l{+}O_1,m{+}O_2)\}$;
  $N_2=1\Rightarrow\{(l{+}qO_1,0):q=0,1,2\}$.
* **Rank 7–8:** $N_2=1\Rightarrow\{(l{+}qO_1,0):q=0,1,2,3\}$; else
  $\{(l,m),(l{+}O_1,m),(l,m{+}O_2),(l{+}O_1,m{+}O_2)\}$.

The per-rank column patterns (each column is $[\,\text{top};\,\text{bottom}\,]/\sqrt{vP}$,
$\varphi = \varphi_n$, $b_i$ = $i$-th beam):

* **Rank 5:** top $[b_0,b_0,b_1,b_1,b_2]$, bottom $[\varphi b_0,-\varphi b_0,b_1,-b_1,b_2]$.
* **Rank 6:** top $[b_0,b_0,b_1,b_1,b_2,b_2]$, bottom $[\varphi b_0,-\varphi b_0,\varphi b_1,-\varphi b_1,b_2,-b_2]$.
* **Rank 7:** top $[b_0,b_0,b_1,b_2,b_2,b_3,b_3]$, bottom $[\varphi b_0,-\varphi b_0,\varphi b_1,b_2,-b_2,b_3,-b_3]$.
* **Rank 8:** top $[b_0,b_0,b_1,b_1,b_2,b_2,b_3,b_3]$, bottom $[\varphi b_0,-\varphi b_0,\varphi b_1,-\varphi b_1,b_2,-b_2,b_3,-b_3]$.

> All higher-rank matrix patterns are transcribed directly from TS 38.214
> §5.2.2.2.1 (the codebase's source paper omits these tables) and pinned by
> closed-form orthonormality tests. The spec's Tables 5.2.2.2.1-11/-12 also halve
> $i_{1,1}$ (for $(4,1)$) or $i_{1,2}$ (for $N_2{=}2,N_1{>}2$) — captured in
> `_n_i11`/`_n_i12` (§5.2).

---

## 8. PMI selection (UE side, `select`)

`select(H, rank)` ([type1.py](../../src/nr_csi/codebooks/type1.py)) does an
**exhaustive rate-maximizing search**, unlike the energy-based Type II procedure.
`select` is *not* standardized (Chapter 0 §1) — only `precoder` is — so this is
one defensible UE procedure:

1. **Guard rails.** Reject `rank` outside $[1, \min(8,P)]$ or prohibited by
   `rank_restriction`. Take the last slot $H_t = H[-1]$, shape $(N_3, N_r, P)$.

2. **Enumerate wideband candidates** $(i_{1,1}, i_{1,2}, i_{1,3})$ over their valid
   ranges, and for each, build the precoder `Wc[ci, i2]` for *every* co-phasing
   state $i_2 \in \{0..N_{i_2}{-}1\}$. `_candidate_allowed` drops candidates whose
   beams violate `beam_restriction`.

3. **Score by per-subband log-det rate.** With $HW$ over all subbands,

   $$\text{rate}_{c,i_2,t} = \log_2 \det\!\big(I_v + \rho\, (H_tW)^H(H_tW)\big),$$

   $\rho =$ `selection_rho`. Disallowed $(c,i_2)$ get $-\infty$.

4. **Pick the wideband index**, then the per-subband co-phase:
   * `metric[c] = sum_t max_{i2} rate[c, i2, t]` — the best achievable rate if
     each subband may pick its own $i_2$. Choose $c^\star = \arg\max$.
   * `i2[t] = argmax_{i2} rate[c*, i2, t]` — independently per subband.

This cleanly separates the **wideband** beam choice ($i_1$, one for the band) from
the **per-subband** co-phasing ($i_2$), exactly matching the report structure.

---

## 9. Feedback overhead (`overhead_bits`)

```python
bits = {
  "i11": ceil(log2(N_i11)),
  "i2":  N3 * ceil(log2(N_i2)),
}
if N2 > 1:            bits["i12"] = ceil(log2(N_i12))
if rank in (2,3,4):   bits["i13"] = ceil(log2(N_i13))
```

Key points:

* $i_{1,2}$ is **omitted entirely when $N_2 = 1$** (a 1-D / linear array has no
  vertical beam index) — both in the bit count and in the serializer
  (`pack_type1`).
* $i_2$ dominates the cost because it is **per subband** ($\times N_3$). Mode 2
  roughly doubles $i_2$ bits at low rank (its larger $N_{i_2}$) in exchange for
  finer beams — the central Mode 1 vs Mode 2 trade-off.
* $i_{1,3}$ appears only for ranks 2–4.

**Worked example** — $N_1{=}4, N_2{=}2$ ($P{=}16$, $G_1{=}16, G_2{=}8$),
$N_3{=}10$, Mode 1, rank 1:
$i_{1,1} = \lceil\log_2 16\rceil = 4$, $i_{1,2} = \lceil\log_2 8\rceil = 3$,
$i_2 = 10\cdot\lceil\log_2 4\rceil = 20$ → **27 bits** for the whole band.

**Worked example (Mode 2)** — same array, $N_3{=}10$, rank 1:
$i_{1,1} = \lceil\log_2 8\rceil = 3$, $i_{1,2} = \lceil\log_2 4\rceil = 2$,
$i_2 = 10\cdot\lceil\log_2 16\rceil = 40$ → **45 bits**. The halved $i_1$ saves
2 bits but the wider $i_2$ costs 20 — Mode 2 pays for finer beams in subband bits,
which is only worth it when the channel is angularly fine relative to the grid.

Compare either to the hundreds of bits a Type II report costs (Chapter 3) — that
gap is the entire reason both exist.

---

## 10. Subset and rank restriction (the spec mechanism)

### 10.1 `typeI-SinglePanel-codebookSubsetRestriction` (`n1-n2` bitmap)

The codebase models subset restriction as a flat $G_1 G_2$-bit beam mask
(`_beam_allowed`). The *standardized* mechanism (Table-2 paragraph of the clause)
is a **beam-group bitmap** with a special-case for high-rank wide arrays:

* The bitmap has $N_1 O_1 \cdot N_2 O_2$ bits, bit $b_0$ = LSB. **Except** the
  high-rank/wide-array case below, bit $b_{(x_1,x_2)}$ is associated with *all*
  precoders whose beam group origin is $(O_1 \lfloor\cdot\rfloor + x_1,\,
  O_2\lfloor\cdot\rfloor + x_2)$ — i.e. it gates an entire orthogonal beam group,
  and a zero bit forbids every precoder built on any beam in that group.
* **When $v\ge5$ and $P\in\{16,24,32\}$** (the $P\ge16$ rank-5–8 constructions),
  a single precoder draws on *several* groups (the 3–4 fixed beams of §7.4 span
  $\{(0,0),(O_1,0),(0,O_2),(O_1,O_2)\}$-type offsets). The spec then requires
  *all* of the associated group bits to be 1; if any one is zero the precoder is
  forbidden.

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE (partial).** The code's
> `_beam_allowed` checks each individual beam against a flat mask and
> `_candidate_allowed` checks the specific beams a candidate uses, which
> *approximates* the spec but is not the exact group-bitmap semantics: it does
> not enforce restriction at orthogonal-group granularity, and it does not
> implement the "all associated bits must be 1" rule for the $P\ge16$, $v\ge5$
> multi-group precoders as a distinct check. For the common low-rank cases the
> behaviour coincides.

### 10.2 `typeI-SinglePanel-ri-Restriction`

Implemented. The 8-bit field $r_7\ldots r_0$ ($r_0$ LSB); $r_{v-1}=0$ forbids
both RI and PMI reporting for $v$ layers. `select` raises if the requested rank's
bit is 0, and `Type1Codebook.__init__` enforces an 8-element bitmap.

### 10.3 `typeI-SinglePanel-codebookSubsetRestriction-i2`

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** The 16-bit $i_2$
> restriction used with `reportQuantity = cri-RI-i1-CQI`.

When the UE reports only $i_1$ (not $i_2$) — the `cri-RI-i1-CQI` quantity, where
the gNB later picks $i_2$ and the UE computes CQI assuming a *random* $i_2$ — the
spec adds a separate 16-bit bitmap $b_{15}\ldots b_0$ ($b_0$ LSB). Bit $b_i$ is
associated with codebook index $i_2=i$; when $b_i=0$, the randomly selected
precoder used for CQI calculation must not correspond to that $i_2$. The codebase
has no `cri-RI-i1-CQI` reporting mode and no $i_2$ bitmap.

### 10.4 Reporting omission

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** Type I has no
> priority-based coefficient omission (it has no coefficients to drop). The only
> standardized "omission" here is the conditional dropping of $i_{1,2}$ when
> $N_2=1$ (implemented, §9) and of $i_{1,3}$ outside ranks 2–4 (implemented). The
> general two-part CSI priority/omission rules of §5.2.3 (which can drop the
> Part-2 wideband/subband PMI under UCI payload limits) are out of scope for this
> chapter and not modelled by `overhead_bits`.

---

## 11. Why Type I is "rigid" — and where it hurts

Type I sends a *direction*, not a *combination*. At rank 1 a single DFT beam is a
fine approximation of the dominant eigenvector when the channel is spatially
sparse (few clusters). At rank 2+ the layers are forced to be **fixed offsets of
one beam with prescribed sign patterns** — the precoder cannot independently shape
each layer. This is visible in the metrics: Type I's rank-2 column-wise SGCS can
look poor (~0.26) even though it spans ~0.52 of the eigen-subspace, because the
rigid pair captures the right *subspace* but not the right *rotation within it*
(see the figure analyses referenced in the repo README). When the channel is rich
or the rank is high, Type II's free coefficients pull decisively ahead — at many
times the feedback cost.

---

## 12. Implemented vs. standardized — quick map

| Spec feature | Status |
|---|---|
| 2-port codebook (Table 5.2.2.2.1-1) + `twoTX-CodebookSubsetRestriction` | 🚩 not implemented |
| Ranks 1–8 UPA constructions ($P\ge4$), Mode 1 | implemented |
| codebookMode 2 (rank 1–2 fine-offset $i_2$) | implemented |
| Tables 5.2.2.2.1-2/-3/-4 (configs, $i_{1,3}$ offsets) | implemented |
| Ranks 3–4 small ($P<16$) and large ($P\ge16$, $\theta$) | implemented |
| Ranks 5–8 fixed-beam patterns | implemented |
| `typeI-SinglePanel-ri-Restriction` | implemented |
| `typeI-SinglePanel-codebookSubsetRestriction` (group-bitmap exact semantics, $P\ge16$/$v\ge5$ all-bits rule) | 🚩 partial (per-beam approximation) |
| `typeI-SinglePanel-codebookSubsetRestriction-i2` (cri-RI-i1-CQI) | 🚩 not implemented |
| Part-2 CSI priority/omission (§5.2.3) | 🚩 not implemented (out of scope) |

---

**Next:** [Chapter 2 — Type I multi-panel](02-type1-multi-panel.md) generalizes
this to arrays built from several panels, adding inter-panel co-phasing.
