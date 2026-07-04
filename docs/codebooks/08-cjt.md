# Chapter 8 — Coherent Joint Transmission (CJT) Type II codebooks

* **Spec:** TS 38.214 v19.4.0 §5.2.2.2.8 (heading line 11136) and §5.2.2.2.9
  (heading line 11705) of
  [38.214-v19.4.0.md](../../specs/38.214-v19.4.0.md).
* **`codebookType`:** `typeII-CJT-r18` (§5.2.2.2.8) /
  `typeII-CJT-PortSelection-r18` (§5.2.2.2.9).
* **Code:** [cjt_r18.py](../../src/nr_csi/codebooks/cjt_r18.py) — classes
  `R18CJTCodebook` (§5.2.2.2.8) and `R18CJTPortSelectionCodebook` (§5.2.2.2.9),
  report type `R18CJTPMI`; configuration tables `CJT_L_COMBOS` /
  `CJT_PV_BETA` / `CJT_ALLOWED` and the PS analogues. Tests:
  [test_cjt_r18.py](../../tests/codebooks/test_cjt_r18.py). The base machinery
  is [Chapter 4 — eType II R16](04-etype2-r16.md) (CJT generalizes it) and
  [Chapter 5 — feType II PS R17](05-fetype2-r17.md) (the PS-CJT generalizes that).
* **Ports:** $P_\text{CSI-RS} = 2N_1N_2 \in \{4,8,12,16,24,32\}$ **per CSI-RS
  resource**, $N_\text{TRP} \in \{1,2,3,4\}$ resources. The code aggregates the
  resources' ports along the channel's port axis (resource-major, `[pol0;
  pol1]` within a resource); unselected resources' precoder rows are zero.
* **Ranks:** 1–4.
* **Compression:** spatial ($L_n$ beams *per TRP*) + delay ($M_v$ taps, shared
  across TRPs) + **inter-TRP co-phasing** across $N$ cooperating transmission
  points.
* **Prereq:** [Chapter 4](04-etype2-r16.md) (eType II spatial+delay machinery;
  CJT reuses it per TRP) and [Foundations](00-foundations.md).

Degeneracy anchors (asserted PMI-for-PMI by the test suite): with
$N_\text{TRP}=1$ and `mode2`, `R18CJTCodebook` reproduces `R16Type2Codebook`
exactly, and `R18CJTPortSelectionCodebook` reproduces `R17Type2Codebook`.

---

## 1. What CJT adds

**Coherent joint transmission** is multi-TRP: several geographically separated
transmission points (TRPs), each with its own CSI-RS resource, jointly serve one
UE as if they were one large distributed array. The UE must report a precoder
**for each cooperating TRP**, plus the **relative phase/timing between TRPs** so
their signals add coherently at the UE.

The CJT codebook is exactly Enhanced Type II (Chapter 4) **replicated per TRP**,
glued together with two CJT-specific ingredients:

1. a **per-TRP spatial basis** — each TRP $j$ selects its own $L_{\sigma_j}$ DFT
   beams in its own orthogonal group $(q_{1,j}, q_{2,j})$;
2. an **inter-TRP co-phasing** $\psi_j$ — a frequency-dependent phase ramp
   $e^{\,j2\pi t \psi_j / N_3}$ applied to TRP $j$'s contribution, modelling the
   relative delay/phase offset between TRPs.

The frequency-domain (delay) basis $M_v$ and the amplitude/phase quantization are
**shared with eType II R16 unchanged**. So if you understand Chapter 4, CJT is
"Chapter 4, per TRP, plus a TRP phase."

§5.2.2.2.9 is the **port-selection** flavour: replace the per-TRP DFT beams with
per-TRP freely/strided-selected ports (as in [Chapter 5](05-fetype2-r17.md)),
keep everything else.

---

## 2. TRP configuration and selection

* **$N_\text{TRP} \in \{1,2,3,4\}$** CSI-RS resources in the measurement resource
  set. All TRPs share the same $(N_1,N_2)$ and $(O_1,O_2)$ (Table 5.2.2.2.1-2),
  and the same $P_\text{CSI-RS} = 2N_1N_2$ per resource.
* **Resource (TRP) selection.** With higher-layer `restrictedCMR-Selection`, the
  UE uses all $N = N_\text{TRP}$ resources. Otherwise it **selects** $N$ resources
  ($1 \le N \le N_\text{TRP}$) and reports the choice as an $N_\text{TRP}$-bit
  bitmap $b_{N_\text{TRP}},\dots,b_1$ (resources ordered by their position in the
  set; the first selected resource is the lowest-index nonzero bit). The selected
  indices in increasing order are $\sigma_1 < \dots < \sigma_N$.

The precoder is then built from $\sum_{j=1}^{N} L_{\sigma_j} + M_v$ vectors:
the per-TRP beam sets plus the shared delay taps.

---

## 3. §5.2.2.2.8 parameter tables (transcribed)

CJT splits the parameter configuration into a **spatial** part (per-TRP beam
counts) and a **compression** part ($p_v, \beta$), then restricts which pairs may
combine.

### Table 5.2.2.2.8-1 — per-TRP beam counts $\{L_1,\dots,L_{N_\text{TRP}}\}$

Selected by `paramCombination-CJT-L-r18`; the number of *configured* combinations
$N_L \in \{1,2,4\}$ is `numberOfSDCombinations`. If $N_L > 1$ the UE reports which
combination it chose (index $0..N_L{-}1$); if $N_L = 1$ it is fixed and not
reported.

| $N_\text{TRP}$ | `paramCombination-CJT-L-r18` | $\{L_1,\dots,L_{N_\text{TRP}}\}$ |
|:--:|:--:|:--|
| 1 | 1 / 2 / 3 | {2} / {4} / {6} |
| 2 | 4 / 5 / 6 / 7 | {2,2} / {2,4} / {4,2} / {4,4} |
| 3 | 8 / 9 / 10 / 11 / 12 | {2,2,2} / {2,2,4} / {2,4,2} / {4,2,2} / {4,4,4} |
| 4 | 13 / 14 / 15 / 16 | {2,2,2,2} / {2,2,2,4} / {2,2,4,4} / {4,4,4,4} |

### Table 5.2.2.2.8-2 — $\{p_\upsilon, \beta\}$

Selected by `paramCombination-CJT-r18`:

| `paramCombination-CJT-r18` | $p_\upsilon$ ($\upsilon\in\{1,2\}$) | $p_\upsilon$ ($\upsilon\in\{3,4\}$) | $\beta$ |
|:--:|:--:|:--:|:--:|
| 1 | 1/8 | 1/16 | 1/4 |
| 2 | 1/8 | 1/16 | 1/2 |
| 3 | 1/4 | 1/8  | 1/4 |
| 4 | 1/4 | 1/8  | 1/2 |
| 5 | 1/4 | 1/4  | 3/4 |
| 6 | 1/2 | 1/4  | 1/2 |
| 7 | 1/2 | 1/2  | 1/2 |

### Table 5.2.2.2.8-3 — which $(L\text{-combo}, \{p_\upsilon,\beta\})$ pairs are allowed

Only the (row, column) pairs marked `x` are configurable (rows = the 16
`paramCombination-CJT-L-r18` values, columns = the 7 `paramCombination-CJT-r18`
values). For example L-combo 12 = {4,4,4} pairs with $\{p_\upsilon,\beta\}$
columns 1, 3, 4, 5 and 7; the L=6 single-TRP combos (1–3, columns near 2/3) are
restricted to the lower-$\beta$ rows. (See the spec table for the full grid.)

**Not-expected configurations** (`paramCombination-CJT-L-r18` barred):
2,3,5,6,7,9,10,11,12,14,15,16 when $P=4$; 3 when $P<32$; 3 when
`typeII-CJT-RI-Restriction-r18` allows any rank $>2$; 3 when $R=2$.

### Other §5.2.2.2.8 parameters

* **$R \in \{1,2\}$** — `numberOfPMI-SubbandsPerCQI-Subband-CJT-r18`; $R$ and
  $N_3$ as in §5.2.2.2.5 (Chapter 4).
* **RI restriction** — `typeII-CJT-RI-Restriction-r18`, a 4-bit bitmap
  $r_3r_2r_1r_0$; $r_i=0$ forbids rank $i{+}1$. $\upsilon \le 4$.
* **Subset restriction** — `n1-n2-codebookSubsetRestriction-CJT-r18`, configured
  per resource (only the '00'/'11' codepoints of Table 5.2.2.2.5-6 are allowed);
  unconfigured resources are unrestricted.

---

## 4. §5.2.2.2.8 PMI structure

The PMI is again split $i_1$ (wideband) / $i_2$ (per coefficient), now with
per-TRP and per-layer substructure. For rank $\upsilon$:

$$i_1 = \big\{\, i_{1,1},\ i_{1,2},\ i_{1,5},\ \{i_{1,6,l}, i_{1,7,l}, i_{1,8,l}\}_{l=1..\upsilon},\ i_{1,9}\,\big\},$$
$$i_2 = \big\{\, \{i_{2,3,l},\ i_{2,4,l},\ i_{2,5,l}\}_{l=1..\upsilon}\,\big\}.$$

| Field | Meaning | Structure |
|---|---|---|
| $i_{1,1}$ | per-TRP orthogonal-group rotation | $[i_{1,1,1}\dots i_{1,1,N}]$, $i_{1,1,j}=[q_{1,j}\,q_{2,j}]$ |
| $i_{1,2}$ | per-TRP beam combination (Algorithm 1) | $[i_{1,2,1}\dots i_{1,2,N}]$, $i_{1,2,j}\in\{0..\binom{N_1N_2}{L_{\sigma_j}}{-}1\}$ |
| $i_{1,5}$ | delay window origin $M_\text{initial}$ ($N_3>19$) | shared across TRPs and layers |
| $i_{1,6,l}$ | delay-tap combination (Algorithm 3), per layer | **common to all TRPs** |
| $i_{1,7,l}$ | nonzero-coefficient bitmap, per layer | spans all TRPs: $[i_{1,7,l,1}\dots i_{1,7,l,N}]$ |
| $i_{1,8,l}$ | strongest-coefficient indicator, per layer | over $0..2\sum_j L_{\sigma_j}{-}1$ |
| $i_{1,9}$ | **inter-TRP co-phasing offsets** $[d_2\dots d_N]$ | mode1 only |
| $i_{2,3,l}$ | per-pol reference amplitude $[k^{(1)}_{l,0}\,k^{(1)}_{l,1}]$ | 4-bit, $\in\{1..15\}$ |
| $i_{2,4,l}$ | differential amplitudes (per TRP, tap, beam) | 3-bit each |
| $i_{2,5,l}$ | phases (per TRP, tap, beam) | 16-PSK, $\in\{0..15\}$ |

The per-TRP beam vectors are built exactly as in §5.2.2.2.3: decode $i_{1,2,j}$ to
$L_{\sigma_j}$ index pairs $(n_{1,j}^{(i)}, n_{2,j}^{(i)})$, map into the TRP's
orthogonal group $m_{1,j}^{(i)} = O_1 n_{1,j}^{(i)} + q_{1,j}$,
$m_{2,j}^{(i)} = O_2 n_{2,j}^{(i)} + q_{2,j}$, and form $v_{m_{1,j},m_{2,j}}$.

The $M_v = \lceil p_\upsilon N_3/R\rceil$ delay taps are **common to all selected
TRPs** (one set of taps per layer), indicated by $i_{1,5}/i_{1,6,l}$ as in
Chapter 4.

---

## 5. §5.2.2.2.8 inter-TRP co-phasing $\psi_j$

This is the genuinely new CJT mechanism. With `codebookMode = mode1`, the UE
reports, for each non-reference selected TRP $j = 2..N$, an offset $d_j$ relative
to the first selected TRP:

$$i_{1,9} = [d_2\dots d_N], \qquad d_j \in \{0,1,\dots,N_3 O_3 - 1\}, \qquad
\psi_j = \frac{d_j}{O_3},$$

where **$O_3 \in \{1,4\}$** (`numberOfO3`) oversamples the inter-TRP phase grid.
The offset enters the reconstruction as a per-TRP frequency-dependent phase ramp
$e^{\,j2\pi t \psi_j / N_3}$ — i.e. a **relative delay** between TRPs, exactly the
quantity needed for coherent combining. With `codebookMode = mode2`, $i_{1,9}$ is
**not** reported and $\psi_j = 0$ (TRPs co-phased only through their per-coefficient
phases).

---

## 6. §5.2.2.2.8 reconstruction (Table 5.2.2.2.8-4)

Each layer's precoder stacks $N$ per-TRP blocks vertically (one block of
$P_\text{CSI-RS}$ rows per selected TRP). For layer $l$ and subband $t$:

$$W^l_{\dots,t} = \frac{1}{\sqrt{N_1 N_2\, \gamma_{t,l}}}
\begin{bmatrix}
\text{TRP }1:\ \sum_{i} v_{m_{1,1}^{(i)},m_{2,1}^{(i)}}\, p^{(1)}_{l,\cdot}\, \sum_f y_{t,l}^{(f)} p^{(2)}_{l,i,f,1}\varphi_{l,i,f,1} \\
\text{TRP }2:\ e^{\,j2\pi t\psi_2/N_3}\sum_{i} v_{m_{1,2}^{(i)},m_{2,2}^{(i)}}\, p^{(1)}_{l,\cdot}\, \sum_f y_{t,l}^{(f)} p^{(2)}_{l,i,f,2}\varphi_{l,i,f,2} \\
\vdots \\
\text{TRP }N:\ e^{\,j2\pi t\psi_N/N_3}\sum_{i} v_{m_{1,N}^{(i)},m_{2,N}^{(i)}}\, p^{(1)}_{l,\cdot}\, \sum_f y_{t,l}^{(f)} p^{(2)}_{l,i,f,N}\varphi_{l,i,f,N}
\end{bmatrix},$$

(each TRP block has the two polarization sub-blocks, as in Chapter 4), with

$$\gamma_{t,l} = \sum_{j=1}^{N}\sum_{i=0}^{2L_{\sigma_j}-1}
\Big(p^{(1)}_{l,\lfloor i/L_{\sigma_j}\rfloor}\Big)^2
\Big|\sum_f y_{t,l}^{(f)} p^{(2)}_{l,i,f,j}\varphi_{l,i,f,j}\Big|^2,$$

and the $\upsilon$-layer precoder is the column stack scaled by $1/\sqrt{\upsilon}$
(prefactors $1, \tfrac1{\sqrt2}, \tfrac1{\sqrt3}, \tfrac12$). Note:

* TRP 1 (reference) has no phase ramp ($\psi_1 \equiv 0$); TRP $j$ carries
  $e^{\,j2\pi t\psi_j/N_3}$.
* The single normalization $\gamma_{t,l}$ sums energy **across all TRPs**, so the
  stacked precoder is unit-norm per subband/layer.
* Phases are 16-PSK, amplitudes reference (4-bit) × differential (3-bit), as in
  Chapter 4 — only now indexed by TRP $j$ as well.

---

## 7. §5.2.2.2.8 coefficient budget and strongest coefficient

The coefficient budget carries the **total** per-TRP beam count:

$$K_0 = \Big\lceil 2\beta M_1 \sum_{j=1}^{N} L_{\sigma_j}\Big\rceil,$$

with $M_1 = M_\upsilon|_{\upsilon=1}$. Per layer $K_l^{NZ} \le K_0$ and across
layers $K^{NZ} = \sum_l K_l^{NZ} \le 2K_0$, encoded by the bitmap $i_{1,7,l}$
(now spanning $2\sum_j L_{\sigma_j} \times M_\upsilon$ per layer).

The **strongest coefficient** of layer $l$ is at $(f^\star_l, i^\star_l)$ over the
concatenated TRP/beam axis $i^\star_l \in \{0..2\sum_j L_{\sigma_j}-1\}$; the spec
remaps so $f^\star_l = 0$ (delay) and its $(q_1,q_2,\text{beam})$ becomes the
reference. As in Chapter 4 the strongest entry is **not reported**:
$k^{(1)}=15$, $k^{(2)}=7$, $c=0$ at that entry; only the *other* polarization's
reference amplitude and the $K^{NZ}-\upsilon$ non-strongest amplitudes/phases are
sent. The dual-mode $i_{1,8,l}$ encoding (rank-1: cumulative position among the
tap-0 bitmap bits; rank>1: the index directly) matches Chapter 4 §6.

---

## 8. §5.2.2.2.9 — the CJT port-selection variant

`typeII-CJT-PortSelection-r18` is the port-selection flavour: replace each TRP's
DFT beams with **per-TRP selected ports** (as in [Chapter 5](05-fetype2-r17.md)),
and use a tiny shared delay window.

### Parameters (Tables 5.2.2.2.9-1/2/3)

* **Per-TRP $\alpha_n$** via `paramCombination-CJT-PS-alpha-r18` (20 rows,
  $N_L\in\{1,2,4\}$ via `numberOfSDCombinations-PS`). Each $\alpha_n \in
  \{\tfrac12, \tfrac34, 1\}$ sets that TRP's port count $L_{\sigma_j} =
  \alpha_{\sigma_j}\, P/2$. Examples: combo 4 = {1/2,1/2}, combo 8 = {1,1},
  combo 16 = {1,1,1}, combo 20 = {1,1,1,1}.
* **$\{M,\beta\}$** via `paramCombination-CJT-PS-r18` (Table 5.2.2.2.9-2):

  | `paramCombination-CJT-PS-r18` | $M$ | $\beta$ |
  |:--:|:--:|:--:|
  | 1 | 1 | 1/2 |
  | 2 | 1 | 3/4 |
  | 3 | 1 | 1 |
  | 4 | 2 | 1/2 |
  | 5 | 2 | 3/4 |

  Configurable $(\alpha\text{-combo}, \{M,\beta\})$ pairs are the `x` cells of
  Table 5.2.2.2.9-3.
* **$N_M \in \{2,4\}$** (`valueOfN-CJT-r18`) and **$R \in \{1,2\}$**
  (`numberOfPMI-SubbandsPerCQI-Subband-CJT-PS-r18`) apply only when $M=2$
  ($R=1$ when $M=1$) — the small reciprocity tap window of Chapter 5.
* **RI restriction** `typeII-CJT-PS-RI-Restriction-r18` (4-bit). TRP selection
  bitmap as in §5.2.2.2.8.

### PMI (§5.2.2.2.9)

$$i_1 = \big\{\, i_{1,2},\ i_{1,6},\ \{i_{1,7,l}, i_{1,8,l}\}_{l=1..\upsilon},\ i_{1,9}\,\big\},
\qquad i_2 = \{i_{2,3,l}, i_{2,4,l}, i_{2,5,l}\}_{l=1..\upsilon}.$$

* $i_{1,2}$ — per-TRP **port** combination (the combinatorial port codec of
  Chapter 5), reported per TRP where $\alpha_{\sigma_j} < 1$.
* $i_{1,6}$ — the second delay-tap offset (only $M=2$, window $N_M=4$), as in
  Chapter 5 — common to all TRPs/layers.
* $i_{1,9}$ — inter-TRP co-phasing offsets $\psi_j$, same role as §5.2.2.2.8.
* $i_{1,7,l}, i_{1,8,l}, i_{2,3,l}, i_{2,4,l}, i_{2,5,l}$ — bitmap, strongest
  indicator, reference/differential amplitudes, phases — per layer, per TRP.

### Reconstruction (§5.2.2.2.9)

Identical stacked form as §5.2.2.2.8 §6, with two changes: the beam vectors
$v_{m_{1,j},m_{2,j}}$ become the per-TRP **standard-basis port vectors**, and the
$\tfrac{1}{\sqrt{N_1N_2}}$ factor is **dropped** (port domain) — exactly the
DFT→port substitution of Chapter 5. The delay sum runs over $M$ taps in the
window $\{0..\min(N_M,N_3){-}1\}$. The inter-TRP phase $e^{\,j2\pi t\psi_j/N_3}$
and the budget $K_0 = \lceil 2\beta M_1 \sum_j L_{\sigma_j}\rceil$ are as in
§5.2.2.2.8.

---

## 9. Overhead and summary

The CJT report is, roughly, **$N$ copies of an eType II (or feType II PS) report**
plus the TRP-selection bitmap ($N_\text{TRP}$ bits), the SD-combination index
($\lceil\log_2 N_L\rceil$ bits when $N_L>1$), and the inter-TRP co-phasing
$i_{1,9}$ ($\sum_{j=2}^N \lceil\log_2(N_3 O_3)\rceil$ bits in mode1, 0 in mode2).
The coefficient payload scales with $K^{NZ} \le 2K_0$, and $K_0$ itself grows with
$\sum_j L_{\sigma_j}$ — so CJT overhead grows roughly linearly in the number of
cooperating TRPs, which is the cost of coherent multi-TRP feedback.

| | §5.2.2.2.8 (CJT eType II) | §5.2.2.2.9 (CJT feType II PS) |
|---|---|---|
| Per-TRP spatial | $L_n$ DFT beams in group $(q_{1,n},q_{2,n})$ | $L_n = \alpha_n P/2$ selected ports |
| Spatial param | `paramCombination-CJT-L-r18` | `paramCombination-CJT-PS-alpha-r18` |
| Compression param | `paramCombination-CJT-r18` ($p_\upsilon,\beta$) | `paramCombination-CJT-PS-r18` ($M,\beta$) |
| Delay | $M_\upsilon = \lceil p_\upsilon N_3/R\rceil$ taps, shared | $M\in\{1,2\}$ taps, window $N_M$, shared |
| Inter-TRP phase | $\psi_j$ via $i_{1,9}$, $O_3\in\{1,4\}$ (mode1) | same |
| $N_1N_2$ factor | yes | no (port domain) |
| Budget | $K_0 = \lceil 2\beta M_1\sum_j L_{\sigma_j}\rceil$ | same |
| Base codebook | [Chapter 4](04-etype2-r16.md) | [Chapter 5](05-fetype2-r17.md) |

> 🚩 **Reminder: none of this is implemented.** To add CJT to the codebase you
> would generalize `R16Type2Codebook` / `R17Type2Codebook` to (a) carry an
> $N_\text{TRP}$ axis on the spatial basis and coefficients, (b) add the
> per-TRP group/port selection, (c) add the $\psi_j$ inter-TRP phase ramp and its
> $i_{1,9}$ report field, and (d) widen $K_0$ to $\lceil 2\beta M_1\sum_j
> L_{\sigma_j}\rceil$. The delay/amplitude/phase/bitmap machinery and the
> serializer's coefficient block could be reused largely as-is.

---

This is the last codebook chapter. Back to the [index](README.md), or see the
base families this one generalizes: [eType II R16](04-etype2-r16.md) and
[feType II PS R17](05-fetype2-r17.md).
