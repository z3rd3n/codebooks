# Chapter 4 — R16 Enhanced Type II codebook (eType II, + port selection)

* **Spec:** TS 38.214 (Release-18, "i00") **§5.2.2.2.5 "Enhanced Type II
  Codebook"** (regular) and **§5.2.2.2.6 "Enhanced Type II Port Selection
  Codebook"** (PS variant). Clause numbers/titles verified against
  [specs/38214-i00.md](../../specs/38214-i00.md).
* **Code:** [etype2_r16.py](../../src/nr_csi/codebooks/etype2_r16.py) — class
  `R16Type2Codebook`; validation in
  [validate.py](../../src/nr_csi/codebooks/validate.py) (`validate_r16`);
  bit packing in [serialize.py](../../src/nr_csi/codebooks/serialize.py)
  (`pack_r16`/`unpack_r16`); param tables in
  [config.py](../../src/nr_csi/config.py).
* **Ports:** $P_\text{CSI-RS} = 2N_1N_2 \in \{4,8,12,16,24,32\}$.
* **Ranks:** 1–4 (`typeII-r16` / `typeII-PortSelection-r16`).
* **Compression:** spatial ($L$ beams) **+ delay-domain (frequency)** — the
  per-subband coefficient block of R15 is replaced by $M_v$ DFT **delay taps**.
* **Prereq:** [Chapter 3 — Type II R15](03-type2-r15.md) (it is the same beam
  combination; this chapter only adds the frequency axis) and
  [Foundations](00-foundations.md) (Algorithm 3, the FFT sign convention,
  `R16_REF_AMP`/`R16_DIFF_AMP`).

R15's biggest cost is the per-subband phase block ($i_{2,1}$, $\propto N_3$).
R16's insight: as a function of frequency, each beam's coefficient is the response
of a small number of **delays**. So instead of $N_3$ independent coefficients per
beam, send a few delay-tap coefficients and let the DFT regenerate all $N_3$. This
is the single most important step in the Type II family — every later release
keeps it.

> **Multi-TRP CJT extension.** Release-18 generalizes this codebook to coherent
> joint transmission across $N$ TRPs (`typeII-CJT-r18`, §5.2.2.2.8). That
> per-TRP extension is documented in [Chapter 8 — CJT](08-cjt.md), not here.

---

## 1. The reconstruction formula

For layer $l$ and frequency unit $t$ (TS 38.214 Table 5.2.2.2.5-5 / paper Table
`tabesII`):

$$W^l_{t} = \frac{1}{\sqrt{N_1 N_2\, \gamma_{t,l}}}
\begin{bmatrix}
\sum_{i=0}^{L-1} v_{m_1^{(i)},m_2^{(i)}}\, p^{(1)}_{l,0}\,
   \sum_{f=0}^{M_v-1} y_{t,l}^{(f)}\, p^{(2)}_{l,i,f}\, \varphi_{l,i,f} \\[3pt]
\sum_{i=0}^{L-1} v_{m_1^{(i)},m_2^{(i)}}\, p^{(1)}_{l,1}\,
   \sum_{f=0}^{M_v-1} y_{t,l}^{(f)}\, p^{(2)}_{l,i+L,f}\, \varphi_{l,i+L,f}
\end{bmatrix},$$

$$\gamma_{t,l} = \sum_{i=0}^{2L-1} \Big(p^{(1)}_{l,\lfloor i/L\rfloor}\Big)^2
\Big|\textstyle\sum_{f=0}^{M_v-1} y_{t,l}^{(f)}\, p^{(2)}_{l,i,f}\,
\varphi_{l,i,f}\Big|^2,$$

with the DFT frequency (delay) basis and 16-PSK phase

$$y_{t,l}^{(f)} = e^{\,j\,2\pi\, t\, n_{3,l}^{(f)}/N_3}, \qquad
\varphi_{l,i,f} = e^{\,j\,2\pi\, c_{l,i,f}/16}.$$

The $\upsilon$-layer precoder is $W^{(\upsilon)} = \frac{1}{\sqrt{\upsilon}}
[W^1 \; W^2 \; \cdots \; W^\upsilon]$ (the $1/\sqrt{\upsilon}$ is explicit in the
spec's layer tables and in `precoder`).

Compared with R15:

* The single per-subband coefficient $p^{(1)}_i p^{(2)}_i \phi_i$ becomes a sum
  over **delay taps** $f$: $\sum_f y_{t,l}^{(f)} p^{(2)}_{l,i,f}\varphi_{l,i,f}$.
  The frequency variation is now *generated* by the DFT, not reported per subband.
* The amplitude factors into a **per-polarization reference** $p^{(1)}_{l,0/1}$
  (4-bit `R16_REF_AMP`) times a **per-coefficient differential** $p^{(2)}$ (3-bit
  `R16_DIFF_AMP`).
* $\gamma_{t,l}$ normalizes per frequency unit (so $\lVert W^l_t\rVert = 1$).
* For coefficients with bitmap bit $k^{(3)}_{l,i,f} = 0$ the spec sets
  $p^{(2)}_{l,i,f} = 0$ and $\varphi_{l,i,f} = 0$ (the bitmap masks them out).

The port-selection variant (§5.2.2.2.6, Table 5.2.2.2.6-2) replaces
$v_{m_1^{(i)},m_2^{(i)}}$ by the standard-basis port vector $v_{i_{1,1}d+i}$ and
**drops the $\tfrac{1}{\sqrt{N_1N_2}}$ factor** (so $W^l_t =
\tfrac{1}{\sqrt{\gamma_{t,l}}}[\cdots]$) — same structural change as R15 PS.

---

## 2. Derived parameters: $M_v$, $K_0$, $N_3$, $R$

R16 is governed by `paramCombination-r16` (`R16_PARAM_COMBOS`), a preset of
$(L, p_v, \beta)$:

```python
@dataclass(frozen=True)
class R16ParamCombo:
    index: int
    L: int                 # beams
    p_v12: Fraction        # p_v for ranks 1-2
    p_v34: Fraction | None # p_v for ranks 3-4 (None = rank 3-4 unsupported)
    beta:  Fraction        # coefficient-budget fraction
```

From these:

* **Number of delay taps** per rank (`m_v`, `Mv()`):
  $$M_v = \big\lceil p_v \cdot N_3 / R \big\rceil .$$
* **Per-layer coefficient budget** (`K0`):
  $$K_0 = \big\lceil \beta \cdot 2L \cdot M_1 \big\rceil ,$$
  where $M_1$ is $M_v$ evaluated at **rank 1** (i.e. $p_{v=1}$). Each layer keeps
  at most $K_0$ nonzero coefficients ($K^{NZ}_l \le K_0$), and the **total over
  layers** is capped at $2K_0$ ($K^{NZ} = \sum_l K^{NZ}_l \le 2K_0$).

### 2.1 $N_3$ and $R$ (frequency-unit count)

$N_3$ is the number of precoding matrices the PMI indicates across frequency. It
is $N_3 = (\text{number of configured CQI subbands in } \textit{csi-ReportingBand})
\times R$, where $R \in \{1,2\}$ comes from
*numberOfPMI-SubbandsPerCQI-Subband*.

* **$R=1$:** one precoder per subband.
* **$R=2$:** two precoders per subband (first/last half-subband), with special
  first/last-subband PRB-splitting rules (TS 38.214 §5.2.2.2.5, the
  $N_\text{BWP}^\text{start} \bmod N_\text{PRB}^\text{SB}$ cases).

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** The $R=2$ first/last
> half-subband PRB-mapping rules (which physical PRBs each of the two precoders
> covers). [config.py](../../src/nr_csi/config.py) computes the *count* $N_3 =
> n\_subbands \cdot R$ but does not map precoders to half-subband PRB ranges; the
> codebook math itself is $R$-agnostic and correct for any $N_3$.

### 2.2 The full `paramCombination-r16` table (Table 5.2.2.2.5-1)

All eight rows (`R16_PARAM_COMBOS`), transcribed verbatim:

| `paramCombination-r16` | $L$ | $p_v,\ v\in\{1,2\}$ | $p_v,\ v\in\{3,4\}$ | $\beta$ |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 2 | 1/4 | 1/8 | 1/4 |
| 2 | 2 | 1/4 | 1/8 | 1/2 |
| 3 | 4 | 1/4 | 1/8 | 1/4 |
| 4 | 4 | 1/4 | 1/8 | 1/2 |
| 5 | 4 | 1/4 | 1/4 | 3/4 |
| 6 | 4 | 1/2 | 1/4 | 1/2 |
| 7 | 6 | 1/4 | — | 1/2 |
| 8 | 6 | 1/4 | — | 3/4 |

Rows 7–8 ($L=6$) have no $p_{v\in\{3,4\}}$: **ranks 3–4 are not supported**
(`p_v34 = None`, and `R16ParamCombo.p_v(3/4)` raises).

**Configuration prohibitions (TS 38.214 §5.2.2.2.5, "the UE is not expected to be
configured with..."):**

* combos 3,4,5,6,7,8 when $P_\text{CSI-RS} = 4$;
* combos 7,8 when $P_\text{CSI-RS} < 32$;
* combos 7,8 when *typeII-RI-Restriction-r16* sets $r_i = 1$ for any $i > 1$;
* combos 7,8 when $R = 2$.

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** These
> per-port-count / RI-restriction / $R=2$ prohibitions on combos 3–8 are not
> enforced. The code validates $L \le N_1N_2$ at construction but accepts any
> table row for any port count.

### 2.3 The Enhanced Type II Port-Selection table (Table 5.2.2.2.6-1)

§5.2.2.2.6 has its **own** parameter table — **rows 1–6 only** (the $L=6$ rows do
not exist for PS). Transcribed verbatim:

| `paramCombination-r16` (PS) | $L$ | $p_v,\ v\in\{1,2\}$ | $p_v,\ v\in\{3,4\}$ | $\beta$ |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 2 | 1/4 | 1/8 | 1/4 |
| 2 | 2 | 1/4 | 1/8 | 1/2 |
| 3 | 4 | 1/4 | 1/8 | 1/4 |
| 4 | 4 | 1/4 | 1/8 | 3/4 *(see note)* |
| 5 | 4 | 1/4 | 1/4 | 3/4 |
| 6 | 4 | 1/2 | 1/4 | 1/2 |

> **Code note (`R16_PS_PARAM_COMBOS`).** The PS table in §5.2.2.2.6 is *its own
> thing*, but its rows 1–6 are numerically **identical to rows 1–6 of the regular
> Table 5.2.2.2.5-1** in the spec (row 4 has $\beta = 1/2$ in both spec tables;
> the "3/4" above is only to flag where you must read the PS table directly). The
> codebase therefore **reuses the regular table**:
> `R16_PS_PARAM_COMBOS = {i: R16_PARAM_COMBOS[i] for i in range(1,7)}`, dropping
> rows 7–8. `R16Type2Codebook(port_selection=True)` raises if you ask for combo
> 7 or 8.

A `combo` argument can override the table for generalized $(L,p_v,\beta)$ sweeps
(used by the Qin Fig. 5 reproduction); the default looks up the spec row.

---

## 3. The PMI structure

The standardized $i_1$/$i_2$ split (Table 5.2.2.2.5-1, expanded per rank) is:

* **$i_1$** (wideband-ish): $i_{1,1}$ (rotation $q_1,q_2$), $i_{1,2}$ (beam
  combination), $i_{1,5}$ ($M_\text{initial}$, only $N_3>19$, shared), and per
  layer $l$: $i_{1,6,l}$ (taps), $i_{1,7,l}$ (bitmap), $i_{1,8,l}$ (strongest).
* **$i_2$** (per layer $l$): $i_{2,3,l}$ (reference amplitudes), $i_{2,4,l}$
  (differential amplitudes), $i_{2,5,l}$ (phases).

For PS (§5.2.2.2.6) $i_1$ has **$i_{1,1}$ only** for the spatial part (no
$i_{1,2}$); everything else is identical.

```python
@dataclass
class R16Type2PMI:
    rank: int
    # spatial: regular (q1,q2,i12) OR port-selection (i11_ps)
    q1, q2, i12: ...     # i_{1,1}=(q1,q2), i_{1,2}=i12
    i11_ps: ...          # i_{1,1} for PS
    # frequency:
    i15: int | None          # M_initial indicator (only when N3 > 19)
    i16: list[int]           # per-layer delay-tap combination (Algorithm 3)
    # coefficients (per layer):
    i17: np.ndarray          # bitmap (v, Mv, 2L), bool  -> i_{1,7,l}
    i18: list[int]           # strongest-coefficient indicator i_{1,8,l} (dual mode)
    k1:  np.ndarray          # (v, 2) per-polarization reference amplitude, 1..15
    k2:  np.ndarray          # (v, Mv, 2L) differential amplitude, 0..7
    c:   np.ndarray          # (v, Mv, 2L) phase, 0..15  (16-PSK)
```

New vs R15:

* **$i_{1,5}, i_{1,6,l}$** — the delay-tap selection (§4). $i_{1,6,l}$ is per
  layer; $i_{1,5}$ ($M_\text{initial}$) appears only when $N_3 > 19$ and is shared
  by all layers.
* **$i_{1,7,l}$** — the nonzero-coefficient **bitmap** ($M_v \times 2L$ per
  layer, $k^{(3)}_{l,i,f}\in\{0,1\}$). R15 used the amplitude-0 level; R16's
  differential table has no zero, so a bitmap is required.
* **$i_{1,8,l}$** — strongest-coefficient indicator (replaces R15's $i_{1,3}$),
  with a "dual mode" encoding (§6).
* Phases are **16-PSK** (`N_PSK = 16`, 4 bits) — finer than R15's 8-PSK.

---

## 4. Delay-tap (FD) basis selection (Algorithm 3 + two-level indication)

The $M_v$ delay taps $n_{3,l} = [n_{3,l}^{(0)},\dots,n_{3,l}^{(M_v-1)}]$,
$n_{3,l}^{(f)} \in \{0,\dots,N_3-1\}$, define the DFT vectors
$y_{t,l}^{(f)} = e^{j2\pi t n_{3,l}^{(f)}/N_3}$. By construction (after remapping
the strongest coefficient's tap to index 0) **$n_{3,l}^{(0)} = 0$ for every
layer**.

`select_taps(tap_energy, Mv, N3, m_initial)` chooses the taps:

* **$N_3 \le 19$:** simply the top-$M_v$ taps by energy (after remapping the
  strongest tap to index 0). Encoded directly by the reverse-lex combinatorial
  index over $\binom{N_3-1}{M_v-1}$ (the strongest tap 0 is fixed and excluded).
  $i_{1,5}$ is not reported ($=0$).
* **$N_3 > 19$:** the nonzero taps must fit a cyclic **window of $2M_v$**
  consecutive positions
  $\mathrm{IntS} = \{(M_\text{initial}+i)\bmod N_3,\ i=0,\dots,2M_v-1\}$ with
  $$M_\text{initial} \in \{-2M_v+1,\dots,0\}.$$
  The selector searches windows, keeps the most energetic, and `encode_taps`
  returns both the window-relative combination $i_{1,6,l}$ (over
  $\binom{2M_v-1}{M_v-1}$) and the window origin $i_{1,5}$. Because $i_{1,5}$ is
  reported once, all layers are forced to share a window (`m_init_common`).

**Index ranges (spec):**

$$i_{1,5} \in \{0,1,\dots,2M_v-1\}, \qquad
i_{1,6,l} \in
\begin{cases}
\{0,\dots,\binom{N_3-1}{M_v-1}-1\} & N_3 \le 19,\\[3pt]
\{0,\dots,\binom{2M_v-1}{M_v-1}-1\} & N_3 > 19.
\end{cases}$$

$i_{1,5}$ maps to $M_\text{initial}$ as $i_{1,5}=M_\text{initial}$ if
$M_\text{initial}=0$, else $i_{1,5}=M_\text{initial}+2M_v$ (this exact convention
is in `select`/`encode_taps`). The combinatorial sum
$i_{1,6,l}=\sum_{f=1}^{M_v-1}C(\cdot,M_v-f)$ uses the $C(x,y)$ table
(5.2.2.2.5-4); see Algorithm 3 in
[combinatorics.py](../../src/nr_csi/utils/combinatorics.py). When $M_v = 1$,
$i_{1,6,l}=0$ and is not reported.

The strongest tap is **remapped to index 0** (the spec's
$n_{3,l}^{(f)} \leftarrow (n_{3,l}^{(f)} - n_{3,l}^{(f^\star_l)})\bmod N_3$,
$f \leftarrow (f - f^\star_l)\bmod M_v$) and excluded from the combinatorial
encoding; only the other $M_v-1$ taps are encoded.

---

## 5. Reconstruction in code (`precoder`)

```python
for l in range(rank):
    taps  = decode_taps(i16[l], N3, Mv, i15)        # Algorithm 3 -> tap indices
    Y     = freq_basis(N3, taps).T                  # (N3, Mv)  DFT regen matrix
    x     = layer_coefficients(pmi, l)              # (2L, Mv), includes p1*p2*phi*bitmap
    ct    = x @ Y.T                                 # (2L, N3)  back to all subbands
    gamma = sum(|ct|^2, axis=0)                     # (N3,) per-subband normalization
    w     = [ B.T @ ct[:L] ; B.T @ ct[L:] ]         # (P, N3)
    W[0,:,:,l] = (w / sqrt(scale * gamma)).T
W /= sqrt(rank)
```

`_layer_coefficients` assembles $x_{i,f} = p^{(1)}_{l,\lfloor i/L\rfloor}\,
p^{(2)}_{l,i,f}\, \varphi_{l,i,f}\, \mathbb{1}[\text{bitmap}]$ — the reference
amplitude per polarization, times the differential amplitude, times the 16-PSK
phase, masked by the bitmap. The matrix multiply `x @ Y.T` is the delay→subband
DFT that regenerates the full frequency response. `scale` is $N_1N_2$ (regular)
or $1$ (PS), per the two layer tables.

### 5.1 Amplitude/phase quantization tables

**Reference amplitude** $i_{2,3,l} = [k^{(1)}_{l,0}\ k^{(1)}_{l,1}]$,
$k^{(1)}_{l,p} \in \{1,\dots,15\}$ — 4 bits per polarization (Table 5.2.2.2.5-2).
The mapping is $p^{(1)}_{l,p} = 2^{-(15-k)/4}$ (code `R16_REF_AMP`,
$k{=}0$ Reserved/`nan`):

| $k^{(1)}$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| $p^{(1)}$ | $\tfrac{1}{\sqrt{128}}$ | $(\tfrac{1}{8192})^{1/4}$ | $\tfrac{1}{8}$ | $(\tfrac{1}{2048})^{1/4}$ | $\tfrac{1}{2\sqrt{8}}$ | $(\tfrac{1}{512})^{1/4}$ | $\tfrac14$ | $(\tfrac{1}{128})^{1/4}$ | $\tfrac{1}{\sqrt8}$ | $(\tfrac{1}{32})^{1/4}$ | $\tfrac12$ | $(\tfrac18)^{1/4}$ | $\tfrac{1}{\sqrt2}$ | $(\tfrac12)^{1/4}$ | $1$ |

**Differential amplitude** $i_{2,4,l}$, $k^{(2)}_{l,i,f} \in \{0,\dots,7\}$ —
3 bits per coefficient (Table 5.2.2.2.5-3). The mapping is
$p^{(2)}_{l,i,f} = 2^{-(7-k)/2}$ (code `R16_DIFF_AMP`):

| $k^{(2)}$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| $p^{(2)}$ | $\tfrac{1}{8\sqrt2}$ | $\tfrac18$ | $\tfrac{1}{4\sqrt2}$ | $\tfrac14$ | $\tfrac{1}{2\sqrt2}$ | $\tfrac12$ | $\tfrac{1}{\sqrt2}$ | $1$ |

Note **the differential table has no zero entry** (minimum $1/(8\sqrt2)$); that is
exactly why a bitmap is needed to switch coefficients off.

**Phase** $i_{2,5,l}$, $c_{l,i,f} \in \{0,\dots,15\}$ — 4-bit **16-PSK**,
$\varphi_{l,i,f} = e^{j2\pi c_{l,i,f}/16}$.

The full reported coefficient magnitude for a kept coefficient is the product
$p^{(1)}_{l,\lfloor i/L\rfloor}\cdot p^{(2)}_{l,i,f}$ (per-polarization reference
× per-coefficient differential).

---

## 6. The strongest-coefficient indicator $i_{1,8}$ (dual mode)

The strongest coefficient of layer $l$ sits at $(f^\star_l, i^\star_l)$; after the
remapping of §4 it is at $f^\star_l = 0$, i.e. $(f{=}0, i^\star_l)$. The spec
defines (TS 38.214 §5.2.2.2.5):

$$i_{1,8,l} =
\begin{cases}
\sum_{i=0}^{i^\star_1} k^{(3)}_{1,i,0} - 1 & \upsilon = 1,\\[3pt]
i^\star_l & 1 < \upsilon \le 4.
\end{cases}$$

Encoded by `encode_i18`/`decode_i18`:

* **Rank 1:** $i_{1,8}$ = the position of $(i^\star, f{=}0)$ *among the nonzero
  bitmap bits of tap 0* — i.e. `cumsum(bitmap[0])[i_star] - 1`, range
  $[0, \sum_i k^{(3)}_{1,i,0})$. Cheaper than naming $i^\star$ outright when only
  a few beams are active in tap 0.
* **Rank > 1:** $i_{1,8} = i^\star$ directly, range $\{0,\dots,2L-1\}$.

The standardized **fixed values for the strongest coefficient** (not reported):
$k^{(1)}_{l,\lfloor i^\star_l/L\rfloor} = 15$, $k^{(2)}_{l,i^\star_l,0} = 7$,
$k^{(3)}_{l,i^\star_l,0} = 1$, $c_{l,i^\star_l,0} = 0$. The **other**
polarization's reference $k^{(1)}_{l,(\lfloor i^\star_l/L\rfloor+1)\bmod 2}$ *is*
reported. `validate_r16` checks the decoded strongest position is present in the
bitmap and carries the fixed values.

---

## 7. Selection (`select`) — the full UE pipeline

This is the canonical Type II selection; R17/R18 mirror it. Per
[etype2_r16.py](../../src/nr_csi/codebooks/etype2_r16.py#L215):

1. **Targets.** `aligned_eigen_targets(H, rank)` — per-subband eigenvectors with
   sequential phase alignment.
2. **Spatial basis.** `select_group_and_beams` (regular) or `select_ps_initial`
   (PS); then `ls_coefficients` → $(v, N_3, 2L)$ complex coefficients.
3. **Per layer:**
   * Transpose to $(2L, N_3)$; pick the reference beam $i_\text{ref}$ (most
     energetic across subbands) and **rotate every subband** so its coefficient
     is real-positive. *Why:* sequential alignment leaves a residual phase chirp
     across frequency; rotating to a single beam's phase removes it so the delay
     taps don't smear.
   * **FFT to the delay domain:** `Ctap = fft(C, axis=1) / N3` (forward FFT, not
     `ifft` — matches the $e^{+j}$ precoder convention). Compute per-tap energy.
   * Roll the strongest tap to index 0, select $M_v$ taps (`_select_taps`),
     encode them (`encode_taps`), share $i_{1,5}$ across layers.
   * Pick the strongest coefficient $i^\star$ at tap 0, normalize the layer so
     $|x_{i^\star,0}| = 1$.
   * **Bitmap:** keep the $\min(K_0, 2L\cdot M_v)$ largest-magnitude coefficients,
     then drop any below half the smallest differential level
     (`> R16_DIFF_AMP[0]/2` — quantizing those upward would be worse than dropping
     them), and force the strongest in. Store `i17[l]` (transposed to
     $(M_v, 2L)$).
4. **Global budget.** `_enforce_total_budget` drops the globally weakest kept
   coefficients until $\sum_l K^{NZ}_l \le 2K_0$ (the strongest of each layer is
   protected).
5. **Quantize.** For each layer, set $i_{1,8}$; the strongest polarization's
   reference $k^{(1)}=15$, the other polarization's reference = nearest level of
   its max kept magnitude (≥ 1); then per nonzero coefficient quantize the
   differential amplitude (relative to the reference) and the 16-PSK phase; fix
   the strongest to $k^{(2)}{=}7, c{=}0$.

---

## 8. Codebook subset restriction & RI restriction (mostly unimplemented)

The spec defines two restriction mechanisms for eType II; neither is enforced by
the codebase.

**`n1-n2-codebookSubsetRestriction-r16` (CSR).** A bitmap $B = B_1 B_2$ that
(i) restricts the allowed SD beam-vector groups $g^{(k)}$ (as in §5.2.2.2.3) and
(ii) caps the **average coefficient amplitude** per restricted beam/polarization
via 2-bit fields $b_2^{(k,\cdot)}$ → maximum average amplitude $\gamma_{i+pL}$
(Table 5.2.2.2.5-6):

| 2-bit field | max average amplitude $\gamma_{i+pL}$ |
|:--:|:--:|
| 00 | 0 |
| 01 | $\sqrt{1/4}$ |
| 10 | $\sqrt{1/2}$ |
| 11 | 1 |

with the constraint
$\sqrt{\tfrac{1}{\sum_f k^{(3)}_{l,i+pL,f}}\sum_f k^{(3)}_{l,i+pL,f}
\big(p^{(1)}_{l,p}p^{(2)}_{l,i+pL,f}\big)^2} \le \gamma_{i+pL}$. Fields `01`/`10`
require the UE to advertise *amplitudeSubsetRestriction-r16 = supported*.

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** No `codebookSubsetRestriction`
> for eType II: neither the SD-group beam restriction nor the per-beam average-amplitude
> cap (Table 5.2.2.2.5-6 / *amplitudeSubsetRestriction-r16*) is modeled in
> [validate.py](../../src/nr_csi/codebooks/validate.py) or `select`.

**`typeII-RI-Restriction-r16`** (and `typeII-PortSelectionRI-Restriction-r16` for
PS): a 4-bit sequence $r_3 r_2 r_1 r_0$; $r_i = 0$ forbids reporting
$\upsilon = i+1$ layers.

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** RI restriction
> (`typeII-RI-Restriction-r16` / `typeII-PortSelectionRI-Restriction-r16`) is not
> enforced; `select`/`precoder` accept any rank in 1–4 (subject only to the
> param-combo's rank-3/4 support).

---

## 9. Overhead (`overhead_bits`)

```python
# spatial:
regular: i11 = ceil(log2(O1*O2)),  i12 = ceil(log2(C(N1*N2, L)))
PS:      i11 = ceil(log2(ceil(P/(2*d))))
# frequency (Algorithm 3):
N3 > 19: i15 = ceil(log2(2*Mv));  i16 = v * ceil(log2(C(2*Mv-1, Mv-1)))
N3 <=19:                          i16 = v * ceil(log2(C(N3-1, Mv-1)))
# coefficients:
i17 = v * 2L * Mv                 # the full bitmap (i_{1,7,l})
i18 = v * ceil(log2(2L))          # strongest indicator (i_{1,8,l})
i23 = 4 * v                       # the OTHER polarization's reference amplitude
i24 = 3 * (K_nz - v)              # differential amplitudes (strongest skipped)
i25 = 4 * (K_nz - v)              # 16-PSK phases (strongest skipped)
```

where `K_nz = i17.sum()` is the total number of reported coefficients across
layers ($K^{NZ} \le 2K_0$). Crucial observations:

* The **per-subband** R15 cost ($\propto N_3$) is gone. The frequency dependence
  now costs $i_{1,6}$ (a tap-index, $\propto \log_2\binom{N_3-1}{M_v-1}$ or
  $\binom{2M_v-1}{M_v-1}$) plus the bitmap — both far smaller than
  $N_3 \cdot 2L \cdot \log_2 N_\text{PSK}$.
* The coefficient payload scales with $K^{NZ}$ (the number of *kept* coefficients,
  $\le 2K_0$), not with $2L\cdot M_v$. The bitmap localizes the few significant
  delays.
* The $-v$ terms in $i_{2,4}/i_{2,5}$ are the one unreported strongest coefficient
  per layer; $i_{2,3}$ ($4v$) is the *other* polarization's reference (the
  strongest polarization's reference is the fixed 15).

> **L = 1 overhead inversion (erratum 4 in the repo [README](../../README.md)).**
> R16's fixed machinery (the tap-combination index and per-coefficient payload) is
> not amortized when there is only one beam, so at $L{=}1$ R16 can cost *more* than
> R15. R16 wins decisively for $L \ge 2$, where the delay compression pays off.
> (Note $L=1$ is not in the standardized table — combos start at $L=2$ — so this is
> a study artifact of the generalized `combo` override.)

---

## 10. Why `fft` and not `ifft`, and why per-subband phase alignment

Two implementation conventions are load-bearing and easy to get wrong (both noted
in the code and the project memory):

* The precoder regenerates subbands with $y_{t,l}^{(f)} = e^{+j2\pi t n/N_3}$. To
  *invert* that on the UE side you need a forward FFT scaled by $1/N_3$. NumPy's
  `ifft` uses $e^{+j}$ too but with a different normalization and an index flip;
  using it would place energy on $N_3 - n$ instead of $n$.
* The eigen targets carry a per-subband phase ambiguity. If you delay-transform
  them without first pinning a common phase reference per subband (rotate so one
  strong beam is real), the residual phase ramp across frequency spreads each
  delay across many taps, defeating the compression. R16 rotates to the strongest
  *beam*; R17/R18 do the analogous rotation in their domains.

---

## 11. Port-selection variant (§5.2.2.2.6)

`port_selection=True`: `_basis` uses `basis_ps` (consecutive ports $v_{i_{1,1}d+i}$,
stride $d$), `_scale()` returns 1, the spatial report collapses to a single
$i_{1,1}$ (no $i_{1,2}$), and the param table is restricted to rows 1–6
(`R16_PS_PARAM_COMBOS`). Everything in the frequency/coefficient machinery is
unchanged (§5.2.2.2.6 explicitly defers $M_v$, $M_\text{initial}$, $K_0$,
$i_{1,5}$, $i_{1,6,l}$, $i_{1,7,l}$, $i_{1,8,l}$, $i_{2,3/4/5,l}$ and the
amplitude/phase tables to §5.2.2.2.5).

* **$d$ / `portSelectionSamplingSize-r16`:** $d \in \{1,2,3,4\}$ and $d \le L$
  (validated at construction; out-of-range raises). $i_{1,1}$ selects the $2L$
  ports as $v_{i_{1,1}d+i}$, $i=0,\dots,L-1$ (mod $P_\text{CSI-RS}/2$), so
  $i_{1,1} \in \{0,\dots,\lceil P/(2d)\rceil - 1\}$ and costs
  $\lceil\log_2\lceil P/(2d)\rceil\rceil$ bits.

It is the natural choice when CSI-RS is already beamformed onto a handful of ports
(reciprocity). The equivalence $\text{eType II PS} \equiv \text{regular eType II
through a unitary port-expansion basis}$ is asserted in the test suite.

---

**Next:** [Chapter 5 — Further-enhanced Type II PS R17](05-fetype2-r17.md) makes
the port selection *free* (any $L$ ports via Algorithm 4) and shrinks the tap set
to $M\in\{1,2\}$ using uplink/downlink delay reciprocity. The multi-TRP coherent
joint transmission generalization is in [Chapter 8 — CJT](08-cjt.md).
