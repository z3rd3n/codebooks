# Chapter 7 — Release-19 refined codebooks

* **Spec:** TS 38.214 v19.x §5.2.2.2.1a (refined Type I single-panel),
  §5.2.2.2.2a (refined Type I multi-panel), §5.2.2.2.5a / §5.2.2.2.9a /
  §5.2.2.2.11a (refined Type II family). Base clauses inherited by the refined
  Type II family: §5.2.2.2.5 (R16 Enhanced Type II), §5.2.2.2.7 (R17 further-
  enhanced port selection), §5.2.2.2.10 (R18 predicted Type II).
* **Code:** [refined_type1_r19.py](../../src/nr_csi/codebooks/refined_type1_r19.py)
  — class `RefinedType1SinglePanelCodebook`;
  [refined_r19.py](../../src/nr_csi/codebooks/refined_r19.py) — classes
  `RefinedEType2Codebook`, `RefinedFeType2PortSelectionCodebook`,
  `RefinedPredictedEType2Codebook`; config in
  [config.py](../../src/nr_csi/config.py) — `SUPPORTED_N1N2_R19`.
* **Ranks:** 1–8 (refined Type I single-panel, `modeA`); 1–4 (refined Type II).
* **Prereq:** the corresponding base chapters
  ([1](01-type1-single-panel.md), [4](04-etype2-r16.md),
  [5](05-fetype2-r17.md), [6](06-etype2-doppler-r18.md)).

> ⚠️ **Spec-source note.** The bundled spec markdown
> [specs/38214-i00.md](../../specs/38214-i00.md) is **TS 38.214 Release-18
> ("i00")**. It does **not** contain the Release-19 *refined* clauses
> (5.2.2.2.1a / 5.2.2.2.2a / 5.2.2.2.5a / 5.2.2.2.9a / 5.2.2.2.11a). Everything
> about the Release-19 *structure* in this chapter is documented from the code,
> the module docstrings, and the Release-19 standard (v19.3.0 / "j30") — it is
> **not** cross-checked line-by-line against the bundled markdown. Where this
> chapter cites a base clause that **is** in the file (5.2.2.2.5 at line 9794,
> 5.2.2.2.7 at line 10745, 5.2.2.2.10 at line 12411, the R15 Type I base
> 5.2.2.2.1 at line 8918), those line anchors are real.

Release 19 targets **very large arrays** — 48, 64, and 128 CSI-RS ports, built by
aggregating several CSI-RS resources, with geometries in Table 5.2.2.2.1a-1
(`SUPPORTED_N1N2_R19`, $O_1{=}O_2{=}4$). There are two distinct kinds of "refined"
codebook:

1. **Refined Type II** (§5.2.2.2.5a/9a/11a) — the *same reconstruction* as R16/R17/R18,
   just on the larger geometries with a few extra configuration guards. Implemented
   as thin subclasses.
2. **Refined Type I** (§5.2.2.2.1a single-panel, §5.2.2.2.2a multi-panel) — a
   *structurally new* beam selection (especially for high rank). Only the
   single-panel `modeA` variant is implemented; see the not-implemented flags
   below.

---

## The Release-19 large-array geometries (Table 5.2.2.2.1a-1)

All Release-19 refined codebooks share one new ingredient: the dual-polarized UPA
geometries for 48/64/128 CSI-RS ports, with **fixed** oversampling $O_1=O_2=4$.
These are the `SUPPORTED_N1N2_R19` rows in
[config.py](../../src/nr_csi/config.py):

| $P_\text{CSI-RS}$ | $(N_1,N_2)$ | $(O_1,O_2)$ | $(G_1,G_2)=(N_1 O_1, N_2 O_2)$ |
|---|---|---|---|
| 48  | (8, 3)  | (4, 4) | (32, 12) |
| 48  | (6, 4)  | (4, 4) | (24, 16) |
| 64  | (16, 2) | (4, 4) | (64, 8)  |
| 64  | (8, 4)  | (4, 4) | (32, 16) |
| 128 | (16, 4) | (4, 4) | (64, 16) |
| 128 | (8, 8)  | (4, 4) | (32, 32) |

with $P_\text{CSI-RS}=2 N_1 N_2$ (single panel, $N_g=1$). `AntennaConfig`
merges this table into the legacy `SUPPORTED_N1N2` map, so
`AntennaConfig.standard(8, 4)` "just works" and `n_beams` returns
$(G_1,G_2)=(N_1 O_1, N_2 O_2)$:

```python
from nr_csi.config import AntennaConfig, SUPPORTED_N1N2_R19
ant = AntennaConfig.standard(8, 4)   # 64-port R19 array, O1=O2=4
ant.P            # 64
ant.n_beams      # (32, 16)
```

Because oversampling is locked at $(4,4)$, the oversampled grid is large
(e.g. 1024 beams for $(8,8)$), which is exactly why high-rank beam selection is
worth refining (Part B).

---

## Part A — Refined Type II family (subclasses)

Each §5.2.2.2.x*a* clause states explicitly "The codebook is defined as in
Clause 5.2.2.2.x", i.e. the **precoder reconstruction is identical** to the
corresponding R16/R17/R18 codebook. The only differences are a *configuration
envelope*:

* the larger $(N_1, N_2)$ geometries (handled by `AntennaConfig` via
  `SUPPORTED_N1N2_R19`);
* `paramCombination` restrictions (the high-overhead $L{=}6$ rows are barred when
  any rank $>2$ is permitted, or when $R = 2$);
* a Release-19 rank (RI) restriction (`typeII-RI-Restriction-r19` and friends).

So these classes subclass the existing implementations and add only guards.
The shared helpers — `_require_r19_array`, `_as_ri_restriction`,
`_rank_gt2_allowed` — live at the top of
[refined_r19.py](../../src/nr_csi/codebooks/refined_r19.py). `_as_ri_restriction`
normalizes the RI bitmap to a 4-bit boolean $r=[r_0,r_1,r_2,r_3]$ (rank $v$ ↔
$r_{v-1}$); `_rank_gt2_allowed` is `bool(r[2] or r[3])`.

### `RefinedEType2Codebook` (§5.2.2.2.5a) — refines R16

```python
class RefinedEType2Codebook(R16Type2Codebook):
    SUPPORTED_PORTS = (48, 64, 128)
    def __init__(self, antenna, N3, param_combination=4, R=1, ri_restriction=None):
        _require_r19_array(antenna, self.SUPPORTED_PORTS)   # geometry + port guard
        r = _as_ri_restriction(ri_restriction)
        # paramCombination 7/8 (L=6) require R=1 and ranks 3,4 disallowed:
        if param_combination in (7, 8) and (R == 2 or _rank_gt2_allowed(r)):
            raise ValueError(...)
        super().__init__(antenna, N3, param_combination=param_combination, R=R)
```

* **Ports:** 48, 64, 128.
* **`paramCombination-r19`:** uses the same `R16_PARAM_COMBOS` table as
  [Chapter 4](04-etype2-r16.md). Rows **7 and 8** are the $L{=}6$ rows
  ($\beta=\tfrac12,\tfrac34$, no rank 3/4 support); they are barred whenever
  `R == 2` **or** the RI restriction permits any rank $>2$. With
  `ri_restriction=[1,1,0,0]` (only ranks 1–2 allowed) and `R=1`, combos 7/8 are
  accepted and yield $L=6$.
* **RI restriction:** `select` raises if the requested `rank` has its bit cleared
  (`typeII-RI-Restriction-r19`, $r_{\text{rank}-1}=0$).
* **Reconstruction, selection, overhead:** entirely inherited from
  [R16](04-etype2-r16.md). Nothing about the precoder changes for 48/64/128 ports.

### `RefinedFeType2PortSelectionCodebook` (§5.2.2.2.9a) — refines R17

```python
class RefinedFeType2PortSelectionCodebook(R17Type2Codebook):
    SUPPORTED_PORTS = (48, 64)
    def __init__(self, antenna, N3, param_combination=7, N_window=4, ri_restriction=None):
        _require_r19_array(antenna, self.SUPPORTED_PORTS)
        if param_combination == 8:                       # UE not expected to be configured with it
            raise ValueError("paramCombination-r19=8 is not supported ...")
        super().__init__(antenna, N3, param_combination=param_combination, N_window=N_window)
```

* **Ports:** 48 and 64 **only** (128-port `(8,8)`/`(16,4)` arrays are rejected —
  the R17 further-enhanced port-selection codebook is not defined for them in
  R19).
* **`paramCombination-r19`:** the R17 `(M, α, β)` table from
  [Chapter 5](05-fetype2-r17.md). **Row 8** ($M=2,\alpha=1,\beta=\tfrac34$) is
  barred — the UE is not expected to be configured with it.
* **`N_window`:** the R17 port-selection window, default 4 (∈ {2,4}).
* **RI restriction:** `select` enforces `typeII-PortSelectionRI-Restriction-r19`.
* **Reconstruction:** = [R17](05-fetype2-r17.md).

### `RefinedPredictedEType2Codebook` (§5.2.2.2.11a) — refines R18

```python
class RefinedPredictedEType2Codebook(R18Type2Codebook):
    SUPPORTED_PORTS = (48, 64, 128)
    def __init__(self, antenna, N3, N4=4, param_combination=3, R=1, ri_restriction=None):
        _require_r19_array(antenna, self.SUPPORTED_PORTS)
        r = _as_ri_restriction(ri_restriction)
        # paramCombination-Doppler 8, 9 (L=6) barred when rank>2 permitted or R=2:
        if param_combination in (8, 9) and (R == 2 or _rank_gt2_allowed(r)):
            raise ValueError(...)
        super().__init__(antenna, N3, N4=N4, param_combination=param_combination, R=R, ri_restriction=r)
```

* **Ports:** 48, 64, 128.
* **`paramCombination-Doppler-r19`:** the R18 `R18_PARAM_COMBOS` table from
  [Chapter 6](06-etype2-doppler-r18.md). **Rows 8 and 9** are the $L{=}6$ rows
  (no rank 3/4 support); barred when `R == 2` or rank $>2$ is permitted, accepted
  with `ri_restriction=[1,1,0,0]` and `R=1` (then $L=6$).
* **`N4`:** the temporal/Doppler basis length, ∈ {1,2,4,8}. As in R18, `N4=1`
  degenerates the codebook to the R16 Enhanced Type II ($Q=1$); otherwise $Q=2$.
* **RI restriction:** the R18 Doppler RI restriction is threaded straight through
  to the base class.
* **Reconstruction:** = [R18 Doppler](06-etype2-doppler-r18.md).

> **Why subclasses suffice.** Because the large-array geometry is fully absorbed
> by `AntennaConfig` (the DFT bases, combinatorial codecs, and coefficient
> machinery are all parameterized by $N_1, N_2, P$), nothing in the
> reconstruction needs to change for 48/64/128 ports. The refinement is purely a
> *configuration envelope*: which geometries, which param rows, which ranks. This
> is the single best illustration in the codebase of how a clean parameterization
> turns "support a bigger array" into a one-line table entry.

### Refined Type II options guarded out / not exposed

* The **$L{=}6$ param rows** are conditionally barred (above) — *implemented as
  guards*, not absent.
* The **128-port** geometries for the **refined feType II PS** codebook
  (§5.2.2.2.9a) are rejected by `SUPPORTED_PORTS = (48, 64)`. **[not implemented]**
  (matches the R19 standard, which does not define §5.2.2.2.9a for 128 ports).
* `RefinedFeType2PortSelectionCodebook` does not take an `R` argument (the R17 PS
  codebook is single-precoder-per-subband in this implementation); `R=2`
  behaviour is therefore not exercised for the refined PS variant. **[not implemented]**

---

## Part B — Refined Type I single-panel (§5.2.2.2.1a, `modeA`)

This one is genuinely new. It is `typeI-SinglePanel-r19`, ranks 1–8, for the large
arrays, **`codebookMode 'modeA'`**. The per-layer column **patterns** are the same
as R15 Type I (each column is $[b;\,\pm\phi_n b]$ for a selected beam $b$,
$\phi_n = e^{j\pi n/2}$), but the **beam selection** is refined — and for high rank
it becomes *independent* per companion beam.

### Configuration & PMI

```python
RefinedType1SinglePanelCodebook(antenna, N3=1, selection_snr_db=10.0)
# requires (N1,N2) in SUPPORTED_N1N2_R19, Ng == 1

@dataclass
class RefinedType1PMI:
    rank: int
    i11, i12: int            # base beam (oversampled grid)
    i2:  np.ndarray          # (N3,) per-subband co-phasing
    i13: int | None          # ranks 2-8 beam-offset selector
    i112: tuple[int, ...]    # ranks 5-8: companion horizontal indices i_{1,1,j}
    i122: tuple[int, ...]    # ranks 5-8: companion vertical indices i_{1,2,j}
```

`selection_snr_db` only drives the UE-side co-phasing search (`select`); it has no
effect on the gNB-side reconstruction. `Ng` must be 1 (single panel).

### Beam selection by rank (`_beams`)

The base beam is $(l,m)=(i_{1,1},i_{1,2})$ on the oversampled grid
$(G_1,G_2)=(N_1 O_1, N_2 O_2)$.

* **Rank 1:** the base beam only.
* **Ranks 2–4:** base beam plus **one** companion at a fixed offset $(k_1,k_2)$
  from the `i13_lowrank` table (Table 5.2.2.2.1a-3) — i.e. companion
  $(l+k_1,\,m+k_2)$, exactly the R15 Type I "orthogonal companion" idea.

  **`i13_lowrank` (Table 5.2.2.2.1a-3, the $(k_1,k_2)$ offsets):**

  | $i_{1,3}$ | rank 2 | ranks 3,4 |
  |---|---|---|
  | 0 | $(0,0)$    | $(O_1,0)$    |
  | 1 | $(O_1,0)$  | $(0,O_2)$    |
  | 2 | $(0,O_2)$  | $(O_1,O_2)$  |
  | 3 | $(2O_1,0)$ | $(2O_1,0)$   |

  So $i_{1,3}$ is a **2-bit** field for ranks 2–4.

* **Ranks 5–8:** the companions are selected **independently**. `i13_highrank`
  maps $i_{1,3} \in \{0,1\}$ to a parametrization $((o_1,k_1),(o_2,k_2))$:

  | $i_{1,3}$ | $(o_1,k_1)$ | $(o_2,k_2)$ |
  |---|---|---|
  | 0 | $(O_1,\ i_{1,1}\bmod O_1)$ | $(1,0)$ |
  | 1 | $(1,0)$ | $(O_2,\ i_{1,2}\bmod O_2)$ |

  and each companion $j$ is

  $$l^{(j)} = o_1\, i_{1,1,j} + k_1, \qquad m^{(j)} = o_2\, i_{1,2,j} + k_2,$$

  with $i_{1,1,j}, i_{1,2,j}$ reported per companion. This is the refinement: high
  rank no longer forces companions onto a rigid offset grid; the UE picks each
  one (subject to an orthogonality constraint, below). The number of independent
  companions is `_N_EXTRA = {5:2, 6:2, 7:3, 8:3}`, so the **total beam count** is
  3 (ranks 5–6) or 4 (ranks 7–8) — base + companions.

  The per-companion index *ranges* depend on which axis is the orthogonal one
  (Table 5.2.2.2.1a-2):

  | $i_{1,3}$ | $i_{1,1,j}$ range (`hi1`) | $i_{1,2,j}$ range (`hi2`) |
  |---|---|---|
  | 0 | $\{0,\dots,N_1-1\}$ (orthogonal axis) | $\{0,\dots,N_2 O_2-1\}$ (free) |
  | 1 | $\{0,\dots,N_1 O_1-1\}$ (free) | $\{0,\dots,N_2-1\}$ (orthogonal axis) |

  i.e. one axis uses the *non-oversampled* range $N$ (orthogonal beams), the other
  uses the *full oversampled* range $N\cdot O$.

### Reconstruction (`_w_at`, `_COLUMN_PATTERNS`)

`_COLUMN_PATTERNS[rank]` (Table 5.2.2.2.1a-4) is the per-rank list of columns, each
a tuple `(beam_index, phi_top, sign_top, phi_bot, sign_bot)` describing a column

$$\begin{bmatrix} s_t\,(\phi_n\text{ if }phi\_top\text{ else }1)\,b \\ s_b\,(\phi_n\text{ if }phi\_bot\text{ else }1)\,b \end{bmatrix},$$

where `beam_index` indexes the ordered beam list $[b_0,b_1,b_2,b_3]$. The
transcribed patterns (with $b_i$ the $i$-th selected beam, $\phi=\phi_n$):

| rank | columns |
|---|---|
| 1 | $[b_0;\,\phi b_0]$ |
| 2 | $[b_0;\,\phi b_0],\ [b_1;\,-\phi b_1]$ |
| 3 | $[b_0;\,\phi b_0],\ [b_1;\,\phi b_1],\ [b_0;\,-\phi b_0]$ |
| 4 | $[b_0;\,\phi b_0],\ [b_1;\,\phi b_1],\ [b_0;\,-\phi b_0],\ [b_1;\,-\phi b_1]$ |
| 5 | $[b_0;\,\phi b_0],\ [b_0;\,-\phi b_0],\ [b_1;\,b_1],\ [b_1;\,-b_1],\ [b_2;\,b_2]$ |
| 6 | $[b_0;\,\phi b_0],\ [b_0;\,-\phi b_0],\ [b_1;\,\phi b_1],\ [b_1;\,-\phi b_1],\ [b_2;\,b_2],\ [b_2;\,-b_2]$ |
| 7 | $[b_0;\,\phi b_0],\ [b_0;\,-\phi b_0],\ [b_1;\,\phi b_1],\ [b_2;\,b_2],\ [b_2;\,-b_2],\ [b_3;\,b_3],\ [b_3;\,-b_3]$ |
| 8 | $[b_0;\,\phi b_0],\ [b_0;\,-\phi b_0],\ [b_1;\,\phi b_1],\ [b_1;\,-\phi b_1],\ [b_2;\,b_2],\ [b_2;\,-b_2],\ [b_3;\,b_3],\ [b_3;\,-b_3]$ |

Note that the high-rank patterns use $\phi_n$ only on the first beam(s) and plain
$\pm 1$ on the later companions — the orthogonality there comes from the beams
being mutually orthogonal, not from the co-phase. The whole matrix is scaled by
$1/\sqrt{\text{rank}\cdot P}$, so each $W^{(v)}$ has orthonormal columns up to that
$1/\sqrt{v}$ factor. Only the co-phase $n = i_2[t]$ is per subband; the number of
co-phase values is $N_{i_2} = 4$ at rank 1, else 2 (`_n_i2`).

### Selection (`select`)

A hybrid of the Type II energy search and the Type I rate search:

1. **Beam energy.** `_beam_energy` projects the aligned eigen targets
   (`aligned_eigen_targets`, the same rank-$v$ dominant-subspace targets used by
   the Type II selectors) onto the full oversampled grid → an energy map over
   $(G_1, G_2)$. It sums $|b^H x|^2$ over both polarizations, all subbands, and all
   $v$ layers. The base beam $(i_{1,1}, i_{1,2})$ is its argmax.
2. **Companions.**
   * Ranks 2–4: pick $i_{1,3}$ whose offset $(k_1,k_2)$ (modulo $G_1,G_2$) lands on
     the most energetic companion.
   * Ranks 5–8: `_select_highrank` greedily picks companions that are mutually
     **orthogonal** (the spec requires it, §5.2.2.2.1a). For $i_{1,3}=0$ each
     companion gets a *distinct* horizontal orthogonal-axis index in
     $\{0,\dots,N_1-1\}\setminus\{i_{1,1}/O_1\}$ (free vertical index over
     $N_2 O_2$); for $i_{1,3}=1$ the roles swap. It scores both $i_{1,3}$ values and
     keeps the more energetic set. If an array has too few orthogonal positions for
     the required number of companions, that $i_{1,3}$ branch is skipped.
3. **Co-phasing.** For each subband $t$, pick $i_2[t]$ by maximizing the per-layer
   log-det rate $\log\det(I + \rho\,W^H H_t^H H_t W)$ at `selection_snr_db` (the
   Type I-style SE metric).

The result is a `RefinedType1PMI`; `precoder` reconstructs $W$ of shape
`(1, N3, P, rank)`.

### Overhead (`overhead_bits`)

```python
bits["i11"] = ceil(log2(G1))
bits["i2"]  = N3 * (2 if rank==1 else 1)        # 2-bit co-phase at rank 1, else 1-bit
if G2 > 1:           bits["i12"] = ceil(log2(G2))
if rank in (2,3,4):  bits["i13"] = 2            # 2-bit offset selector
elif rank >= 5:
    bits["i13"]  = 1                            # 1-bit (i13 in {0,1})
    bits["i112"] = n_extra * ceil(log2(hi1))    # companion horizontal indices
    if hi2 > 1: bits["i122"] = n_extra * ceil(log2(hi2))   # companion vertical
```

where `n_extra = _N_EXTRA[rank]` and the index ranges `hi1`/`hi2` are the
$\{N\ \text{vs}\ N\cdot O\}$ pair from the table above (one axis is the orthogonal
"N" range, the other the full oversampled "N·O" range). The high-rank cost is the
$i_{1,1,j}/i_{1,2,j}$ companion indices — the price of independent beam selection,
which buys much better high-rank precoders on the large arrays than R15's rigid
offsets would. Worked example for $(8,4)$ ($G_1{=}32,G_2{=}16$): `i11`=5 bits,
`i12`=4 bits; rank-1 `i2`=$2N_3$; rank-5 adds 1-bit `i13` plus the companion
indices.

### Not implemented in this codebase

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** §5.2.2.2.1a
> `codebookMode 'modeB'` (the alternative per-layer / combinatorial beam selection
> for the refined Type I single-panel codebook). Only `modeA` is implemented.

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** §5.2.2.2.2a, the refined
> Type I **multi-panel** codebook (`typeI-MultiPanel-r19`) for the large arrays.
> No class exists; the module docstring states this explicitly.

---

## Summary of the Release-19 envelope

| Class | Base clause | Ports | Status / what R19 adds |
|---|---|---|---|
| `RefinedEType2Codebook` (§5.2.2.2.5a) | R16 §5.2.2.2.5 | 48/64/128 | implemented — large geometry + param (rows 7,8) + RI guards |
| `RefinedFeType2PortSelectionCodebook` (§5.2.2.2.9a) | R17 §5.2.2.2.7 | 48/64 | implemented — large geometry + RI guard, bars combo 8 |
| `RefinedPredictedEType2Codebook` (§5.2.2.2.11a) | R18 §5.2.2.2.10 | 48/64/128 | implemented — large geometry + param (rows 8,9) + RI guards |
| `RefinedType1SinglePanelCodebook` (§5.2.2.2.1a) | — (new) | 48/64/128 | implemented — `modeA` only; independent high-rank beam selection |
| §5.2.2.2.1a `modeB` | — (new) | 48/64/128 | **[not implemented]** |
| §5.2.2.2.2a refined Type I multi-panel | — (new) | 48/64/128 | **[not implemented]** |

The refined Type II classes show how cleanly the framework's parameterization
scales: a new array size is a config table entry, not a new codebook. The refined
Type I single-panel class shows the one place R19 actually changes the *structure*
— letting the UE pick high-rank companion beams independently instead of from a
fixed offset table. The refined Type I `modeB` and the refined Type I multi-panel
codebook complete the standardized R19 picture but are not implemented here.

---

This is the final family chapter. Back to the [index](README.md), or revisit the
[Foundations](00-foundations.md) for the shared machinery that ties all of them
together.
