# Chapter 0 — Foundations: the machinery every codebook shares

This chapter documents the building blocks that the codebook families reuse. Read
it once; the family chapters assume it.

This chapter is anchored to **TS 38.214 Release-18 ("i00")**, clause 5.2.2.2
("Precoding matrix indicator (PMI)") and the report-configuration clauses
5.2.1.4.2, 5.2.3 and 5.2.5 that surround it. Where the standard defines something
the codebase does not implement, it is described here and marked:

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.**

and inline options that exist in the spec but not the code are tagged
**[not implemented]**.

Contents:

1. [What a PMI codebook is](#1-what-a-pmi-codebook-is)
2. [The report-configuration context (codebookType, reportQuantity)](#2-the-report-configuration-context-codebooktype-reportquantity)
3. [Tensors and conventions](#3-tensors-and-conventions)
4. [The antenna model](#4-the-antenna-model-antennaconfig)
5. [Frequency-domain reporting granularity (N3)](#5-frequency-domain-reporting-granularity-n3)
6. [DFT bases: spatial, frequency, temporal](#6-dft-bases-spatial-frequency-temporal)
7. [Combinatorial index codecs (Algorithms 1–4)](#7-combinatorial-index-codecs-algorithms-14)
8. [Amplitude and phase quantization](#8-amplitude-and-phase-quantization)
9. [The `CodebookScheme` interface](#9-the-codebookscheme-interface)
10. [The shared UE-side algorithm](#10-the-shared-ue-side-algorithm)
11. [How "good" is measured: SE, SGCS, overhead](#11-how-good-is-measured-se-sgcs-overhead)
12. [Strongest-coefficient and reporting-reduction conventions](#12-strongest-coefficient-and-reporting-reduction-conventions)
13. [CSI report structure, Part 1/Part 2 and omission](#13-csi-report-structure-part-1part-2-and-omission)
14. [Codebook subset restriction](#14-codebook-subset-restriction)

---

## 1. What a PMI codebook is

In NR downlink, the base station (**gNB**) precodes data with a matrix
$W \in \mathbb{C}^{P \times v}$ — $P$ antenna ports, $v$ spatial layers
(the **rank**). To pick a good $W$ the gNB needs to know the channel, but the
channel lives at the user (**UE**). The UE therefore measures the channel from
CSI-RS, finds a good precoder, and feeds back a compact **index** into an
agreed-upon **codebook** of precoders. That index triple — rank indicator (RI),
PMI, channel-quality indicator (CQI) — is the entire downlink-CSI report. This
repository implements the **PMI** part.

A codebook is two algorithms over a shared parameterization:

* **`select`** (UE): channel $H \rightarrow$ PMI. *Not standardized* — the UE may
  use any method; the spec only fixes the set of representable precoders.
* **`precoder`** (gNB): PMI $\rightarrow W$. *Fully standardized* — given the
  reported indices, the reconstruction is a fixed formula.

The families differ in **how much structure they impose on $W$**, which trades
feedback overhead against fidelity:

* **Type I** quantizes a *direction*: one (or a few) DFT beam(s) plus a QPSK
  co-phasing between the two polarizations. A few bits; coarse.
* **Type II** quantizes a *linear combination*: $L$ DFT beams with per-beam,
  per-frequency complex coefficients. Many more bits; near-eigenvector fidelity.
  Each later Type II release adds another compression axis (delay, then Doppler)
  so the extra coefficients can be sent cheaply.

The full set of standardized codebook families lives in clauses 5.2.2.2.1 …
5.2.2.2.11 (see the next section for the `codebookType` ↔ clause map).

---

## 2. The report-configuration context (`codebookType`, `reportQuantity`)

A PMI never travels alone. It is one field of a **CSI report** configured by a
`CSI-ReportConfig` RRC information element (TS 38.214 clause 5.2.1.4). Two of its
parameters decide *which* codebook applies and *whether* a PMI is reported at
all.

### `codebookType` → clause → this repo

`codebookType` selects the family. The standardized values and the clause that
defines each precoder are:

| `codebookType` | Clause | Family | This repo |
|---|---|---|---|
| `typeI-SinglePanel` | 5.2.2.2.1 | Type I single-panel | implemented |
| `typeI-MultiPanel` | 5.2.2.2.2 | Type I multi-panel | **[not implemented]** |
| `typeII` | 5.2.2.2.3 | Type II (R15) | implemented |
| `typeII-PortSelection` | 5.2.2.2.4 | Type II port selection (R15) | **[not implemented]** |
| `typeII-r16` | 5.2.2.2.5 | Enhanced Type II (R16) | implemented |
| `typeII-PortSelection-r16` | 5.2.2.2.6 | Enhanced Type II PS (R16) | implemented |
| `typeII-PortSelection-r17` | 5.2.2.2.7 | Further-enhanced Type II PS (R17) | implemented |
| `typeII-CJT-r18` | 5.2.2.2.8 | Enhanced Type II for CJT (R18) | **[not implemented]** |
| `typeII-CJT-PortSelection-r18` | 5.2.2.2.9 | Further-enhanced Type II PS for CJT (R18) | **[not implemented]** |
| `typeII-Doppler-r18` | 5.2.2.2.10 | Enhanced Type II for predicted PMI (R18) | implemented |
| `typeII-Doppler-PortSelection-r18` | 5.2.2.2.11 | Further-enhanced Type II PS for predicted PMI (R18) | **[not implemented]** |

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** Type I multi-panel
> (5.2.2.2.2), R15 Type II port selection (5.2.2.2.4), and the two **CJT**
> (coherent joint transmission, multi-TRP) families (5.2.2.2.8/.9). The
> multi-panel $(N_g, N_1, N_2)$ table *is* present in
> [config.py](../../src/nr_csi/config.py) (`SUPPORTED_NG_N1N2`, Table
> 5.2.2.2.2-1) but no multi-panel/CJT `precoder` exists. See the package index
> [codebooks/\_\_init\_\_.py](../../src/nr_csi/codebooks/__init__.py) for what is
> built.

The R19 "…a" sub-clauses (the refined Type I/II codebooks and the 48/64/128-port
arrays) are *not* in the i00 spec text used here; the `SUPPORTED_N1N2_R19`
geometries in [config.py](../../src/nr_csi/config.py) anticipate them but the R19
precoder refinements are out of scope for this chapter.

### `reportQuantity` → is a PMI reported?

`reportQuantity` (clause 5.2.1.4.2) lists the quantities the UE feeds back. Only
some of them contain a PMI:

| `reportQuantity` | Contains PMI? | Notes |
|---|---|---|
| `none` | no | UE reports nothing |
| `cri-RI-PMI-CQI` | **yes** | the canonical full report — RI, PMI, CQI |
| `cri-RI-LI-PMI-CQI` | **yes** | adds the **layer indicator** (LI) |
| `cri-RI-i1` | partial | only the wideband part $i_1$ of a Type I PMI; wideband only |
| `cri-RI-i1-CQI` | partial | $i_1$ plus a CQI computed over a *random* $i_2$ per PRG |
| `cri-RI-CQI` | no | non-PMI CSI; precoder assumed identity-scaled |
| `cri-RSRP` / `ssb-Index-RSRP` / `…-Index` | no | beam-management RSRP |
| `cri-SINR` / `ssb-Index-SINR` / `…-Index` | no | beam-management SINR |
| `tdcp` | no | time-domain channel properties |

This repository models only the **PMI** (the $i_1$/$i_2$ index machinery and the
$W$ reconstruction). RI, CQI, CRI and LI selection, the partial `cri-RI-i1`
reporting mode, and the non-PMI quantities are **not** modeled:

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** The `reportQuantity`
> machinery (RI/CQI/CRI/LI computation, the `cri-RI-i1` / `cri-RI-i1-CQI`
> wideband-only modes, non-PMI CSI). The package fixes `rank` as a `select`
> argument and returns a `PMI` dataclass; everything outside the PMI is the
> caller's concern. The **LI** (layer indicator) — clause 5.2.1.4.2: "indicates
> which column of the precoder of the reported PMI corresponds to the strongest
> layer of the codeword with the largest reported wideband CQI" — depends on a
> CQI the repo does not compute, so it is absent. **[not implemented]**

### `cqi-FormatIndicator` / `pmi-FormatIndicator`

Two more knobs set the frequency granularity of the report (clause 5.2.1.4):

* `pmi-FormatIndicator ∈ {widebandPMI, subbandPMI}`. With **wideband PMI** a
  single PMI covers the whole reporting band. With **subband PMI** (and >2 ports)
  a single wideband part $i_1$ is reported once, and a subband part $i_2$ is
  reported per subband. With exactly 2 ports, a full PMI is reported per subband.
* `cqi-FormatIndicator ∈ {widebandCQI, subbandCQI}` — analogous for the CQI.
* A UE is **not** expected to be configured with `pmi-FormatIndicator` for any of
  the enhanced (R16+) codebook types — those always use the frequency-compressed
  $N_3$-unit machinery instead of an explicit wideband/subband split.

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** The
> wideband-vs-subband PMI *format* distinction for Type I is not a separate code
> path; the repo always carries an $i_1$ (wideband) plus an $i_2$ per
> frequency-unit (`N3`), which matches subband PMI. There is no
> wideband-only-PMI mode. **[not implemented]**

---

## 3. Tensors and conventions

These conventions are stated in [base.py](../../src/nr_csi/codebooks/base.py) and
hold across all families.

**Channel** `H[slot, t, rx, port]`:

* `slot` — time/“interval” axis. Length 1 for every codebook *except* R18
  Doppler, where it is $N_4$ (the future intervals being predicted).
* `t = 0 .. N3-1` — the **PMI frequency unit** (a "subband" when $R=1$).
* `rx` — UE receive antennas $N_r$.
* `port` — gNB CSI-RS ports $P$, **polarization-major** for single panel
  (first half = pol 0, second half = pol 1).

**Precoder** `W[interval, t, port, layer]` with the normalization

$$\operatorname{tr}(W^H W) = 1 \quad \text{per } (\text{interval}, t).$$

Each layer column is unit-norm and the rank-$v$ matrix carries the spec's
$1/\sqrt{v}$ factor, so the *total* transmit energy is split equally across
layers and frozen to 1. The SE metric then multiplies by linear SNR $\rho$
(noise power 1), i.e. $\sqrt{\rho}\,W$ carries the physical transmit power.

**Spatial beam ordering** is *vertical-fastest* Kronecker:
$v_{m_1,m_2} = a_{m_1} \otimes u_{m_2}$ (horizontal $\otimes$ vertical). This
matters whenever a flat beam index $n$ is split into $(n_1, n_2)$.

---

## 3. The antenna model (`AntennaConfig`)

[config.py](../../src/nr_csi/config.py) — `AntennaConfig` describes a
dual-polarized uniform planar array (UPA):

| Field | Meaning |
|---|---|
| `N1`, `N2` | ports per polarization in the horizontal / vertical dimension |
| `O1`, `O2` | DFT **oversampling** factors (beam grid is finer than the array) |
| `Ng` | number of panels (1 = single panel; 2 or 4 = multi-panel) |

Derived properties:

* `P = 2 · Ng · N1 · N2` — total CSI-RS ports (the leading 2 is the two
  polarizations).
* `n_ports_per_pol = N1 · N2`.
* `n_beams = (N1·O1, N2·O2)` — the size of the oversampled DFT beam grid
  $G_1 \times G_2$.

The legal $(N_1,N_2)\rightarrow(O_1,O_2)$ pairs are tabulated in
`SUPPORTED_N1N2` (TS 38.214 Table 5.2.2.2.1-2), with multi-panel combos in
`SUPPORTED_NG_N1N2` and the large Release-19 arrays (48/64/128 ports) in
`SUPPORTED_N1N2_R19`. `strict=True` enforces them; `strict=False` allows
experimental geometries (used by some figures). Construct the standard config
with `AntennaConfig.standard(N1, N2[, Ng])`.

```python
ant = AntennaConfig.standard(4, 2)   # N1=4, N2=2 -> O1=O2=4, P = 2*4*2 = 16 ports
ant.n_beams                          # (16, 8)  oversampled grid
```

---

## 4. Frequency-domain reporting granularity (N3)

A wideband carrier is divided into **subbands**. The PMI is reported per *PMI
frequency unit*; there are

$$N_3 = (\text{number of CQI subbands}) \times R,$$

where $R \in \{1, 2\}$ is `numberOfPMI-SubbandsPerCQI-Subband`
(`SubbandConfig`). Larger $N_3$ = finer frequency resolution = more accurate but
more expensive feedback. Helpers:

* `n3_for_bwp(n_rb, subband_size, R)` maps a bandwidth-part size (in resource
  blocks) and a chosen subband size to $N_3$ (Table 5.2.2.2.1.2-1 / "tabCSS").
  Worked example from the paper: 273 RB at subband size 16 → 18 subbands.
* `m_v(p_v, N3, R) = ceil(p_v · N3 / R)` — the number of **delay taps** the
  frequency-compressing codebooks keep (R16/R17/R18). $p_v$ is a per-rank
  fraction from the parameter-combination tables.

The **parameter-combination tables** `R16_PARAM_COMBOS`, `R17_PARAM_COMBOS`,
`R18_PARAM_COMBOS` are RRC-signalled presets of $(L, p_v, \beta)$ /
$(M, \alpha, \beta)$ controlling how many beams, taps, and coefficients the
codebook may use. They are introduced in the relevant family chapters.

---

## 5. DFT bases: spatial, frequency, temporal

[utils/dft.py](../../src/nr_csi/utils/dft.py). All three compression domains use
oversampled DFT vectors.

### Spatial beam $v_{m_1,m_2}$

A length-$N_1$ horizontal steering vector and length-$N_2$ vertical one, with
oversampling:

$$a_{m_1}[k] = e^{\,j 2\pi m_1 k /(O_1 N_1)},\ k=0..N_1{-}1, \qquad
  u_{m_2}[k] = e^{\,j 2\pi m_2 k /(O_2 N_2)},\ k=0..N_2{-}1,$$

$$v_{m_1,m_2} = a_{m_1} \otimes u_{m_2} \in \mathbb{C}^{N_1 N_2}.$$

Entries are unit-modulus, so $\lVert v\rVert^2 = N_1 N_2$. Code:
`steering(N, O, idx)` and `spatial_beam(cfg, m1, m2)`. `spatial_grid(cfg)`
returns all $G_1 \times G_2$ beams at once.

**Orthogonal groups.** The oversampled grid contains $O_1 O_2$ disjoint sets of
$N_1 N_2$ *mutually orthogonal* beams. Group $(q_1, q_2)$, $q_i \in \{0..O_i{-}1\}$,
consists of the beams with indices $m_i = O_i n_i + q_i$ — see
`beam_index` and `orthogonal_group`. Type II selects its $L$ beams from a single
such group, which is what makes $W_s^H W_s$ block-diagonal and the coefficient
fit a clean least-squares projection. Note `orthogonal_group` rows have norm
$\sqrt{N_1 N_2}$; `unitary_peb` rescales them to an orthonormal "port-expansion
basis" $F$ used when evaluating port-selection codebooks in the beam domain.

### Frequency (delay) basis $y$

$$y_t^{(f)} = e^{\,j 2\pi t\, n_3^{(f)} / N_3}, \quad t = 0..N_3{-}1.$$

`freq_basis(N3, n3)`. Selecting a few delay taps $\{n_3^{(f)}\}$ and a small
coefficient per tap reconstructs the whole frequency response — this is the R16+
"delay-domain" compression.

### Temporal (Doppler) basis $z$

$$z_\iota^{(\tau)} = e^{\,j 2\pi \iota\, n_4^{(\tau)} / N_4}, \quad \iota = 0..N_4{-}1.$$

`time_basis(N4, n4)` (identical math to `freq_basis`). R18 selects a few Doppler
shifts to predict the precoder across $N_4$ slot intervals.

> **Sign convention (important, see [conventions memory]).** The precoder uses
> $e^{+j2\pi t n/N_3}$, so the UE's delay-domain coefficients are computed with a
> forward FFT scaled by $1/N_3$ (`np.fft.fft(C)/N3`), **not** `ifft` — `ifft`
> would flip the tap indices. The same holds for the Doppler axis.

---

## 6. Combinatorial index codecs (Algorithms 1–4)

[utils/combinatorics.py](../../src/nr_csi/utils/combinatorics.py). Every codebook
that selects "$k$ things out of $n$" (beams, restricted groups, delay taps,
ports) encodes the *sorted* selection as a single integer with the **combinatorial
number system**:

$$\text{index} = \sum_{i=0}^{k-1} \binom{\,n - 1 - n_i\,}{\,k - i\,},
\qquad n_0 < n_1 < \dots < n_{k-1}.$$

`combo_to_index(indices, n_total)` and `index_to_combo(index, n_total, k)` are the
generic encode/decode; the four protocol algorithms are thin wrappers:

| Algorithm | Wrapper | Selects |
|---|---|---|
| 1 | `encode/decode_beam_combination` | $i_{1,2}$: $L$ beams out of $N_1 N_2$ (Type II) |
| 2 | `encode/decode_restriction_groups` | $\beta_1$: 4 restricted groups out of $O_1 O_2$ |
| 3 | `encode/decode_taps` | $i_{1,6,l}$: $M_v{-}1$ delay taps (R16/R18) |
| 4 | `encode/decode_ports` | $i_{1,2}$: $L$ ports out of $P/2$ (R17, errata-corrected) |

This is a **reverse-lexicographic** ordering: index 0 is the combination of the
*largest* indices. Two subtleties worth knowing:

* **Beam index flattening** is $n = N_1 n_2 + n_1$ (vertical major in the flat
  index even though the *beam vector* is vertical-fastest) — see
  `encode_beam_combination`.
* **Two-level delay indication.** When $N_3 > 19$, the $M_v$ taps must fit in a
  cyclic window of $2M_v$ taps. `encode_taps` then returns *both* $i_{1,6,l}$
  (window-relative tap combination, per layer) and $i_{1,5}$ ($M_\text{initial}$,
  the window origin, reported once for all layers). For $N_3 \le 19$ the
  $M_v{-}1$ non-strongest taps are encoded directly out of $N_3{-}1$.

The paper's Algorithm 4 has copy-paste typos (`C(x*, 4-k)`, `s_{k-1}`); the
corrected generic codec is used for all four — see the docstring.

---

## 7. Amplitude and phase quantization

[utils/quantization.py](../../src/nr_csi/utils/quantization.py). Type II
coefficients are quantized amplitude × PSK phase. The tables (exact values from
the verification run):

| Table | Symbol | Bits | Levels |
|---|---|---|---|
| `R15_WB_AMP` | $p^{(1)}$ wideband (R15) | 3 | $\{0,\sqrt{1/64},\sqrt{1/32},\dots,\sqrt{1/2},1\}$ — index 0 is **zero** |
| `R15_SB_AMP` | $p^{(2)}$ subband (R15) | 1 | $\{\sqrt{1/2},\,1\}$ |
| `R16_REF_AMP` | $p^{(1)}$ reference (R16+) | 4 | $2^{-(15-k)/4}$, $k=1..15$ (index 0 reserved → `NaN`) |
| `R16_DIFF_AMP` | $p^{(2)}$ differential (R16+) | 3 | $2^{-(7-k)/2}$, $k=0..7$ → $\{1/(8\sqrt2),\dots,1\}$ |

```
R15_WB_AMP  = [0,     0.125, 0.177, 0.25, 0.354, 0.5,  0.707, 1.0  ]   (k=0..7)
R16_REF_AMP = [nan,   0.088, 0.105, 0.125, ... , 0.707, 0.841, 1.0  ]   (k=0..15)
R16_DIFF_AMP= [0.088, 0.125, 0.177, 0.25, 0.354, 0.5,  0.707, 1.0  ]   (k=0..7)
```

* `quantize_amplitude(value, table)` → nearest-neighbour index. Reserved (`NaN`)
  entries are mapped to $-\infty$ so they are never chosen.
* `quantize_phase(angle, n_psk)` → $c \in \{0..N_\text{PSK}{-}1\}$ with
  $\phi_c = e^{\,j 2\pi c / N_\text{PSK}}$. $N_\text{PSK} = 4$ or $8$ for R15;
  $16$ for R16 and later.

Note `R16_DIFF_AMP` has **no zero level** (its minimum is $1/(8\sqrt2)\approx0.088$).
That is why R16+ uses an explicit **bitmap** to mark which coefficients are
nonzero, instead of relying on an amplitude of 0 (which only R15 has).

---

## 8. The `CodebookScheme` interface

[base.py](../../src/nr_csi/codebooks/base.py). The abstract base:

```python
class CodebookScheme(ABC):
    name: str
    def select(self, H, rank=1) -> PMI: ...          # UE
    def precoder(self, pmi) -> np.ndarray: ...        # gNB -> W[interval, t, port, layer]
    def overhead_bits(self, pmi) -> dict[str, int]: ...
    def total_overhead_bits(self, pmi) -> int:        # sum of the above
```

`normalize_columns(W)` unit-normalizes layer columns. The PMI itself is a small
per-family `@dataclass` of integer indices and integer arrays (never floats —
the report is a bitstream). Each family chapter lists its PMI fields.

Two cross-checks pin the implementation honest:

* **Compact / Tucker model** ([compact.py](../../src/nr_csi/codebooks/compact.py)) —
  an *independent* matrix/tensor expression of the same precoder
  ($W = \hat W_s W_c \hat W_f^T$ for R16, a Tucker product for R18). Tests assert
  it matches `precoder` direction-wise.
* **Serializer** ([serialize.py](../../src/nr_csi/codebooks/serialize.py)) —
  packs the PMI to the actual bitstream; `unpack(pack(pmi)) == pmi` and
  `len(pack(pmi)) == total_overhead_bits(pmi)`.

A **validator** ([validate.py](../../src/nr_csi/codebooks/validate.py)) runs at
the top of every `precoder`: it rejects malformed PMIs (wrong shapes, out-of-range
indices, broken strongest-coefficient conventions) so a gNB never reconstructs
garbage from a bad report.

---

## 9. The shared UE-side algorithm

[\_spatial.py](../../src/nr_csi/codebooks/_spatial.py) holds the UE helpers that
every Type II family reuses. The standardized part is `precoder`; `select` is one
defensible procedure. Its skeleton:

1. **Target precoders.** Compute the per-frequency-unit dominant right singular
   vectors of $H$ (`baselines.ideal.eigen_precoder`) — the ideal unquantized
   precoder. `aligned_eigen_targets` additionally removes the SVD's
   per-subband phase ambiguity by **sequential phase alignment** along $t$, so
   that a delay-domain (DFT) transform of the targets is meaningful.

2. **Spatial basis selection** (regular codebooks): `select_group_and_beams`
   tries every orthogonal group $(q_1,q_2)$ and picks the one whose best $L$
   beams capture the most target energy; the $L$ beams are encoded with
   Algorithm 1. Port-selection codebooks instead pick a starting port via
   `select_ps_initial` (consecutive ports) or strongest-port search (R17).

3. **Least-squares coefficients.** `ls_coefficients` projects the targets onto
   the selected basis (both polarizations) → complex coefficients per
   $(\text{layer}, t, \text{beam})$.

4. **Domain transforms and quantization** (per family): FFT to the delay domain
   and pick $M_v$ taps; (R18) FFT to the Doppler domain and pick $Q$ shifts;
   then nearest-neighbour quantize amplitudes/phases and prune to the bitmap
   budget.

Because the metric in step 2 is *energy capture*, while Type I instead maximizes
a **rate** (log-det) over candidate precoders, the Type I `select` looks
different — it enumerates beam/co-phase candidates and scores each by SE at a
nominal selection SNR (`selection_snr_db`, default 10 dB).

---

## 10. How "good" is measured: SE, SGCS, overhead

[metrics/se.py](../../src/nr_csi/metrics/se.py),
[metrics/overhead.py](../../src/nr_csi/metrics/overhead.py).

**Single-user spectral efficiency** (bits/s/Hz), averaging over slots/subbands:

$$\text{SE} = \mathbb{E}_t\!\left[\log_2 \det\!\left(I_v + \rho\,(H_t W_t)^H (H_t W_t)\right)\right].$$

`su_rate(H, W, rho)`. This is the metric a codebook ultimately optimizes;
inter-layer interference is treated exactly (joint decoding of the $v$ layers).
`mu_rate` adds inter-*user* interference for MU-MIMO comparisons.

**SGCS** (squared generalized cosine similarity) — the standard 3GPP AI/ML CSI
metric — and **NMSE** live in `metrics/similarity.py`; they measure how close the
reconstructed precoder direction is to the eigenvector target, independent of
SNR.

**Feedback overhead.** Each codebook's `overhead_bits(pmi)` returns a per-field
dict; `metrics/overhead.py` has closed-form transcriptions of the spec's bit
tables for parametric sweeps (`r15_bits`/`r16_bits`/`r18_bits`). The fairness
convention for comparing across releases is *equal time coverage*: R15/R16 need
$N_4$ separate reports to cover $N_4$ intervals, R18 covers them with one
predicted report.

---

## 11. Strongest-coefficient and reporting-reduction conventions

Type II codebooks save bits by **not transmitting redundant coefficients**. Two
patterns recur and are implemented identically across families:

**Strongest coefficient.** Each layer designates one "strongest" coefficient
(largest magnitude). It is rotated to be real-positive and used as the reference,
so it is *not reported*: its amplitude/phase are fixed by convention. The
validator enforces these fixed values.

* R15: the strongest index $i_{1,3,l}$ has wideband-amplitude index
  $k^{(1)} = 7$ (value 1), phase $c = 0$.
* R16/R17/R18: the strongest entry's polarization has reference index
  $k^{(1)} = 15$ (value 1); its differential amplitude $k^{(2)} = 7$ and phase
  $c = 0$. Its position is reported via $i_{1,8,l}$ (a "dual-mode" indicator —
  see the R16 chapter).

**Coefficient pruning / bitmap.** Only the strongest $K_0$ (R16+) coefficients
are kept per layer, with a global cap $\sum_l K^{NZ}_l \le 2K_0$, signalled by a
bitmap $i_{1,7,l}$. R15 instead uses the amplitude-0 level plus a $K^{(2)}$ cap on
how many *subband* amplitudes are sent (weak coefficients fall back to QPSK
phases). The exact rules are in each chapter.

These conventions are why the overhead formulas subtract $v$ (one unreported
strongest coefficient per layer) and why the serializer skips those fields.

---

**Next:** [Chapter 1 — Type I single-panel](01-type1-single-panel.md), the
simplest codebook and the structural template for the rest.
