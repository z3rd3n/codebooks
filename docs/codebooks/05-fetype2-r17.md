# Chapter 5 — R17 Further-enhanced Type II port selection (+ R18 predicted PS)

* **Spec:** TS 38.214 §5.2.2.2.7 *Further enhanced Type II port selection
  codebook* (R17); §5.2.2.2.11 *Further enhanced Type II port selection codebook
  for predicted PMI* (R18). Clause numbers/titles verified against
  `specs/38214-i00.md`.
* **Code:** [fetype2_r17.py](../../src/nr_csi/codebooks/fetype2_r17.py) — class
  `R17Type2Codebook`; [predicted_ps_r18.py](../../src/nr_csi/codebooks/predicted_ps_r18.py)
  — class `R18PredictedPortSelectionCodebook`. Validation in
  [validate.py](../../src/nr_csi/codebooks/validate.py) (`validate_r17`),
  bit-packing in [serialize.py](../../src/nr_csi/codebooks/serialize.py)
  (`pack_r17`/`unpack_r17`), parameter table in
  [config.py](../../src/nr_csi/config.py) (`R17_PARAM_COMBOS`).
* **`codebookType`:** `typeII-PortSelection-r17` (§5.2.2.2.7);
  `typeII-Doppler-PortSelection-r18` with $N_4=1$ (§5.2.2.2.11).
* **Ports:** $P_{\text{CSI-RS}} \in \{4, 8, 12, 16, 24, 32\}$ (antenna ports
  3000…2999+$P$).
* **Ranks:** 1–4 (the UE shall not report $\upsilon > 4$).
* **Compression:** **free** port selection ($L = K_1/2$ arbitrary ports) + a tiny
  delay window ($M \in \{1,2\}$ taps) exploiting delay reciprocity.
* **Prereq:** [Chapter 4 — eType II R16](04-etype2-r16.md) — the coefficient,
  bitmap, amplitude/phase, and budget machinery are identical; this chapter only
  changes the **spatial basis** and the **tap window**.

R16's port-selection variant picks *consecutive* ports with a fixed stride. R17
generalizes two things at once:

1. **Free port selection** — the UE may pick *any* $L$ of the $P/2$ ports per
   polarization (combinatorial, Algorithm 4), not a strided run.
2. **Reciprocity-based delays** — in TDD the gNB already knows the downlink delays
   from the uplink, so it pre-compensates them. The dominant delay then sits at
   tap 0 and only $M \in \{1,2\}$ taps in a tiny window $\{0..\min(N,N_3){-}1\}$
   are needed, instead of R16's $M_v$ taps spread across $N_3$.

The result is a codebook tuned for beamformed/reciprocity-based CSI-RS: very few
taps, freely chosen ports.

---

## 1. The reconstruction formula

For layer $l$, subband $t$ (TS 38.214 Table 5.2.2.2.7-3, the per-layer block
$W^l$):

$$W^l_{m,n_3,p^{(1)}_l,p^{(2)}_l,i_{2,5,l},t}
= \frac{1}{\sqrt{\gamma_{t,l}}}
\begin{bmatrix}
\sum_{i=0}^{L-1} v_{m^{(i)}}\, p^{(1)}_{l,0}\, \sum_{f=0}^{M-1} y_t^{(f)}\, p^{(2)}_{l,i,f}\, \varphi_{l,i,f} \\[4pt]
\sum_{i=0}^{L-1} v_{m^{(i)}}\, p^{(1)}_{l,1}\, \sum_{f=0}^{M-1} y_t^{(f)}\, p^{(2)}_{l,i+L,f}\, \varphi_{l,i+L,f}
\end{bmatrix},
\quad l = 1,2,3,4$$

$$\gamma_{t,l} = \sum_{i=0}^{2L-1}\left(p^{(1)}_{l,\lfloor i/L\rfloor}\right)^2
\left|\sum_{f=0}^{M-1} y_t^{(f)}\, p^{(2)}_{l,i,f}\, \varphi_{l,i,f}\right|^2$$

The $\upsilon$-layer precoder stacks the per-layer columns with the usual
$\tfrac{1}{\sqrt{\upsilon}}$ normalization (Table 5.2.2.2.7-3:
$\tfrac{1}{\sqrt2}$ for $\upsilon=2$, $\tfrac{1}{\sqrt3}$ for 3,
$\tfrac12$ for 4). The DFT taps and phase are

$$y_t^{(f)} = e^{\,j\frac{2\pi t\, n_3^{(f)}}{N_3}}, \qquad
\varphi_{l,i,f} = e^{\,j\frac{2\pi c_{l,i,f}}{16}}.$$

For coefficients with bitmap bit $k^{(3)}_{l,i,f}=0$ the spec sets
$p^{(2)}_{l,i,f}=0$ and $\varphi_{l,i,f}=0$ (the term drops out).

This is structurally identical to R16 (Chapter 4 §1) with two changes:

* $v_{m^{(i)}}$ is the **standard basis vector for the freely selected port**
  $m^{(i)}$ — a $P/2$-element column with a 1 at index $m^{(i)}$ and zeros
  elsewhere — **not** a DFT beam. There is **no $N_1N_2$** oversampling factor
  (this is the port domain).
* The frequency sum runs over $M \in \{1,2\}$ taps confined to
  $\{0..\min(N,N_3){-}1\}$.

In [`precoder`](../../src/nr_csi/codebooks/fetype2_r17.py) the per-layer block is
built as `B.T @ (x @ Y.T)` then normalized by $\sqrt{\gamma_{t,l}}$ and
$\sqrt{\upsilon}$, exactly matching the formula above.

---

## 2. Parameters: $\alpha$, $M$, $\beta$, $K_1$, $K_0$, $N$, $R$

The triple $(M, \alpha, \beta)$ is selected by the higher-layer parameter
`paramCombination-r17`. The **complete** Table 5.2.2.2.7-1, transcribed verbatim
(and mirrored in `R17_PARAM_COMBOS`):

| `paramCombination-r17` | $M$ | $\alpha$ | $\beta$ |
|---|---|---|---|
| 1 | 1 | 3/4 | 1/2 |
| 2 | 1 | 1   | 1/2 |
| 3 | 1 | 1   | 3/4 |
| 4 | 1 | 1   | 1   |
| 5 | 2 | 1/2 | 1/2 |
| 6 | 2 | 3/4 | 1/2 |
| 7 | 2 | 1   | 1/2 |
| 8 | 2 | 1   | 3/4 |

```python
@dataclass(frozen=True)
class R17ParamCombo:
    index: int
    M:     int        # number of delay taps, in {1, 2}
    alpha: Fraction   # fraction of ports used:  K1 = alpha * P
    beta:  Fraction   # coefficient-budget fraction
```

**Configuration restrictions on `paramCombination-r17`** (spec §5.2.2.2.7). The UE
is *not expected* to be configured with:

* combo **1 or 6** when $P_{\text{CSI-RS}} \in \{4,12\}$ (these give a
  non-integer or odd $K_1=\alpha P$, e.g. $\tfrac34\cdot4=3$);
* combo **7 or 8** when $P_{\text{CSI-RS}} = 32$ (would need $K_1=32$, i.e.
  all 32 ports with $M=2$ — too large a budget);
* combo **5** when $P_{\text{CSI-RS}}=4$ *and*
  `typeII-PortSelectionRI-Restriction-r17` has $r_i=1$ for any $i>1$.

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** These
> `paramCombination-r17` $\times$ $P$ restrictions are not enforced in
> `R17Type2Codebook.__init__`; the constructor only checks that
> $K_1=\alpha P$ is a positive even integer $\le P$ (which happens to reject the
> combo-1/6-at-$P{=}4$ case via the even-integer guard, but not the others).

Derived quantities (`__init__` / `K0` property):

* $K_1 = \alpha\,P_{\text{CSI-RS}}$ must be a positive even integer $\le P$; the
  number of selected ports **per polarization** is $L = K_1/2$. The precoder is
  built from $L + M$ vectors.
* If $\alpha = 1$, *all* ports are used ($m^{(i)}=i$, $i=0..P/2-1$) and $i_{1,2}$
  is **not reported**.
* The coefficient budget is $K_0 = \lceil \beta\, K_1\, M\rceil$, with the same
  global cap $\sum_l K^{\text{NZ}}_l \le 2K_0$ and per-layer
  $K^{\text{NZ}}_l \le K_0$.
* $N \in \{2,4\}$ (`valueOfN`, parameter `N_window`) — the size of the delay
  window for the second tap; only meaningful when $M=2$.
* $R \in \{1,2\}$ (`numberOfPMI-SubbandsPerCQI-Subband-r17`, defined in
  §5.2.2.2.5) sets how many PMI subbands map per CQI subband when $M=2$; $R=1$
  when $M=1$. $R$ feeds into the effective $N_3$ (number of FD units).

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** `R17Type2Codebook`
> takes $N_3$ directly and has no $R$ parameter; the subband-pairing factor $R$
> (and its effect on $N_3 = R\cdot N_{\text{SB}}$) is modeled only in the R18
> predicted-PS subclass (§9), where it is stored but not used in reconstruction.

---

## 3. The PMI

The R17 $i_1$/$i_2$ structure per rank $\upsilon$ (Table-of-indices in
§5.2.2.2.7), one $(i_{1,7,l}, i_{1,8,l})$ pair and one
$(i_{2,3,l}, i_{2,4,l}, i_{2,5,l})$ triple per layer:

| Component | Symbol | Meaning | Reported when |
|---|---|---|---|
| Port combination | $i_{1,2}$ | $L$ ports out of $P/2$ (Algorithm 4) | $\alpha < 1$ |
| Second-tap offset | $i_{1,6}$ | nonzero offset of $n_3^{(1)}$ | $M=2$ and $N=4$ |
| Bitmap (per layer) | $i_{1,7,l}$ | which $(i,f)$ are nonzero | unless $\upsilon\le2$ and $K^{\text{NZ}}=K_1 M\upsilon$ |
| Strongest indicator | $i_{1,8,l}$ | $K_1 f^\star + i^\star$ | always |
| Reference amplitude | $i_{2,3,l}$ | $[k^{(1)}_{l,0}, k^{(1)}_{l,1}]$ | other pol only |
| Differential amplitude | $i_{2,4,l}$ | $k^{(2)}_{l,i,f}$ | for $K^{\text{NZ}}-\upsilon$ coeffs |
| Phase | $i_{2,5,l}$ | $c_{l,i,f}$ | for $K^{\text{NZ}}-\upsilon$ coeffs |

```python
@dataclass
class R17Type2PMI:
    rank: int
    i12: int | None    # port combination (None when alpha == 1)
    i16: int | None    # second-tap offset (only when M==2 and effective N>2)
    i17: np.ndarray    # bitmap (v, M, K1), bool
    i18: list[int]     # strongest indicator = K1*f* + i*  per layer
    k1:  np.ndarray    # (v, 2) reference-amplitude indices k^(1)_{l,p}
    k2:  np.ndarray    # (v, M, K1) differential-amplitude indices k^(2)_{l,i,f}
    c:   np.ndarray    # (v, M, K1) 16-PSK phase indices c_{l,i,f}
```

Differences from R16's PMI: there is no orthogonal-group $(q_1,q_2)$ — the spatial
report is the **port combination** $i_{1,2}$ (Algorithm 4). The delay report
collapses to at most one offset $i_{1,6}$. There is no $i_{1,5}$
(intermediate-set / window) report.

---

## 4. Free port selection (Algorithm 4)

The $L$ selected ports are $m = [m^{(0)} \ldots m^{(L-1)}]$,
$m^{(i)} \in \{0,\ldots,P/2-1\}$, encoded into

$$i_{1,2} \in \left\{0,1,\ldots,\binom{P/2}{L}-1\right\}.$$

**Decoding** (spec algorithm, with $C(x,y)$ from Tables 5.2.2.2.5-4 and
5.2.2.2.7-2): $s_{-1}=0$; for $i=0,\ldots,L-1$ find the largest
$x^\star \in \{L-1-i,\ldots,P/2-1-i\}$ such that
$i_{1,2}-s_{i-1} \ge C(x^\star, L-i)$, set $e_i=C(x^\star,L-i)$,
$s_i=s_{i-1}+e_i$, $m^{(i)} = P/2-1-x^\star$.

**Encoding** (inverse):
$i_{1,2} = \sum_{i=0}^{L-1} C\!\left(\tfrac{P}{2}-1-m^{(i)},\, L-i\right)$,
with indices assigned so that $m^{(i)}$ increases with $i$. This is a
**reverse-lexicographic** ordering of the $\binom{P/2}{L}$ subsets.

`_basis(pmi)` builds the $L$ standard basis vectors:

* $\alpha = 1$: ports $0..L{-}1$ (all of them), no index reported.
* $\alpha < 1$: decode $i_{1,2}$ to $L$ ports out of $P/2$ via
  `decode_ports` (Algorithm 4 — the errata-corrected combinatorial codec; see
  [Foundations §6](00-foundations.md#6-combinatorial-index-codecs-algorithms-14)).

> **Algorithm-4 errata.** The originating paper's Algorithm 4 carries copy-paste
> typos (a hard-coded `C(x*, 4-k)` and a stray `s_{k-1}` from a 4-port special
> case). [combinatorics.py](../../src/nr_csi/utils/combinatorics.py) implements
> the spec's general $C(x^\star, L-i)$ / $s_{i-1}$ recursion correctly.

The combinatorial table $C(x,y)$ for $y\le9$ is Table 5.2.2.2.5-4 (shared with
R16); for $y>9$ (i.e. $L-i \in \{10,11,12\}$, reachable only at $P=24,32$) it is
the R17-specific **Table 5.2.2.2.7-2**, whose nonzero entries are:

| $x \backslash y$ | 10 | 11 | 12 |
|---|---|---|---|
| 10 | 1 | 0 | 0 |
| 11 | 11 | 1 | 0 |
| 12 | 66 | 12 | 1 |
| 13 | 286 | 78 | 13 |
| 14 | 1001 | 364 | 91 |
| 15 | 3003 | 1365 | 455 |

(all entries with $x<y$ are 0). The repo uses `math.comb` directly, so these
values are obtained analytically rather than from a hard-coded table.

On the UE side, the selected ports are simply the $L$ strongest by target energy:

```python
port_energy = sum(|targets|^2 over subbands & layers, both polarizations)
ports = sorted(argsort(port_energy)[::-1][:L])
i12 = encode_ports(ports, P)
```

This is strictly more flexible than R16 PS's strided run — the UE can pick a
scattered set of strong ports.

---

## 5. The tap window (`taps`)

The $M$ taps are $n_3 = [n_3^{(0)} \ldots n_3^{(M-1)}]$ with

$$n_3^{(f)} \in \begin{cases}
\{0\} & M=1 \\
\{0,1,\ldots,\min(N,N_3)-1\} & M=2
\end{cases}$$

and $f$ assigned so $n_3^{(f)}$ increases with $f$. `taps(pmi)` returns them, all
in $\{0..\min(N,N_3){-}1\}$:

| Condition | Taps | $i_{1,6}$ reported? |
|---|---|---|
| $M = 1$ | $[0]$ | no |
| $M = 2$, $\min(N, N_3) = 2$ | $[0, 1]$ | no |
| $M = 2$, $N = 4$ (and $\min(N,N_3) > 2$) | $[0,\, i_{1,6}{+}1]$ | yes — $i_{1,6}\in\{0,1,2\}$ |

Tap 0 is always present — that is the reciprocity assumption: the gNB has
pre-compensated the bulk delay, so the dominant energy is at delay 0
($n_3^{(0)}=0$ is the implicit reference). The UE only needs to point at the
*second* tap (if any) within the window. When $M=2$ and $N=4$, $i_{1,6}$ encodes
the **nonzero offset** $n_3^{(1)}-n_3^{(0)} = n_3^{(1)} \in \{1,2,3\}$, mapped in
increasing order (offset 1 → index 0). The window is tiny ($N \in \{2,4\}$)
compared with R16's $N_3$-wide tap search.

---

## 6. Amplitude, phase and the strongest coefficient

Identical machinery to R16 (Chapter 4 §4–5), restated with R17 ranges:

* **Reference amplitude** $k^{(1)}_{l,p} \in \{1,\ldots,15\}$ (4 bits),
  per polarization $p\in\{0,1\}$, mapped by Table 5.2.2.2.5-2 to $p^{(1)}_{l,p}$.
* **Differential amplitude** $k^{(2)}_{l,i,f} \in \{0,\ldots,7\}$ (3 bits),
  mapped by Table 5.2.2.2.5-3 to $p^{(2)}_{l,i,f}$.
* **Phase** $c_{l,i,f} \in \{0,\ldots,15\}$ (4 bits, **16-PSK**),
  $\varphi = e^{j2\pi c/16}$.

**Strongest coefficient.** Let $f^\star_l \in \{0,\ldots,M-1\}$ and
$i^\star_l \in \{0,\ldots,K_1-1\}$ index the strongest coefficient
$k^{(2)}_{l,i^\star,f^\star}$ of layer $l$. The strongest indicator is

$$i_{1,8,l} = K_1\, f^\star_l + i^\star_l \in \{0,\ldots,K_1 M - 1\}.$$

`divmod(i18, K1)` recovers $(f^\star, i^\star)$ at the gNB. The strongest entry is
fixed (not reported):
$k^{(1)}_{l,\lfloor i^\star/L\rfloor}=15$, $k^{(2)}_{l,i^\star,f^\star}=7$,
$k^{(3)}_{l,i^\star,f^\star}=1$, $c_{l,i^\star,f^\star}=0$. Only the **other**
polarization's reference amplitude
$k^{(1)}_{l,(\lfloor i^\star/L\rfloor+1)\bmod 2}$ is reported (hence
$i_{2,3,l}$ costs 4 bits, not 8). `validate_r17` checks all of these invariants.

---

## 7. The bitmap and budget

The per-layer bitmap $i_{1,7,l} = [k^{(3)}_{l,i,f}]$, $k^{(3)}_{l,i,f}\in\{0,1\}$,
of size $K_1 \times M$ marks the nonzero coefficients, with

$$K^{\text{NZ}}_l = \sum_{i=0}^{K_1-1}\sum_{f=0}^{M-1} k^{(3)}_{l,i,f} \le K_0,
\qquad K^{\text{NZ}} = \sum_{l=1}^{\upsilon} K^{\text{NZ}}_l \le 2K_0.$$

If $\upsilon \le 2$ **and** $K^{\text{NZ}} = K_1 M \upsilon$ (the bitmap is fully
populated), $i_{1,7,l}$ is **not reported**.

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** The "skip the bitmap
> when full" optimization is not applied: `pack_r17` always writes the full
> $\upsilon K_1 M$-bit bitmap, and `overhead_bits` always charges
> $i_{1,7}=\upsilon K_1 M$. (Functionally lossless — the bits are simply never
> omitted in the rare full-bitmap case.)

The reported coefficients are the $K^{\text{NZ}}-\upsilon$ non-strongest entries
with $k^{(3)}=1$; the remaining $K_1 M\upsilon - K^{\text{NZ}}$ entries are not
reported.

---

## 8. Reconstruction & selection in code

### 8.1 Reconstruction (`precoder`)

```python
B = basis(pmi)                              # (L, P/2) standard basis vectors
Y = freq_basis(N3, taps(pmi)).T            # (N3, M)
for l in range(rank):
    x     = layer_coefficients(pmi, l)     # (K1, M)  = p1*p2*phi*bitmap
    ct    = x @ Y.T                         # (K1, N3)
    gamma = sum(|ct|^2, axis=0)
    w     = [ B.T @ ct[:L] ; B.T @ ct[L:] ]
    W[0,:,:,l] = (w / sqrt(gamma)).T
W /= sqrt(rank)
```

Same shape as R16's loop, minus the $N_1N_2$ scale and with $M$ in place of $M_v$.

### 8.2 Selection (`select`)

The pipeline mirrors R16, in the port/tap domain:

1. **Targets** (`aligned_eigen_targets`) from the per-subband eigenvectors.
2. **Port selection** (§4): pick the $L$ strongest ports by accumulated energy
   across subbands/layers/both polarizations → $i_{1,2}$ → basis $B$ →
   `ls_coefficients(B, targets, 1.0)` (note scale 1: port domain) →
   $(\upsilon, N_3, K_1)$.
3. **Common second tap** (only $M{=}2$, $\min(N,N_3) > 2$): accumulate per-tap
   energy across layers (with per-layer phase alignment to the strongest port),
   pick the strongest in-window offset → $i_{1,6}$ (one offset shared by all
   layers, per the single $n_3$ vector in the spec).
4. **Per layer:** phase-align to the strongest port, **LS-project onto the
   selected taps** (`Ct = (C @ Y.conj().T) / N3` — the taps are orthogonal DFT
   columns), pick the strongest coefficient $(i^\star, f^\star)$, normalize it to
   1, build the bitmap (top-$K_0$ by magnitude, drop sub-floor entries below
   $p^{(2)}_{\min}/2$, force the strongest in).
5. **Global budget** (`_enforce_total_budget`, $\le 2K_0$ by dropping the
   weakest non-strongest coefficients), then **quantize** exactly as R16
   (reference per polarization, differential amplitude, 16-PSK phase; strongest
   fixed at $k^{(1)}=15$, $k^{(2)}=7$, $c=0$).

The strongest indicator is the flat index $i_{1,8} = K_1 f^\star + i^\star$.

---

## 9. RI restriction

`typeII-PortSelectionRI-Restriction-r17` is a 4-bit bitmap
$r_3 r_2 r_1 r_0$ ($r_0$ = LSB). When $r_i=0$, PMI/RI reporting must not
correspond to any precoder with $\upsilon = i+1$ layers. (The UE shall never
report $\upsilon>4$.)

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** `R17Type2Codebook`
> has no `rank_restriction` argument and `select`/`validate_r17` only check
> $1\le\upsilon\le4$. The RI-restriction bitmap **is** implemented for the R18
> predicted-PS subclass (§10), where it is enforced in `select`.

---

## 10. Overhead (`overhead_bits`)

```python
if alpha < 1:                          bits["i12"] = ceil(log2(C(P/2, L)))
if M==2 and min(N_window,N3) > 2:      bits["i16"] = ceil(log2(N_window - 1))
bits["i17"] = v * K1 * M               # bitmap
bits["i18"] = v * ceil(log2(K1 * M))   # strongest indicator
bits["i23"] = 4 * v                    # other-pol reference amplitude (k^(1), 4 bits)
bits["i24"] = 3 * (K_nz - v)           # differential amplitudes (k^(2), 3 bits)
bits["i25"] = 4 * (K_nz - v)           # 16-PSK phases (c, 4 bits)
```

Exact bit-widths / ranges:

| Field | Range | Width |
|---|---|---|
| $i_{1,2}$ | $0..\binom{P/2}{L}-1$ | $\lceil\log_2\binom{P/2}{L}\rceil$ (omitted if $\alpha=1$) |
| $i_{1,6}$ | $0..2$ | $\lceil\log_2(N-1)\rceil = 2$ (only $M=2$, $N=4$) |
| $i_{1,7,l}$ | bitmap | $K_1 M$ per layer |
| $i_{1,8,l}$ | $0..K_1 M-1$ | $\lceil\log_2(K_1 M)\rceil$ per layer |
| $i_{2,3,l}$ | $k^{(1)}\in\{1..15\}$ | 4 (other pol only; strongest pol fixed) |
| $i_{2,4,l}$ | $k^{(2)}\in\{0..7\}$ | 3 per reported coeff ($K^{\text{NZ}}-\upsilon$) |
| $i_{2,5,l}$ | $c\in\{0..15\}$ | 4 per reported coeff ($K^{\text{NZ}}-\upsilon$) |

Versus R16: no $(q_1,q_2)$/group term, $i_{1,2}$ only when $\alpha<1$, the
frequency report shrinks to a single optional offset $i_{1,6}$, and $M$ replaces
$M_v$ in the bitmap. For reciprocity channels this is a very small report.

> **Fidelity can *rise* with $P$.** Because $K_1 = \alpha P$ grows with the array,
> a larger array gives the UE more ports to combine *and* a proportionally larger
> coefficient budget — so R17's reconstruction accuracy can improve as $P$
> increases (analyzed in the repo's `fig_10` notes), the opposite of the usual
> "more antennas, harder to feed back" intuition.

---

## 11. R18 predicted port selection (`R18PredictedPortSelectionCodebook`)

TS 38.214 **§5.2.2.2.11** (`typeII-Doppler-PortSelection-r18`) reuses the R17
indices ($i_1$, $i_2$ as in §5.2.2.2.7) and the R17 reconstruction
(Table 5.2.2.2.7-3) for a **predicted PMI** at $N_4 = 1$: the PMI indicates
$N_3$ precoder matrices for **one** future slot interval of duration $d$ slots
(per Clause 5.2.1.4.2). It is the degenerate, single-instant case of the
Doppler codebook of [Chapter 6](06-etype2-doppler-r18.md) restricted to the
port-selection basis.

Configuration (same Table 5.2.2.2.7-1 and combination restrictions as §5.2.2.2.7):

* $(M,\alpha,\beta)$ from `paramCombination-Doppler-PS-r18`;
* $N \in \{2,4\}$ from `valueOfN-Doppler-r18` (when $M=2$);
* $R \in \{1,2\}$ from `numberOfPMI-SubbandsPerCQI-Subband-Doppler-PS-r18`
  (when $M=2$; $R=1$ when $M=1$) — $R$ and the corresponding $N_3$ per
  §5.2.2.2.5;
* $N_4 = 1$ fixed by higher-layer `N4`;
* `typeII-Doppler-PS-RI-Restriction-r18` — the same 4-bit
  $r_3 r_2 r_1 r_0$ RI bitmap, $\upsilon \le 4$.

The class is a thin subclass of `R17Type2Codebook`:

```python
class R18PredictedPortSelectionCodebook(R17Type2Codebook):
    name = "R18 FeType II PS predicted PMI"
    N4 = 1
    # adds: R in {1,2} (with R=2 forbidden when M==1) and a 4-bit rank_restriction
```

It adds only configuration guards — the subband-pairing factor $R$ (with $R=2$
disallowed when $M{=}1$, matching the spec) and a `rank_restriction` bitmap that
`select` enforces (rejecting a requested rank whose bit is 0). Reconstruction and
selection are inherited unchanged from R17.

> 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** $R$ is validated but
> not applied: the predicted-PS class takes $N_3$ directly and does not derive
> $N_3$ from $R$, so the subband-pairing semantics of $R=2$ are not modeled in
> reconstruction.

The *full* Doppler codebook with $N_4 > 1$ (a genuine multi-slot prediction over
the temporal DFT axis) is the regular eType II Doppler of
[Chapter 6](06-etype2-doppler-r18.md); this PS variant predicts only a single
interval.

> **Multi-TRP CJT port selection.** The coherent-joint-transmission port-selection
> codebook (`typeII-CJT-PortSelection-r18`, §5.2.2.2.8) — multiple TRPs feeding
> back a joint port-selection precoder — is covered in
> [Chapter 8 — CJT](08-cjt.md).

---

## 12. R17 vs R16 PS in one line

R16 PS: strided ports, $M_v$ taps spread over $N_3$, fixed stride $d$.
R17 PS: freely chosen ports, $M\in\{1,2\}$ taps in a tiny reciprocity window. Same
coefficient/bitmap/amplitude/phase machinery; R17 trades a combinatorial port
index for far cheaper (and reciprocity-justified) delay reporting.

---

**Next:** [Chapter 6 — eType II Doppler R18](06-etype2-doppler-r18.md) adds the
*temporal* (Doppler) DFT axis on top of R16, so a single report predicts the
precoder across $N_4$ future slots.
