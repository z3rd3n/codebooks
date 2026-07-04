# 3GPP NR PMI Codebooks — Technical Tutorial

This folder is a deep, code-anchored tutorial on the 5G NR **precoding matrix
indicator (PMI) codebooks** implemented in [`src/nr_csi/codebooks/`](../../src/nr_csi/codebooks/).
It explains, for every codebook family, **what the UE reports, how those report
fields are computed, and how the gNB reconstructs the precoder** from them — and
ties each statement to the exact function in the code and the clause of
TS 38.214 §5.2.2.2 it transcribes.

Each chapter is a **spec-faithful** reference for its clause(s) of
TS 38.214 Release-18 ([specs/38214-i00.md](../../specs/38214-i00.md)), so it
documents the *full* standard — including features that are standardized but **not
implemented** in this codebase. Those are flagged with a consistent marker (see
[Marker convention](#marker-convention) below).

The tutorial is split **one file per codebook family** so each can be read on its
own. They share a common substrate (DFT bases, combinatorial index codecs,
amplitude/phase quantizers, the `select`/`precoder`/`overhead_bits` interface),
which is documented once in the *Foundations* chapter and referenced everywhere
else.

## How to read this

Start with **Foundations** — every Type II family is built on the same spatial /
frequency / Doppler compression machinery, and the later chapters assume you know
it. Then read whichever family you care about. The families form a clean
generalization chain:

```
Type I  (beam + co-phasing, scalar coefficients)
   │
Type II R15  (L-beam linear combination, per-subband coefficients)
   │   + delay-domain (frequency) compression
Type II R16  (eType II:  Mv DFT delay taps)
   │   + free port selection,  reciprocity tap window
Type II R17  (feType II PS)
   │   + Doppler-domain (temporal) compression
Type II R18  (eType II Doppler / predicted PMI)
   │   + large arrays / refined beam selection
Release 19  (refined Type I & Type II)

   ⌐ branch off R16 / R17: + multi-TRP (N_TRP) coherent joint transmission
Type II R18 CJT  (eType II / feType II PS for CJT)
```

## Chapters

| # | File | Codebook family | Spec clause | Ranks |
|---|------|-----------------|-------------|-------|
| 0 | [00-foundations.md](00-foundations.md) | Shared machinery: bases, indexing, quantization, metrics, conventions | §5.2.2.2 (common) | — |
| 1 | [01-type1-single-panel.md](01-type1-single-panel.md) | R15 **Type I** single-panel | §5.2.2.2.1 | 1–8 |
| 2 | [02-type1-multi-panel.md](02-type1-multi-panel.md) | R15 **Type I** multi-panel | §5.2.2.2.2 | 1–4 |
| 3 | [03-type2-r15.md](03-type2-r15.md) | R15 **Type II** (regular + port selection) | §5.2.2.2.3/4 | 1–2 |
| 4 | [04-etype2-r16.md](04-etype2-r16.md) | R16 **Enhanced Type II** (+ PS) | §5.2.2.2.5/6 | 1–4 |
| 5 | [05-fetype2-r17.md](05-fetype2-r17.md) | R17 **Further-enhanced Type II PS** (+ R18 predicted PS) | §5.2.2.2.7, §5.2.2.2.11 | 1–4 |
| 6 | [06-etype2-doppler-r18.md](06-etype2-doppler-r18.md) | R18 **eType II Doppler** (predicted PMI) | §5.2.2.2.10 | 1–4 |
| 7 | [07-refined-r19.md](07-refined-r19.md) | Release-19 **refined** Type I & Type II | §5.2.2.2.1a/2a/5a/9a/11a | 1–8 |
| 8 | [08-cjt.md](08-cjt.md) | R18 **CJT** eType II & feType II PS (multi-TRP) | §5.2.2.2.8/9 | 1–4 |

## Marker convention

Because each chapter documents the whole standard, not just the code, anything
**standardized but not implemented in this codebase** is flagged consistently:

* a whole feature/section gets a callout —
  > 🚩 **STANDARDIZED — NOT IMPLEMENTED IN THIS CODEBASE.** …
* a single missing option/value is tagged inline — **[not implemented]**.

Everything unmarked is implemented. Every codebook family — including the CJT
pair, the 2-port Type I codebook, and all Release-19 refined classes — now has
an implementation; the remaining flags cover isolated reporting-stack options.
Grep the docs for `NOT IMPLEMENTED` to enumerate the gaps between the standard
and the code.

## The one interface they all share

Every codebook is a `CodebookScheme`
([base.py](../../src/nr_csi/codebooks/base.py)) implementing three methods:

```python
pmi  = scheme.select(H, rank)      # UE side:  channel H[slot, t, rx, port] -> PMI
W    = scheme.precoder(pmi)        # gNB side: PMI -> W[interval, t, port, layer]
bits = scheme.overhead_bits(pmi)   # feedback cost, per PMI information element
```

* `precoder` is a **pure TS 38.214 reconstruction** — it is fully standardized
  and cross-checked against an independent "compact / Tucker" tensor model
  ([compact.py](../../src/nr_csi/codebooks/compact.py)).
* `select` is the **UE algorithm**, which the standard does *not* fix. The
  implementation uses one reasonable, well-documented procedure (eigenvector
  targets → beam/group selection → FFT-based tap/shift selection → spec
  quantization). A learned ("ML") feedback scheme drops in at exactly this point:
  implement the same three methods and run it through the same harness.
* `overhead_bits` is validated by a **bit-exact serializer**
  ([serialize.py](../../src/nr_csi/codebooks/serialize.py)): the round-trip
  `unpack(pack(pmi)) == pmi` and `len(pack(pmi)) == total_overhead_bits(pmi)`
  hold for every configuration, so the bit counts are honest.

> Scope. These chapters cover the codebooks themselves — the math of the report
> and the reconstruction. They deliberately do **not** cover the web UI, the
> figure gallery, the Sionna channel adapter, or the evaluation harness beyond
> the spectral-efficiency / overhead metrics needed to define what "good" means.
