# nr-csi — 3GPP NR PMI Codebook Benchmark Framework (R15–R18)

NumPy implementations of the 5G NR beamforming codebooks, built as a
trustworthy baseline set for comparing learned (ML) CSI feedback algorithms
against the standardized ones under identical channels and metrics.

The authoritative source is the tutorial paper in `paper/main.tex`
(*"Precoding Matrix Indicator in the 5G NR Protocol: A Tutorial on 3GPP
Beamforming Codebooks"*), which mirrors TS 38.214 §5.2.2.2.

## Implemented codebooks

| Class | Codebook | Compression | Ranks |
|---|---|---|---|
| `Type1Codebook` | R15 Type I single-panel, Modes 1–2 | beam + co-phasing | 1–2 |
| `R15Type2Codebook` | R15 Type II (+ port-selection variant) | spatial (L beams) | 1–2 |
| `R16Type2Codebook` | R16 eType II (+ PS variant) | spatial + delay (M_v taps) | 1–4 |
| `R17Type2Codebook` | R17 FeType II port-selection | free ports + M taps | 1–4 |
| `R18Type2Codebook` | R18 eType II Doppler (predicted PMI) | spatial + delay + Doppler (Q shifts over N₄ intervals) | 1–4 |

Each codebook implements the same interface (`nr_csi.codebooks.base.CodebookScheme`):

```python
pmi  = scheme.select(H, rank)     # UE side:  H[slot, t, rx, port] -> PMI
W    = scheme.precoder(pmi)       # gNB side: PMI -> W[interval, t, port, layer]
bits = scheme.overhead_bits(pmi)  # feedback cost per information element
```

`precoder` is a pure TS 38.214 reconstruction (verified against the paper's
compact matrix/Tucker models); `select` is one reasonable UE algorithm
(eigen targets → orthogonal-group/beam selection → FFT tap/shift selection →
spec quantization), since the UE side is not standardized.

## Install & test

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # numpy/scipy/matplotlib + pytest
pytest -m "not slow and not sionna"   # spec-level unit suite (fast)
pytest -m slow                        # Fig. f1 statistical reproduction
pip install -e ".[sionna]"            # optional: TensorFlow + Sionna
pytest -m sionna                      # 38.901 CDL end-to-end integration
```

## Reproducing the paper's figures

```bash
python scripts/reproduce_f1.py   # SE vs SNR: Type I vs Type II vs eigen bound -> results/f1.png
python scripts/reproduce_f2.py   # feedback bits vs L: R15/R16/R18            -> results/f2.png
```

## Comparing your ML CSI algorithm

Implement the same three methods and run it through the harness next to the
3GPP codebooks:

```python
import numpy as np
from nr_csi.config import AntennaConfig
from nr_csi.codebooks.base import CodebookScheme
from nr_csi.codebooks.etype2_r16 import R16Type2Codebook
from nr_csi.channel.sionna_adapter import SionnaCDLChannel
from nr_csi.eval import evaluate

class MyMLScheme(CodebookScheme):
    name = "my-autoencoder"
    def select(self, H, rank=1):       # encode: channel -> latent/bits
        ...
    def precoder(self, pmi):           # decode: bits -> W[1, N3, P, rank]
        ...
    def overhead_bits(self, pmi):      # {"latent": n_bits}
        ...

ant = AntennaConfig.standard(4, 2)
chan = SionnaCDLChannel(ant, N3=8, model="A", n_rx=2)
for scheme in [R16Type2Codebook(ant, N3=8, param_combination=4), MyMLScheme()]:
    res = evaluate(scheme, chan, snr_db=[0, 10, 20], rank=1, n_drops=100)
    print(scheme.name, res.se, res.sgcs, res.overhead_bits)
```

Metrics: spectral efficiency (`metrics.se`), SGCS — the standard 3GPP AI/ML
CSI metric — and NMSE (`metrics.similarity`), feedback bits
(`metrics.overhead` + per-scheme `overhead_bits`).

## Conventions

* Channels: `H[slot, t, rx, port]`, `t = 0..N3-1` PMI frequency units;
  port index = `pol * N1*N2 + n1*N2 + n2` (vertical fastest), matching the
  Kronecker spatial bases `v_{m1,m2} = a_{m1} ⊗ u_{m2}`.
* Precoders: `W[interval, t, port, layer]` with `tr(WᴴW) = 1` per `(interval, t)`.
* The synthetic ray channel (`channel.synthetic`) uses `H = g · a_rx v_txᴴ`,
  so an on-grid codebook beam is exactly the optimal precoder — handy for
  exact-recovery tests.  Note: the optimal precoder rotates with the
  *conjugate* of the channel's delay/Doppler phase (a channel Doppler of
  +n appears as precoder shift N₄−n).

## Paper errata / open interpretation points found while implementing

1. **Algorithm 4 (R17 port selection)** has copy-paste typos: line 3 must use
   `C(x*, L−i)` and `s_{i−1}` (not `C(x*, 4−k)`, `s_{k−1}`). Implemented
   corrected (`utils.combinatorics`, shared codec for Algorithms 1–4).
2. **Fig. f1 text**: "phase quantized to 3 bits, i.e., N_PSK = 4" is
   self-contradictory (N_PSK=4 is 2 bits). The reproduction defaults to
   8-PSK (3 bits); configurable in `eval.f1`.
3. **Fig. f1 channel** is unspecified; we use a sparse multipath ULA channel
   normalized so E‖h‖² = N, which matches the figure's upper-bound curve.
   The paper's 2-stream Type I procedure is also unspecified; we restrict the
   second beam to the spec's i₁,₃ offsets (Table tabmap), which brackets the
   paper's curve from below (an unrestricted search brackets it from above).
4. **Fig. f2 absolute bar values are not derivable from the paper's own
   Tables bit1/bit2** with the stated parameters (the bars imply an R15/R16
   ratio ≈17× at L=4; the table formulas give ≈3× under any consistent
   accounting). `metrics.overhead` implements the table formulas; the
   qualitative claims (R15 ≫ R16 > R18 for L≥2, equal growth in L, gap
   growing with N₃/N₄) hold and are asserted in `tests/test_overhead.py`.
5. **R16 PS parameter table** (TS 38.214 Table 5.2.2.2.6-1) is referenced but
   not transcribed in the paper; the regular `paramCombination-r16` table is
   reused for the PS variant here.
6. Type I ranks 3–8 need TS 38.214 tables that the paper does not include;
   they are out of scope (extension hooks in `codebooks/type1.py`).

## Layout

```
src/nr_csi/
  config.py             antenna/subband configs, param-combination tables
  utils/                DFT bases, combinatorial codecs (Algs 1–4), quantizers
  codebooks/            the five codebook families + compact/Tucker models
  channel/              synthetic ray channel, Sionna 38.901 CDL adapter
  baselines/            eigen/SVD/ZF/RZF/MMSE full-CSI baselines
  metrics/              SE, SGCS/NMSE, feedback-overhead formulas
  eval/                 Monte-Carlo harness + Fig. f1 experiment
scripts/                reproduce_f1.py, reproduce_f2.py
tests/                  ~150 spec-anchored tests (see test matrix in plan)
```
