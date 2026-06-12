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

Also implemented:

* **Bit-level serialization** (`nr_csi.codebooks.pack/unpack`): the actual
  feedback bitstream; `len(pack(pmi)) == total_overhead_bits(pmi)` and
  `unpack(pack(pmi)) == pmi` are asserted for every configuration, so the
  overhead numbers used in comparisons are honest.
* **Runtime PMI validation** (`codebooks/validate.py`): every `precoder()`
  rejects malformed reports (shapes, index ranges, strongest-coefficient
  conventions).
* **Codebook subset restriction** (`TypeIIRestriction`, paper eqs. a58/a59 +
  Table tabmaxap): restricted beams are never selected; wideband amplitudes
  are capped. Rank restriction bitmaps for Type I (8-bit `r`) and R18
  (4-bit `typeII-Doppler-RI-Restriction-r18`).
* **Evaluation harness extras** (`nr_csi.eval`): `feedback_delay_slots`
  (CSI aging) and `measurement_snr_db` (estimation noise) knobs on
  `evaluate`, plus `evaluate_mu` for MU-MIMO ZF/RZF sum rates from reported
  PMIs (paper eq. 2) against a full-CSI reference.
* **Table tabCSS wiring**: `nr_csi.config.n3_for_bwp(n_rb, subband_size, R)`
  and BWP-driven subband mapping in `SionnaCDLChannel(n_rb=..., subband_size=...)`.

## Install & test

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # numpy/scipy/matplotlib + pytest
pytest -m "not slow and not sionna"   # spec-level unit suite (fast)
pytest -m slow                        # Fig. f1 statistical reproduction
pip install -e ".[sionna]"            # optional: TensorFlow + Sionna
pytest -m sionna                      # 38.901 CDL end-to-end integration
scripts/check.sh                      # lint (ruff) + fast + slow suites
scripts/check.sh --with-sionna --cov  # everything, with coverage
```

## Reproducing the paper's figures

```bash
python scripts/reproduce_f1.py   # SE vs SNR: Type I vs Type II vs eigen bound -> results/f1.png
python scripts/reproduce_f2.py   # feedback bits vs L: R15/R16/R18            -> results/f2.png
```

## Comparison figure gallery

`scripts/fig_*.py` compare all codebook families along every axis the
harness measures (plan: `plans/plan_figures.md`; shared conventions in
`scripts/figlib.py` — paired seeds, port-selection codebooks evaluated
through a unitary DFT PEB, every PNG paired with a JSON of the plotted
numbers). Regenerate everything with:

```bash
python scripts/make_all_figures.py           # full quality -> results/fig_*.png
python scripts/make_all_figures.py --fast    # smoke-test sizes
python scripts/fig_02_rate_distortion.py --drops 200 --seed 1   # any one figure
```

| Figure | Comparison |
|---|---|
| `fig_01_se_vs_snr` | SE vs SNR, all families + eigen bound, ranks 1–2 |
| `fig_02_rate_distortion` | SGCS / SE vs feedback bits over every config knob, Pareto frontier (the plane an ML scheme must beat) |
| `fig_03_overhead_breakdown` | per-PMI-element bits, grouped by what they encode |
| `fig_04_overhead_scaling` | bits vs N₃ / L / N₄ coverage (Tables bit1/bit2) |
| `fig_05_mobility` | CSI aging vs the R18 predicted PMI (per-interval SGCS, feedback delay) |
| `fig_06_mu_mimo` | ZF sum rate from reported PMIs vs SNR and user count |
| `fig_07_rank_adaptation` | fixed ranks 1–4 vs auto-RI, rank distribution vs SNR |
| `fig_08_channel_sensitivity` | robustness to channel sparsity and estimation noise |
| `fig_09_port_selection` | regular vs PS codebooks on antenna- vs beam-domain channels |
| `fig_10_array_scaling` | SE / gap-to-bound / SGCS / bits vs array size (P = 8…32, all families + (16,1) aspect contrast) |
| `fig_11_frequency_granularity` | fidelity and cost vs N₃: per-subband (R15) vs M_v-tap (R16) reporting |
| `fig_12_summary` | normalized radar scorecard + raw-numbers table (`results/fig_12_summary_table.md`) |

Every figure has a hand-written analysis next to it
(`results/<figure>.md`): what it shows, the mechanism behind each trend,
and investigations of the initially surprising results (Type I's rank-2
SGCS collapse, the R18 K₀ = ⌈2βLM₁Q⌉ static-channel budget bonus, R17's
fidelity *rising* with P, the windowed-PS vs free-PS gap, the measurement-
noise ranking inversion, the L = 1 overhead inversion in f2).

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
   The paper's 2-stream Type I procedure is also unspecified: the spec's
   i₁,₃ offsets (Table tabmap) bracket the paper's curve from below, an
   unrestricted second-beam search from above. A least-squares calibration
   against the digitized figure (`eval.f1.calibrate_f1`) selected 8 paths
   with the unrestricted variant (`eval.f1.F1_REPRODUCTION`); all 63
   digitized points then reproduce within max(±0.6 b/s/Hz, ±8%)
   (`tests/test_f1_paper_values.py`). Known residual: the paper's N=4
   single-stream Type I curve sits closer to Type II (~0.15 b/s/Hz) than
   spec-faithful selection allows (~0.5) on every channel family swept.
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

## Traceability matrix (paper → code → test)

| Paper anchor | Implementing module | Test |
|---|---|---|
| Table tabNO (N1,N2,O1,O2) | `config.SUPPORTED_N1N2` | `test_config_tables.py::TestTabNO` |
| Tables tabp / tabp17 / tabpredic (param combos) | `config.R16/R17/R18_PARAM_COMBOS` | `test_config_tables.py::TestParamComboTables` |
| Table tabCSS (subband sizes), eq. c66 (M_v), K0 | `config.n3_for_bwp`, `config.m_v`, codebook `.K0` | `test_config_tables.py::TestTabCSS/TestDerivedQuantities` |
| Eq. vmm / beamv (spatial DFT bases), y_f, z_τ | `utils.dft` | `test_dft_bases.py` |
| Algorithms 1–4 (combinatorial codecs) | `utils.combinatorics` | `test_combinatorics.py` |
| Tables tabk1 / tabk2 / tabmapkuan / tabmapzhai | `utils.quantization` | `test_quantization.py`, `test_compression_properties.py::TestQuantizerDistortionBounds` |
| Tables tabmode1/tabmode2 + tabmap (Type I) | `codebooks.type1` | `test_type1.py` |
| Eq. a46 + Table tabtypeii (R15 Type II), SA rules | `codebooks.type2_r15` | `test_type2_r15.py` |
| Eqs. a58/a59 + Table tabmaxap (subset restriction) | `codebooks.type2_r15.TypeIIRestriction` | `test_restrictions_and_harness.py::TestSubsetRestriction` |
| RI restriction (Type I 8-bit r; r18 4-bit) | `type1`/`etype2_r18` `rank/ri_restriction` | `test_restrictions_and_harness.py::TestRankRestriction` |
| Table tabesII (R16), i18 dual mode, two-level taps | `codebooks.etype2_r16` | `test_etype2_r16.py` |
| Table tabesps + eq. a86 (R16 PS), eq. psy ≡ regy | `etype2_r16(port_selection=True)` | `test_equivalences.py::TestR16PortSelectionPEB` |
| Table tabfesp + eq. a104 (R17), Algorithm 4 errata | `codebooks.fetype2_r17` | `test_fetype2_r17.py`, `test_equivalences.py::TestR17VsR16PS` |
| Table tab1 + eq. a127 (R18 Doppler), N4=1 ≡ R16 | `codebooks.etype2_r18` | `test_etype2_r18.py`, `test_compression_properties.py::TestR18N4Sweep` |
| PMI compositions (a85/a86/a104/a127 + R15/Type I) | each codebook's `overhead_bits` | `test_pmi_composition.py` |
| Compact/Tucker models (Sec. "Compact Model") | `codebooks.compact` | `test_type1.py`/`test_type2_r15.py`/`test_etype2_r16.py`/`test_etype2_r18.py` `TestCompactModel` rows |
| Tables bit1/bit2 (overhead formulas) | `metrics.overhead` | `test_overhead.py` (incl. frozen golden dicts) |
| Actual feedback bitstream | `codebooks.serialize` | `test_serialize.py` |
| Eq. (2) SU/MU achievable rate, SGCS | `metrics.se`, `metrics.similarity` | `test_eval_spine.py` |
| Fig. f1 (SE vs SNR, digitized 9×7 table) | `eval.f1` | `test_f1_paper_values.py`, `test_f1_curves.py` (slow) |
| Fig. f2 (feedback bits vs L) | `metrics.overhead.f2_comparison` | `test_overhead.py::TestF2Claims` |
| Qualitative comparison table (Sec. "Discussion") | — | `test_paper_claims.py` (slow) |
| Appendix A (SVD/MRT/ZF/RZF/MMSE/GMD/EZF/BD/WMMSE, water-filling, harmonic mean) | `baselines.ideal` | `test_eval_spine.py`, `test_baselines_appendix.py` |
| 38.901 CDL channels (Sionna), port mapping | `channel.sionna_adapter` | `test_sionna_integration.py` (`-m sionna`) |

Out of scope (documented): CSI-RS resource mapping / Gold sequences (paper
Appendices B–C, orthogonal to the codebooks), Type I multi-panel and ranks
3–8 (Modes 1–2; tables not in the paper), R19 (future phase), exact f2 bar
heights (erratum 4).

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
