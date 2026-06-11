# 3GPP NR Codebook Benchmark Framework (R15–R18)

## Context

The user will develop an ML-based CSI feedback algorithm and needs a trustworthy baseline framework implementing the 3GPP PMI codebooks so the ML algorithm can be compared against them under identical channels/metrics. The authoritative source is the tutorial paper in `paper/main.tex` (IEEE COMST, "Precoding Matrix Indicator in the 5G NR Protocol"), which gives spec-level math for:

- **R15 Type I** regular, Modes 1 & 2 (single-beam + co-phasing, Tables 5.2.2.2.1-5/-6, k1/k2 offset table, codebook subset restriction)
- **R15 Type II** regular + port-selection (L∈{2,3,4} beam combination, 3-bit WB / 1-bit SB amplitude, N_PSK∈{4,8} phase, Algorithms 1–2)
- **R16 eType II** regular + port-selection (adds delay-domain DFT compression: M_v of N₃ taps, bitmap i₁,₇, K₀ budget, two-level tap indication for N₃>19, Algorithm 3)
- **R17 FeType II** port-selection (free port selection via α, M∈{1,2} taps, Algorithm 4)
- **R18 eType II Doppler** (adds temporal DFT basis: Q=2 shifts over N₄∈{1,2,4,8} slot intervals; **degenerates exactly to R16 when N₄=1**)
- Compact matrix/tensor (Tucker) models for every codebook; bit-overhead tables (Tables "bit1"/"bit2"); two quantitative figures: **f1** (SE vs SNR) and **f2** (feedback bits vs L).

Decisions made with user: **Python + NumPy**; **R15–R18 now, R19 later**; **Sionna 38.901 CDL/UMa** as the evaluation channel model (plus a tiny internal synthetic channel for deterministic unit tests); **all port-selection variants included**.

Repo is empty apart from `paper/` (gitignored). Everything below is new code at repo root.

## Package layout

```
pyproject.toml                 # deps: numpy, scipy, matplotlib, pytest; extras: sionna (sionna, tensorflow)
src/nr_csi/
  config.py                    # AntennaConfig(N1,N2,O1,O2) + Table tabNO validation, P_CSIRS=2·N1·N2;
                               # SubbandConfig(N3,R); ParamCombo tables: R16 (L,p_v,β), R17 (M,α,β), R18 (L,p_v,β)
  utils/
    dft.py                     # oversampled 2D spatial bases v_{l,m}=â_l⊗û_m; group indexing m=O·n+q;
                               # spectral basis y_f (N3-DFT), temporal basis z_τ (N4-DFT)
    combinatorics.py           # C(x,y); ONE core combinatorial-index codec (encode combo→index, decode index→combo)
                               # parametrized to realize Algorithms 1, 2, 3 (incl. N3>19/M_initial), 4
    quantization.py            # exact tables: R15 WB 3-bit (tabk1), R15 SB 1-bit (tabk2),
                               # R16+ WB 4-bit reference (tabmapkuan, k=0 reserved), SB 3-bit (tabmapzhai);
                               # N_PSK ∈ {4,8,16} phase quantizers; nearest-neighbor quantize helpers
  codebooks/
    base.py                    # CodebookScheme ABC:
                               #   select(H, cfg) -> PMI               (UE side)
                               #   precoder(pmi, cfg) -> W [P, v, N3, N4]  (gNB side, pure spec reconstruction)
                               #   overhead_bits(cfg) -> dict per i1/i2 element
                               # PMI dataclasses mirror spec fields (i11, i12, i_{1,3,l}, …) exactly
    type1.py                   # R15 Type I regular, Modes 1 & 2, ranks 1–2 (paper-detailed); subset restriction bitmap
    type2_r15.py               # R15 Type II regular + PS variant (i11·d+i port window)
    etype2_r16.py              # R16 regular + PS; strongest-tap remap, i18 dual mode (v=1 vs v>1), K0/KNZ logic
    fetype2_r17.py             # R17 PS; L=αP/2, α=1 ⇒ i12 omitted; M∈{1,2}, window N
    etype2_r18.py              # R18 Doppler regular; N4=1 code path MUST delegate to R16 reconstruction
    compact.py                 # compact/Tucker models for all codebooks (paper Sec. "Compact Model") —
                               #   independent second implementation used only for cross-validation in tests
  channel/
    base.py                    # ChannelSource → H[slot, subband, Nr, Nt] + carrier/mobility metadata
    synthetic.py               # deterministic multipath: rays with (AoD az/el, delay tap, Doppler shift) on/off grid;
                               #   used by unit tests (no Sionna dependency)
    sionna_adapter.py          # 38.901 CDL-A..E + UMa via Sionna; dual-pol UPA(N1,N2) at gNB;
                               #   OFDM grid → per-PMI-subband averaging (subband sizes per Table tabCSS);
                               #   time evolution over N4·d slots for Doppler
  baselines/
    ideal.py                   # eigen-beamforming upper bound (dominant eigenvectors per subband);
                               # appendix schemes: SVD, MRT, ZF, RZF, MMSE, EZF (closed-form ones)
  metrics/
    se.py                      # achievable rate eq. (2) of paper (per-subcarrier log-det, interference-aware), SU + MU
    similarity.py              # SGCS (3GPP AI/ML CSI standard metric) + NMSE vs ideal precoder
    overhead.py                # bit-overhead formulas of Tables bit1/bit2 per codebook/config
  eval/
    harness.py                 # evaluate(scheme, channel_source, snr_grid, …) -> results dict
                               # `scheme` is any CodebookScheme — the future ML algorithm implements the same ABC
    figures.py                 # plotting helpers
scripts/
  reproduce_f1.py              # SE vs SNR (Type I vs Type II L=4 vs upper bound; N=4/16; 1–2 streams) → results/f1.png
  reproduce_f2.py              # feedback overhead vs L (R15/R16/R18, log scale) → results/f2.png
tests/                         # see test matrix below
```

### UE-side selection algorithms (not standardized; paper describes its f1 procedure)
- Common: per-subband target = dominant eigenvector of channel (covariance) per layer.
- Type I: exhaustive search over (i11,i12,[i13],i2) maximizing mean projected power (Mode 1: wideband beam + per-subband co-phase; Mode 2: wideband 4-beam group + per-subband beam&phase).
- R15 Type II: pick orthogonal group (q1,q2) by projected energy → OMP/top-L beams → least-squares weights → quantize (WB amp, optional SB amp, PSK phase), strongest-coefficient normalization (k=7/c=0 at i_{1,3,l}, not reported).
- R16: project targets onto 2L selected spatial bases per subband → coefficient matrix (2L×N₃) → DFT along frequency → keep strongest M_v taps (remap strongest to index 0; two-level i15/i16 when N₃>19) → keep top coefficients via bitmap i17 s.t. K_l^NZ ≤ K₀=⌈β·2L·M₁⌉ → quantize per reference-amplitude scheme.
- R17: top-K₁ ports by received energy (free selection, Algorithm 4 encoding), taps restricted to window {0..N−1}.
- R18: additionally DFT along N₄ slot intervals → keep Q=2 Doppler shifts (n₄⁽⁰⁾=0 reference, offset i_{1,10}); reconstruction yields **predicted** precoders for future intervals.

## Implementation phases (each ends with its tests green; commit per phase)

- **P0 — Scaffold + numerics core**: pyproject, package skeleton, `config.py`, `utils/` (dft, combinatorics, quantization) + their tests.
- **P1 — Type I + eval spine**: `type1.py`, `channel/synthetic.py`, `baselines/ideal.py`, `metrics/se.py`, `metrics/similarity.py` + tests.
- **P2 — R15 Type II** regular + PS + tests.
- **P3 — R16 eType II** regular + PS + `compact.py` cross-check tests.
- **P4 — R17 FeType II PS** + tests.
- **P5 — R18 Doppler** + tests (N₄=1 ≡ R16 identity; Doppler prediction property; Tucker equivalence).
- **P6 — Overhead + f2**: `metrics/overhead.py`, `scripts/reproduce_f2.py` + tests against paper claims/bar values.
- **P7 — Sionna + f1 + harness**: `sionna_adapter.py`, `eval/harness.py`, `scripts/reproduce_f1.py`, integration tests (skipped when Sionna absent).
- **P8 — README**: usage, how to plug an ML scheme into the harness (implement `CodebookScheme.select/precoder/overhead_bits`), figure gallery, known paper errata.

## Test matrix (pytest; the anti-derailment safety net)

| File | Anchors (from the paper) |
|---|---|
| `test_dft_bases.py` | beam-group orthogonality; O₁O₂ groups partition all N₁O₁·N₂O₂ beams (Fig. pic_6); Kronecker structure; steering-angle sanity |
| `test_combinatorics.py` | round-trip encode↔decode for Algorithms 1–4, exhaustive on small (N₁,N₂,L) and randomized large; Alg. 3 two-level path with M_initial; Alg. 2 (β₁→g⁽ᵏ⁾,r₁,r₂) |
| `test_quantization.py` | exact table values incl. k=0 "reserved" (tabmapkuan) and √(1/64)…1 ladder (tabk1); quantizer idempotence |
| `test_type1.py` | 1/√P norm; Table tabmap (k1,k2) for all three (N₁,N₂) regimes; Mode 2 i₂ even/odd ↔ φ_n and beam-in-group; subset-restriction bitmap honored; single-grid-ray synthetic channel ⇒ exact beam recovered, SE == upper bound |
| `test_type2_r15.py` | β_l normalization ⇒ unit-norm precoder; strongest-coeff convention; SA=false ⇒ p⁽²⁾=1; K⁽²⁾ caps (4 / 6); channel = exact combo of ≤L grid beams with table-exact amplitudes ⇒ SGCS ≈ 1; PS variant: PS codebook + DFT PEB F ≡ regular codebook (eqs. regy vs psy) |
| `test_etype2_r16.py` | y_f = N₃-DFT column; γ_{t,l} ⇒ unit norm per subband; forced M_v=N₃ (test-only) ⇒ FD compression lossless vs per-subband LS; K₀=⌈β·2L·M₁⌉, K^NZ ≤ 2K₀ enforced; strongest-tap remap (n₃⁽⁰⁾=0); i₁,₈ v=1 vs v>1 modes; two-level indication N₃>19; spec precoder == compact model Ŵ_s·W_c·Ŵ_fᵀ for random PMIs |
| `test_fetype2_r17.py` | L=K₁/2=αP/2; α=1 ⇒ i₁,₂ not reported & m⁽ⁱ⁾=i; M=1 ⇒ wideband-flat precoder; M=2,N=4 offset; non-contiguous strong ports ⇒ R17 beats R15-style consecutive selection |
| `test_etype2_r18.py` | **N₄=1 precoders byte-identical to R16** given same channel; z_τ = N₄-DFT; n₄⁽⁰⁾=0 + i₁,₁₀ offset; synthetic single-Doppler channel (DFT-grid phase rotation across slots) ⇒ SGCS(R18) ≈ const over future intervals while static R16 decays; Tucker eq. (tucker) == spec formula |
| `test_overhead.py` | Tables bit1/bit2 formulas; f2 config (N₁N₂=16, O₁O₂=4, N₃=18, N₄=4, Q=2, v=2, M_v=5, N_PSK=4, K⁽²⁾=6, K^NZ=20): assert R15 > 10× R16 (paper claim), R18 < R16, monotone in L; compare to digitized f2 bars (R15 ≈ 5.9e3/9.8e3/2.6e4/7.3e4; R16 ≈ 6.0e2/8.6e2/1.75e3/4.3e3; R18 ≈ 3.3e2/4.7e2/9.5e2/2.25e3 for L=1..4) within ±20% |
| `test_f1_curves.py` (slow marker) | paper f1 setup (single-pol, (N,1) arrays, O=(4,1), Type II L=4, OMP+LS, SB amp off): UB ≥ TypeII ≥ TypeI at every SNR; N=4 curves within ~0.3 bps/Hz of each other; gap grows with N and with streams; spot-check digitized values (e.g. N=16,Ns=2 @20 dB: TypeI ≈ 16.1, TypeII ≈ 17.5, UB ≈ 18.0 bps/Hz) within ±0.5 |
| `test_sionna_integration.py` (`sionna` marker, auto-skip) | CDL-A through full pipeline: valid PMIs for all 7 codebook variants; SE monotone in SNR; R18 beats R16 in SGCS at UE speed ≥ 30 km/h for future-slot prediction |

## Known paper errata to handle (document in README, implement corrected)
- R17 Algorithm 4 line 3 has copy-paste errors: condition must use `C(x*, L−i)` and `s_{i−1}` (not `C(x*,4−k)`, `s_{k−1}`).
- f1 text says "phase quantized to 3 bits, i.e., N_PSK=4" (contradiction: N_PSK=4 is 2 bits). Try N_PSK=8 first when matching f1; keep both configurable.
- f1's exact channel realization is unspecified ⇒ reproduce orderings/gaps and spot values with tolerance, not exact equality.
- Table bit2's R15 column reuses M_v notation for subband count; totals in f2 scale i₂ by N₃ (and N₄ intervals) — calibrate against the bar chart and document the interpretation.

## Scope notes
- Ranks: Type I ranks 1–2 (paper-detailed; ranks 3–8 need TS 38.214 tables not in the paper — leave extension hooks, document); R15 Type II ranks 1–2 (spec max); R16/R17/R18 ranks 1–4 (paper tables given).
- R19 deliberately deferred (user decision) — `CodebookScheme` ABC and config tables are the extension points.
- Sionna is an optional extra (`pip install -e .[sionna]`); core library and all non-integration tests run on NumPy only.

## Verification
1. `pytest -m "not slow and not sionna"` — full spec-level unit suite green (P0–P6).
2. `pytest -m slow` — f1 trend/value reproduction on synthetic channels.
3. `pip install -e .[sionna] && pytest -m sionna` — 38.901 integration.
4. `python scripts/reproduce_f2.py` → compare `results/f2.png` against `paper/pictures/f2.pdf` (log-scale bars, R15 ≈ 10× R16 > R18).
5. `python scripts/reproduce_f1.py` → compare `results/f1.png` against `paper/pictures/f1.pdf` (curve ordering, gaps, spot values).
6. README example: wrap a dummy "ML scheme" (e.g., truncated-SVD oracle) in the harness to demonstrate the drop-in comparison workflow end-to-end.
