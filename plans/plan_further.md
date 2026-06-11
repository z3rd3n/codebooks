# Create `plan_further.md`: verification deepening + structural review plan

## Context

The R15–R18 codebook framework is implemented (9 commits, 112 test functions / 145 runs, all green; Sionna/TF finished installing but its integration suite has not run yet). The user now wants a **planning document** (`plan_further.md`, in the repo root, next to the existing untracked `plan.md` which must not be touched) that lays out, in detail:

1. **further tests comparing implementation results against every value/table/figure in `paper/main.tex`** — as exhaustively as possible;
2. **many more correctness-verification tests** beyond the paper;
3. **a structural review from the beginning** — errors, missing spec features, design gaps — with concrete fixes.

This plan was prepared with fresh evidence: the real `f1.pdf` was digitized (9 curves × 7 SNR points), and four review claims were verified against the code (`compact_r15` untested, `SUBBAND_SIZES` unused, missing `L ≤ N1N2` validation, missing forced-`Mv=N3` test).

**Execution = write `plan_further.md` with the content below.** No code changes in this task; the document is the deliverable. (Its own "Phase V0" begins with running the now-installed Sionna suite.)

---

## Content of `plan_further.md`

### V0 — Immediate: close out the pending integration step
- Run `pytest -m sionna` (TF/Sionna finished installing after the last commit; `tests/test_sionna_integration.py` has never executed). Fix any Sionna 1.x API drift in `channel/sionna_adapter.py` (import paths, `PanelArray`/`CDL` kwargs, tensor shapes).
- Add the missing strong check there: **port-mapping orientation test** using near-LoS CDL-D/E — Type I beamforming gain must approach the full array gain (≈10·log10(N1N2) dB over single-antenna); a wrong `_port_permutation` cannot pass this, whereas the current SGCS sanity check could.

### Part A — Paper-anchor comparison tests (exhaust `main.tex`)

**A1. Full digitized f1 regression** (`tests/test_f1_paper_values.py`, slow marker).
Freeze the digitized curve table (read from `paper/pictures/f1.pdf`, ±0.1 reading error) and assert the reproduction tracks it within a documented tolerance band (suggest ±0.6 b/s/Hz absolute or ±8% relative, whichever larger; current code already matches N=16 cases within ~0.3):

| SNR (dB) | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| N=4,Ns=1 UB | 2.45 | 3.95 | 5.4 | 6.9 | 8.65 | 10.35 | 12.05 |
| N=4,Ns=1 TII | 2.4 | 3.9 | 5.35 | 6.85 | 8.6 | 10.3 | 12.0 |
| N=4,Ns=1 TI | 2.25 | 3.6 | 5.15 | 6.75 | 8.5 | 10.15 | 11.85 |
| N=16,Ns=1 UB | 4.2 | 5.5 | 7.15 | 8.95 | 10.55 | 12.25 | 13.9 |
| N=16,Ns=1 TII | 4.0 | 5.3 | 6.95 | 8.75 | 10.35 | 12.05 | 13.7 |
| N=16,Ns=1 TI | 3.35 | 4.8 | 6.35 | 8.1 | 9.8 | 11.5 | 13.05 |
| N=16,Ns=2 UB | 5.3 | 8.2 | 11.35 | 14.65 | 18.0 | 21.3 | 24.7 |
| N=16,Ns=2 TII | 5.0 | 7.8 | 10.95 | 14.25 | 17.55 | 20.9 | 24.3 |
| N=16,Ns=2 TI | 4.1 | 6.65 | 9.6 | 12.85 | 16.1 | 19.25 | 22.7 |

Plus a **channel-calibration sweep** in `eval/f1.py`: fit `n_paths ∈ {2..8}` (and restricted vs unrestricted rank-2 Type I — paper's 2-stream TI of 16.1 @20 dB sits between our two variants, 12.9/16.6) by least-squares against the table; freeze the best config + seed as the documented reproduction setting. N=4 currently shows a slightly larger TI gap than the paper (paper ≈0.15, ours ≈0.5 @20 dB) — the sweep should resolve or document this residual.

**A2. Exhaustive parameter-table tests** (`tests/test_config_tables.py`).
Currently tables are exercised only via spot checks. Add a literal row-by-row assertion against the paper for: `R16_PARAM_COMBOS` (8 rows of Table tabp), `R17_PARAM_COMBOS` (8 rows, tabp17), `R18_PARAM_COMBOS` (9 rows, tabpredic), `SUPPORTED_N1N2` (all 13 rows of tabNO incl. P_CSIRS grouping), `SUBBAND_SIZES` (tabCSS: 24–72→{4,8}, 73–144→{8,16}, 145–275→{16,32}), and derived quantities: `M_v = ceil(p_v·N3/R)` for the paper's worked example (273 RB / subband 16 → 18 subbands → M_v ∈ {3,5}); `K0` for every R16/R18 combo at N3 ∈ {12, 18, 36}.

**A3. f2 golden-number regression** (`tests/test_overhead.py` extension).
Freeze the per-element bit values for the f2 configuration ((16,1), v=2, N3=18, Mv=5, K_NZ=20, N_PSK=4, K2=6) for every L ∈ {1..4} and codebook column of Tables bit1/bit2 as explicit golden dicts (e.g. R15 L=4: i11=2, i12=11, i13=6, i14=42, i21=2·18·14, i22=2·18·5). This locks the formulas against regressions and makes the documented paper-bars erratum auditable.

**A4. Compact-model coverage for R15** (extend `tests/test_type1.py`, `tests/test_type2_r15.py`).
`codebooks/compact.py::compact_r15` and `dual_block` are currently untested; the paper gives compact models for *all* codebooks. Add: Type I — `W == W_s · w_PMI` with `w_PMI = blockdiag(e_lm, e_lm)·[1; φ]/√P` (paper "Compact Model for R15 Type I"); R15 Type II — spec reconstruction ≡ `compact_r15(B, coeffs)` direction-wise for random valid PMIs, both regular and PS bases (models (a) and (b): effective vs full bases — full-basis form uses the sparse `w_PMI` with support = selected beams).

**A5. PMI composition tables** (`tests/test_pmi_composition.py`).
The paper lists exactly which indicators exist per (codebook, rank, SA): R15 `i1=[i11 i12 i131 i141 (i132 i142)]`, `i2` per SA/v (4 cases); R16 eq. (a85); R16-PS eq. (a86) (no i12, explicit i11); R17 eq. (a104) (no i11/i15, unified i16); R18 eq. (a127) (adds i110). Test that `overhead_bits()` keys match these element sets exactly per configuration (e.g. R17 with α=1 drops i12; M=1 drops i16; R18 N4=1 drops i110; R16 N3≤19 drops i15). This is a cheap, high-value spec-conformance net.

**A6. Missing spec mechanics from the paper — implement + test:**
- **Type II codebook subset restriction** (paper eqs. a58/a59, bit sequences B=B1B2, Table tabmaxap: bits 00/01/10/11 → max p(1) of {0, √¼, √½, 1}): `R15Type2Codebook` accepts a restriction object; `select()` must avoid restricted groups/beams and cap quantized wideband amplitudes; reuse the already-tested `decode_restriction_groups` (Algorithm 2 codec — currently implemented but unused by any codebook). Tests: restricted beam never selected; amplitude cap honored; B1 is 11 bits (`< 2**11` already asserted).
- **Rank-restriction bit sequences**: Type I `r = [r0..r7]` (paper eq.) and R18 `typeII-Doppler-RI-Restriction-r18` (4 bits): `select(H, rank)` raises/refuses prohibited ranks. Small feature, completes the reporting model.
- **Type I Mode 1-2 note**: ranks 3–8 stay out of scope (tables not in paper) — restate in plan and README.

**A7. Qualitative Table "mastructure" as integration assertions** (`tests/test_paper_claims.py`, slow).
One statistical test per row over random sparse channels: R15-II SGCS > R15-I (precision); R16 overhead < R15-II at comparable SGCS (±0.05) ("Medium vs High overhead"); only R18 improves future-interval SGCS under mobility (already partially covered — generalize across seeds); PS codebooks beat regular ones on beam-domain channels and vice versa (the "Applicable Scenarios" section; `scripts/compare_schemes.py` already demonstrates the R17 boundary — promote to a test).

### Part B — Verification-hardening tests (beyond the paper)

**B1. Property-based randomized invariants** (`tests/test_invariants.py`).
Single parametrized harness over **all 13 supported (N1,N2) configs × all codebooks × ranks**, random channels (seeded): precoder shape/finite/unit-norm per (interval, t); per-layer columns norm 1/√v; all PMI integer fields within their spec ranges (write one `validate_pmi(pmi, cbk)` helper per codebook family — also reusable as runtime validation); `select→precoder→select` idempotence on the reconstructed precoder's own channel (SGCS must not decrease); overhead keys/values stable across repeated calls. Use modest trial counts to keep the suite < 30 s.

**B2. PMI bit-level serialization** (`src/nr_csi/codebooks/serialize.py` + `tests/test_serialize.py`).
Currently `overhead_bits()` *claims* sizes but no actual bitstream exists — the single biggest credibility gap for ML-vs-codebook comparisons. Implement generic pack/unpack driven by per-field (name, width) descriptors derived from the same logic as `overhead_bits`; round-trip test `unpack(pack(pmi)) == pmi` for every codebook/config, and `len(pack(pmi)) == total_overhead_bits(pmi)` — this *forces* the bit accounting to be honest (it will catch any element the formulas over/under-count, e.g. unreported strongest-coefficient fields).

**B3. Compression-fidelity properties** (`tests/test_compression_properties.py`).
- Forced `Mv=N3` (test-only subclass overriding `Mv`/`K0`): R16 FD compression is lossless vs per-subband LS — planned originally, never written.
- Monotonicity (statistical, fixed seeds): SGCS non-decreasing in L (R15: 2→3→4; R16/R18 combos with growing L), in Mv (p_v ¼→½), in β (K0), in N_PSK (4→8 for R15); overhead strictly increasing alongside — verifies the rate-distortion knobs all point the right way.
- R18 N4 sweep: prediction SGCS at fixed Doppler improves with N4 ∈ {2,4,8} on on-grid channels; degrades gracefully off-grid.
- Quantizer distortion bound: for random coefficients, reconstruction error ≤ half-step of the corresponding tables (amplitude in log domain, phase ±π/N_PSK).

**B4. Negative tests & input validation** (fixes in constructors + `tests/test_validation.py`).
Add the missing guards, each with a raising test: `L ≤ N1N2` (R15-II; (2,1) with L=4 currently constructs and breaks downstream), `Mv ≤ N3` at construction not first use, R17 `K1 ≤ P`, rank vs `min(Nr, P)` error message (exists — test it explicitly), malformed PMI rejection in `precoder()` via the B1 `validate_pmi` helpers (wrong shapes, out-of-range indices, bitmap/strongest-coefficient inconsistency), N3 mismatch between channel and config (exists in Type I only — extend to all).

**B5. Cross-codebook equivalence tests** (`tests/test_equivalences.py`).
- R16-PS + DFT PEB ≡ R16 regular (mirror of the existing R15 PEB test, at the R16 level with taps).
- R17 with α=1, M=1 vs R16-PS with d=1, Mv=1-style config on a flat channel: same selected ports ⇒ same precoder direction.
- R15 Type II with L beams on a frequency-flat channel ≡ R16 with any Mv (only tap 0 active): same SGCS within quantizer-resolution differences (8PSK/3-bit vs 16PSK/4+3-bit — assert R16 ≥ R15 − ε).
- Type I ≡ "Type II with L=1" on single-beam channels (paper's f2 remark).

**B6. Metric & harness edge cases** (`tests/test_eval_spine.py` extension).
`sgcs`/`nmse` with zero columns; `su_rate` rank-deficient W; `mu_rate` K=1 degenerates to `su_rate`; `evaluate()` with R18 (n_slots=N4) — currently only manual in tests, no harness-level test; `EvalResult.se ≤ se_upper_bound` invariant; deterministic across runs with the same rng.

**B7. Tooling for confidence**: add `ruff` to the dev extra + clean run; `pytest --cov` target with a goal of ≥90% line coverage on `src/nr_csi/codebooks` and `utils` (current blind spots: `compact_r15`, error paths, `SionnaCDLChannel`); a `make check`-style script (`scripts/check.sh`) running lint + fast suite + slow suite.

### Part C — Structural review findings (errors / missing / design)

**C1. Confirmed gaps to fix** (each paired with tests above):
1. `SUBBAND_SIZES` transcribed but unused — wire into `SionnaCDLChannel` (derive N3 from BWP RBs + subband size per tabCSS, instead of free `fft_size//N3` chunking), or expose `n3_for_bwp(n_rb, size)` helper used by the adapter. (A2 tests the table; this makes it real.)
2. Missing input validation (B4 list).
3. Type II subset restriction + rank restriction not implemented (A6) — Algorithm 2 codec currently dead code.
4. `compact_r15`/`dual_block` untested (A4).
5. Sionna integration unexecuted; port permutation unproven (V0).

**C2. Design improvements**:
1. **Harness realism knobs** (`eval/harness.py`): optional `feedback_delay_slots` (score W on slots *after* the measurement window — makes the R16-vs-R18 mobility comparison a one-liner instead of hand-rolled test code) and optional channel-estimation noise (`H_est = H + σ·noise` at given measurement SNR) — both essential for honest ML comparisons later.
2. **MU-MIMO evaluation mode**: `evaluate_mu(schemes, K users)` pairing reported PMIs with ZF/RZF cross-user precoding and `metrics.mu_rate` (paper eq. 2) — Type II's raison d'être is MU-MIMO; SU-only evaluation undersells it vs Type I. Reuses `baselines.zf/rzf`.
3. **Promote `RandomRayChannel`** from `scripts/compare_schemes.py` into `channel/synthetic.py` (tests + scripts both need a re-randomizing sparse channel; it currently lives in a script — design smell).
4. **Auto-rank (RI) selection** (optional, P2): `select_rank(H)` maximizing SE over allowed ranks, honoring the A6 rank-restriction bitmaps.
5. **Type1Codebook.select vectorization** (P2): current nested Python loops are O(G1·G2·i13·N3·n_i2) with per-candidate beam construction; precompute the candidate precoder tensor once (numpy einsum) — matters once Sionna evaluations loop over drops.
6. **Baseline completeness** (P2, paper Appendix A): add GMD, EZF, BD, WMMSE (Algorithm "WMMSE" pseudocode in the paper), water-filling/harmonic-mean power allocation to `baselines/` with closed-form unit tests (e.g. BD: zero inter-user leakage; WMMSE: monotone objective) — gives the future ML work the full-CSI reference family the paper tabulates.
7. **Docs**: README section mapping each paper table/figure/equation → implementing module + test (a traceability matrix; doubles as the review's completion checklist).

**C3. Explicit non-goals** (re-stated so the review is honest): CSI-RS resource mapping / Gold sequences (paper Appendix B–C, orthogonal to codebooks), Type I multi-panel & ranks 3–8, R19 (separate future phase), exact f2 bar reproduction (documented erratum stands).

### Suggested execution phases (each lands green before the next)
- **V0**: Sionna suite + port-orientation test + any adapter fixes. *(small)*
- **V1**: A2 + A3 + A5 (pure table/composition tests — fast wins, no production code). *(small)*
- **V2**: B4 validation guards + B1 invariants + `validate_pmi` helpers. *(medium)*
- **V3**: A4 + B3 + B5 + B6 (compact-R15, forced-Mv, monotonicity, equivalences, harness edges). *(medium)*
- **V4**: A1 f1 digitized regression + calibration sweep; A7 claims tests. *(medium, slow-marked)*
- **V5**: B2 bit serialization (forces overhead honesty). *(medium)*
- **V6**: A6 subset/rank restriction; C2.1–C2.3 harness/channel design items; C1.1 tabCSS wiring; B7 tooling. *(large)*
- **V7 (optional/P2)**: C2.4–C2.6 (auto-RI, Type I vectorization, Appendix-A baselines), traceability matrix.

Rough scale: ~90–120 new test functions across 10 new/extended test files; production changes concentrated in `serialize.py` (new), constructor guards, restriction support, harness knobs, Sionna adapter.

## Verification (for this task)
1. `plan_further.md` exists at repo root, contains all sections above (V0, A1–A7, B1–B7, C1–C3, phases), and renders cleanly as Markdown.
2. The digitized f1 table in it matches `paper/pictures/f1.pdf`.
3. `plan.md` (pre-existing, untracked) is untouched; `git status` shows only the new file.
