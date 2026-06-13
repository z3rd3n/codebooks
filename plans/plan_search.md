# Verify NR CSI Codebook Correctness & Expand Algorithmic Tests

## Context

The user wants assurance that the NR CSI PMI codebook implementations (3GPP TS 38.214:
Type I, Type II R15, Enhanced Type II R16, Further-enhanced PS R17, Doppler R18) are
**algorithmically correct**, by (1) cross-checking against external spec sources and
(2) increasing test coverage for correctness.

### Spec comparison findings (the "compare with the database" deliverable)

I researched the spec online and compared it to the implementation. **Result: the
implementation is a faithful transcription of the spec; no correctness bug was found.**

- Every parameter-combination table matches **exactly**: R16 `tabp`, R17 `tabp17`,
  R18 `tabpredic`, the `(N1,N2)→(O1,O2)` table `tabNO`, and subband-size table `tabCSS`
  (verified line-by-line against `paper/main.tex` lines 1205–1222, 1477–1490, 1723–1740,
  647–690).
- Both R16 amplitude tables match: the closed forms in `quantization.py`
  (`R16_REF_AMP = 2^(-(15-k)/4)`, `R16_DIFF_AMP = 2^(-(7-k)/2)`) reproduce every literal
  row of paper Tables `tabmapkuan` (incl. `k=0` Reserved) and `tabmapzhai`.
- `K0`/`M_v` formulas match (R16 `K0=⌈β·2L·M1⌉`, R18 `K0=⌈2βLM1Q⌉`, `M_v=⌈p_v·N3/R⌉`).
- The paper is a peer-reviewed IEEE Communications Surveys & Tutorials article on
  TS 38.214; its structure was independently confirmed against MATLAB 5G Toolbox docs,
  Sharetechnote, and arXiv reviews (2302.09222, 2210.08218).
- **Limitation (user-acknowledged):** official ETSI/3GPP TS 38.214 PDFs returned HTTP 403
  to automated fetching, so golden values are anchored to the in-repo `paper/main.tex`
  plus secondary sources — which is exactly what the existing anchor tests already do.
- Minor: paper prose (line 693) lists `P_CSI-RS ∈ {4,8,12,24,32}` (omits 16); the code
  correctly supports `P=16` via `(8,1)`/`(4,2)`. This is a paper typo, **not** a code bug.

### Existing coverage (so we don't duplicate)

The suite is already strong: [tests/test_config_tables.py](tests/test_config_tables.py)
independently re-transcribes all param/antenna/subband tables;
[tests/test_quantization.py](tests/test_quantization.py) anchors every amplitude/phase
value; [tests/paper_oracles.py](tests/paper_oracles.py) provides closed-form
precoders built directly from the paper equations (no `nr_csi` helpers);
[tests/test_paper_reconstruction_oracles.py](tests/test_paper_reconstruction_oracles.py)
already compares production precoders to those oracles for several shapes.

This plan adds a **focused** set of high-value tests closing genuine gaps. No production
code changes are expected (none are needed). **No existing test is modified, weakened,
skipped, or marked `xfail`.**

## Approach

Add one new test module: **`tests/test_paper_correctness_extensions.py`**, reusing the
existing oracle helpers (`from tests.paper_oracles import ...`, matching the import style
at [test_paper_reconstruction_oracles.py:22](tests/test_paper_reconstruction_oracles.py#L22))
and config constants from `nr_csi.config`. Four focused test classes:

### Class 1 — `TestUnsupportedRankRejection`  (gap: end-to-end rank-3/4 rejection)
The table-level guard `combo.p_v(3)` is tested, but the **public codebook API** rejection
is not. Assert that for combos with `p_v34 is None`:
- R16 combos 7, 8 and R18 combos 8, 9: `cbk.Mv(3)` and `cbk.Mv(4)` raise `ValueError`,
  and `cbk.select(H, rank=3)` raises `ValueError` (use a valid channel with `n_rx=4` so
  selection reaches the `Mv()` call at [etype2_r16.py:194](src/nr_csi/codebooks/etype2_r16.py#L194)).
- Positive control: combos that **do** support rank 3-4 (R16 5/6, R18 4-7) succeed at
  `Mv(3)`/`Mv(4)`.
Antenna `AntennaConfig.standard(4, 2)` (N1·N2=8 ≥ L=6).

### Class 2 — `TestReconstructionAcrossAllArrays`  (gap: spatial DFT math for every array)
The strongest "is the math right for all geometries" check. For **every** shape in
`SUPPORTED_N1N2` (all 13), drive a synthetic channel through `select()` then assert the
production precoder equals the independent paper-equation oracle:
`np.allclose(cbk.precoder(pmi), paper_oracles.rXX_precoder(cbk, pmi), atol=1e-10)`.
- Families: R16 regular, R16 port-selection, R17 (free PS), R18 (regular).
- Per shape pick a valid combo via a tiny helper: L ≤ N1·N2 for regular families
  (L=2 combo for small arrays, larger L where N1·N2 permits); R17/PS need only N3 ≥ M.
- Ranks: 1 for all 13 shapes × 4 families; rank 2 for a representative subset
  (smallest, a 16-port, a 32-port, and `(16,1)`). ~60-70 fast nodes.
- This implicitly verifies the beam-grid oversampling formula `m = O·n + q`, since
  `paper_oracles.regular_basis` ([paper_oracles.py:67](tests/paper_oracles.py#L67)) builds
  beams from that formula and the production basis must match it.

### Class 3 — `TestType1Orthonormality`  (gap: inter-layer orthogonality, not just norms)
Type I precoder columns are genuinely orthonormal (DFT beam + ± co-phasing), unlike
quantized Type II. For modes 1/2, rank 2, across a few shapes: build `W` via `select()`,
and for each subband assert `W[0,t]^H W[0,t] ≈ I/rank` (Gram matrix = identity/rank),
strengthening the existing column-norm-only checks.

### Class 4 — `TestRestrictionGroupCodecEdges`  (gap: codec for O1·O2 ≠ 16)
[test_combinatorics.py](tests/test_combinatorics.py) covers the `O1·O2=16` restriction-group
codec only. Add round-trip checks for `encode_restriction_groups`/`decode_restriction_groups`
at `O1·O2=4` (`O1=4,O2=1`, the degenerate single-selection case) and `O1=1,O2=4`,
asserting `decode(encode(groups)) == groups`.

## Critical files
- **New:** `tests/test_paper_correctness_extensions.py`
- **Reuse (no edits):** `tests/paper_oracles.py` (oracles + `regular_basis`/`ps_basis`),
  `nr_csi.config` (`SUPPORTED_N1N2`, `R16/R17/R18_PARAM_COMBOS`),
  `nr_csi.utils.combinatorics` (restriction-group codec),
  `nr_csi.channel` (`SyntheticRayChannel`/Gaussian) for valid selection inputs.
- **Pattern references:** valid-PMI-via-`select()` and per-shape combo selection mirror
  [test_paper_reconstruction_oracles.py](tests/test_paper_reconstruction_oracles.py);
  rank-3/4 table guard mirrors
  [test_config_tables.py:125](tests/test_config_tables.py#L125).

## Verification
```bash
.venv/bin/ruff check src tests scripts
.venv/bin/pytest -q tests/test_paper_correctness_extensions.py        # new module green
.venv/bin/pytest -q -m "not slow and not sionna"                      # full fast suite
.venv/bin/pytest -q -m slow                                           # full slow suite
git diff --stat                                                        # only the new test file
```
Acceptance:
- New module passes; full fast + slow suites still pass (no regressions).
- `git diff` shows **only** the added test file — no production or existing-test changes.
- If any new test unexpectedly fails, that surfaces a real defect — investigate the
  implementation rather than relaxing the test (consistent with the repo's tests-as-spec
  policy).
