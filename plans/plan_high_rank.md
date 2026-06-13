# Complete the non-CJT 3GPP NR PMI Codebooks (TS 38.214 §5.2.2.2)

## Context

`nr_csi` implements five PMI codebooks as trustworthy NumPy baselines for benchmarking a
future ML CSI scheme ([memory: nr-csi-project-goal]). Audited against TS 38.214 §5.2.2.2,
the implemented families are complete to their spec rank ranges:

| Spec § | Codebook | Status |
|---|---|---|
| 5.2.2.2.1 | Type I Single-Panel | **ranks 1–2 only** — missing 3–8 → **this plan** |
| 5.2.2.2.2 | Type I Multi-Panel | **missing entirely** → **this plan** |
| 5.2.2.2.3 | Type II (R15) | done (1–2; spec caps RI≤2) |
| 5.2.2.2.4 | Type II Port-Selection (R15) | done |
| 5.2.2.2.5 | Enhanced Type II (R16) | done (1–4) |
| 5.2.2.2.6 | Enhanced Type II PS (R16) | done |
| 5.2.2.2.7 | Further-enhanced Type II PS (R17) | done (1–4) |
| 5.2.2.2.8 | Enhanced Type II **for CJT** (R18) | missing — **deferred (later effort)** |
| 5.2.2.2.9 | Further-enh. Type II PS **for CJT** (R18) | missing — **deferred (later effort)** |
| 5.2.2.2.10 | Enhanced Type II predicted-PMI (R18 Doppler) | done (1–4) |
| 5.2.2.2.11 | Further-enh. Type II PS predicted-PMI (R18) | **missing** → **this plan** |

**Scope of this plan (3 gaps):** Type I Single-Panel ranks 3–8, Type I Multi-Panel, and the
predicted-PMI port-selection codebook. **The two CJT codebooks (§5.2.2.2.8/9) are explicitly
out of scope** and will be done separately — none of the work below adds an `N_TRP` resource
dimension or otherwise touches the multi-TRP channel/harness path, so nothing currently
working is put at risk.

Each gap gets the full pipeline the existing codebooks have — `precoder` / `select` /
`overhead_bits` / `serialize.pack`+`unpack` / `validate_*` — wired into `evaluate`, the
figure scripts, and the test matrix.

### Two hard constraints discovered during exploration (these shape the whole plan)

1. **The spec markdown is degraded for exactly the missing pieces.** In
   [specs/38214-i00.md:8955-9249](specs/38214-i00.md#L8955-L9249) every Type I precoder
   *W*-matrix equation and the body of Tables 5.2.2.2.1-3 … -12 (and the §5.2.2.2.2 *W*
   equations) were images and are now **empty cells**.
2. **The in-repo paper (`paper/main.tex`) omits all of this too** (README erratum 6:
   "Type I ranks 3–8 need TS 38.214 tables that the paper does not include"). So there is
   **no in-repo golden source** for the new precoders.

**Consequence (decided):** reconstruct the formulas from the TS 38.214 structure and pin
correctness with an **external reference oracle** — golden precoder matrices generated
offline (MATLAB 5G Toolbox primary; Sionna where it has coverage), committed as fixtures,
and asserted against `precoder()`. Property/structural tests and an independent in-repo
closed-form oracle (the existing `tests/paper_oracles.py` pattern) back this up wherever the
external tool cannot reach.

### What informs the ordering

- **§5.2.2.2.11 is nearly free.** The spec fixes **N4 = 1** and states the precoder is
  obtained "as in Table 5.2.2.2.7-3" — i.e. it is the R17 codebook with the predicted-PMI
  report framing. Almost entirely a config + thin-wrapper job over existing R17 code, with
  no Doppler/TRP machinery.
- **Type I ranks 3–8** is the explicit user ask and the highest-value gap.
- **Type I Multi-Panel** reuses Phase 1's rank-2/3/4 beam maps + adds inter-panel co-phasing.

---

## Conventions to reuse (do not reinvent)

- Spatial beams / groups: [dft.spatial_beam](src/nr_csi/utils/dft.py#L33),
  [dft.orthogonal_group](src/nr_csi/utils/dft.py#L57),
  [dft.unitary_peb](src/nr_csi/utils/dft.py#L75).
- Combinatorial codecs (Algs 1–4): [utils/combinatorics.py](src/nr_csi/utils/combinatorics.py).
- Quantizers: [utils/quantization.py](src/nr_csi/utils/quantization.py).
- UE-side helpers: [_spatial.py](src/nr_csi/codebooks/_spatial.py)
  `aligned_eigen_targets`, `select_group_and_beams`, `ls_coefficients` — generalize, don't fork.
- Eigen targets / rate: [baselines/ideal.py](src/nr_csi/baselines/ideal.py) `eigen_precoder`
  already supports arbitrary rank; metrics `se`, `sgcs`, `subspace_sgcs`, `su_rate`.
- Serialization primitives: [serialize.py](src/nr_csi/codebooks/serialize.py)
  `BitWriter`/`BitReader`/`_w`.
- Validation primitives: [validate.py](src/nr_csi/codebooks/validate.py)
  `_check`/`_check_array`/`_validate_spatial`/`_validate_coefficients`.
- Test patterns: [tests/paper_oracles.py](tests/paper_oracles.py) (independent closed-form
  precoders), [tests/test_invariants.py](tests/test_invariants.py) (`make_schemes` +
  `check_precoder` property loop), [tests/test_config_tables.py](tests/test_config_tables.py)
  (re-transcribe tables independently).

---

## Phase 0 — External-oracle fixture infrastructure (prerequisite)

**New:** `tests/oracle/` package + `tests/fixtures/oracle/*.npz` + generator scripts.

1. `scripts/gen_oracle_fixtures.m` (MATLAB 5G Toolbox): for a grid of
   `(P, N1, N2, Ng, codebookType, codebookMode, rank, i1…, i2)` emit the golden precoder
   `W[t, port, layer]` and chosen indices to `*.npz`/JSON. Cover Type I SP 1–8 and Type I MP
   1–4; also dump eType II for a few configs as a *self-check that the oracle harness itself
   is correct* against the already-trusted code.
2. `scripts/gen_oracle_fixtures_sionna.py`: Sionna fallback for configs MATLAB lacks.
3. `tests/oracle/loader.py`: load a fixture → `(scheme_kwargs, pmi_fields, W_golden)`.
4. `tests/oracle/compare.py`: `assert_precoder_matches(W, W_golden)` comparing **up to a
   per-(interval,t,layer) global phase** (an unobservable unit-modulus factor), else exact
   within `atol=1e-6`. Reuse the SGCS idea from `metrics.similarity` for the phase-blind compare.
5. `tests/oracle/closed_form.py`: extend the `paper_oracles.py` style with **independent**
   closed-form TS 38.214 reconstructions (no `nr_csi` imports) for a second opinion when no
   external fixture exists.

Fixtures are committed; tests never invoke MATLAB at run time. Add a `@pytest.mark.oracle`
marker (in `pyproject.toml`) gating *regeneration*; fixture-compare tests run in the normal
suite. Produce the Type I fixtures alongside Phase 1.

---

## Phase 1 — Type I Single-Panel ranks 3–8 (§5.2.2.2.1)

**Files:** [codebooks/type1.py](src/nr_csi/codebooks/type1.py) (extend), `validate.py`,
`serialize.py`, `tests/test_type1_higher_ranks.py` (new), extend `tests/test_invariants.py`,
`tests/test_pmi_composition.py`, `tests/test_serialize.py`, `tests/test_overhead.py`.

### Spec structure to transcribe (anchors: Tables 5.2.2.2.1-4, -7…-12)
Composite index: ranks 3–4 use `i_{1,1},i_{1,2},i_{1,3}` for `P<16` and add `i_{1,4}` for
`P∈{16,24,32}`; ranks 5–8 use `i_{1,1},i_{1,2},i_{1,3}` with fixed beam patterns; `i_2` is
the per-subband co-phase `n` (`φ_n = e^{jπn/2}`). All columns are unit-modulus DFT beams
`v_{l,m}` from `dft.spatial_beam`, scaled `1/√(ν·P)` so `tr(WᴴW)=1`.

**Confident structures (implement directly, then oracle-verify):**
- Rank 3, `P<16`: `W = 1/√(3P) [[v₁, v₂, v₁],[φv₁, φv₂, −φv₁]]`
- Rank 4, `P<16`: `W = 1/√(4P) [[v₁, v₂, v₁, v₂],[φv₁, φv₂, −φv₁, −φv₂]]`
  where `v₁=v_{l,m}`, `v₂=v_{l+k₁,m+k₂}`, `(k₁,k₂)` from **Table 5.2.2.2.1-4** (distinct from
  the rank-2 Table -3 already encoded in [i13_offsets](src/nr_csi/codebooks/type1.py#L33)).

**Oracle-pinned structures (skeleton known, exact beam offsets/co-phase columns taken from
the fixture):** ranks 3–4 for `P≥16` (the four-beam `i_{1,4}` variant) and ranks 5–8 (the
fixed multi-beam patterns of Tables -9…-12). Implement `_w_rankN(...)` to *match the
committed golden columns*; the generator script is the source of truth for the offset tables.

### Pipeline changes
- **Data model:** extend `Type1PMI` with `i14: int | None = None` (ranks 3–4, `P≥16`); keep
  `i2` per-subband.
- **`precoder`:** dispatch on `rank` to `_w_rank{1..8}`; add `i13_offsets_rank34(...)`
  (Table -4) and the `P≥16`/5–8 builders. Generalize the `1/√(ν·P)` scaling.
- **`select`:** generalize the rank-2 branch
  ([type1.py:186-206](src/nr_csi/codebooks/type1.py#L186-L206)) to general ν using the full
  `log2 det(I + (ρ/ν)(HW)ᴴHW)` rate (replace the hand-expanded 2×2 with `np.linalg.slogdet`),
  enumerating candidate `(i11,i12,i13[,i14])` and per-subband `n`. Enforce
  `typeI-SinglePanel-ri-Restriction` (already an 8-bit field) and beam restriction across all
  beams the chosen rank uses; honour "shall not report `i_{1,2}` when `N2=1`".
- **`overhead_bits`:** add `i13`/`i14` widths per rank and `P` regime; `i2` width per rank.
- **`serialize`:** extend `pack_type1`/`unpack_type1` for `i13`/`i14` by rank; keep the
  round-trip invariant.
- **`validate_type1`:** lift the `rank in (1,2)` cap; range-check `i13`/`i14` per rank/`P`.

### Tests (`tests/test_type1_higher_ranks.py` + extensions)
- **Oracle match:** `precoder()` vs committed golden for every supported
  `(P, mode, rank, i1, i2)` in the fixture grid (phase-blind compare).
- **Closed-form cross-check:** vs `tests/oracle/closed_form.py` for ranks 3–4 `P<16` on all
  `(N1,N2)` of Table 5.2.2.2.1-2.
- **Semi-unitarity / power:** `WᴴW = I_ν/ν` ⇒ `tr(WᴴW)=1`; each column norm `1/√ν`; columns
  mutually orthogonal.
- **Rank validity per P:** ranks supported only where ports allow (rank 8 needs `P≥16`);
  unsupported `(P,rank)` raise `ValueError` at `select` and `precoder`.
- **RI restriction:** `ri_restriction[ν-1]=0` ⇒ `select(H,ν)` raises; positive control passes.
- **Beam restriction:** a forbidden beam used by the rank-ν pattern is never selected.
- **Selection sanity:** on an on-grid multi-ray channel matched to a rank-ν pattern, the
  codebook reaches the eigen bound (`su_rate` ≈ `eigen_precoder`), mirroring
  [test_type1.py:75-133](tests/test_type1.py#L75-L133).
- **Overhead:** golden bit dicts per rank/P; monotone vs rank; `i14` present iff `P≥16` & ranks 3–4.
- **Serialize round-trip:** `unpack(pack(pmi))==pmi` and `len(pack)==total_overhead_bits`
  for every supported `(P,mode,rank)` — add rows to `tests/test_serialize.py`.
- **Invariants:** add Type I ranks 3–8 to `make_schemes`/`check_precoder` in
  `test_invariants.py` (parametrized by `P`).
- **Harness integration:** `evaluate(Type1Codebook, …, rank∈{3,4})` runs end-to-end and
  `subspace_sgcs∈[0,1]` (extends `fig_07_rank_adaptation`).

---

## Phase 2 — Type I Multi-Panel (§5.2.2.2.2)

**New:** `codebooks/type1_multipanel.py` (`Type1MultiPanelCodebook`, `Type1MPPMI`); extend
`config.py`, `validate.py`, `serialize.py`; `tests/test_type1_multipanel.py`.

> Multi-panel is a *single-resource* codebook (`Ng` panels behind one CSI-RS resource), so it
> does **not** introduce the `N_TRP` cross-resource dimension that the deferred CJT codebooks
> need. The only new array axis is the panel index inside the existing port axis.

### Spec structure (anchors: Tables 5.2.2.2.2-1…-6)
`Ng∈{2,4}` panels, each a UPA `(N1,N2)`; `P_CSI-RS = 2·Ng·N1·N2 ∈ {8,16,32}` (Table
5.2.2.2.2-1, present in markdown — transcribe to a new `SUPPORTED_NG_N1N2`). codebookMode 1
or 2 (mode 2 only `Ng=2`). The precoder reuses the single-panel per-panel beam `v_{l,m}` and
adds **inter-panel co-phasing** (indices `i_{1,4}`, `i_{2,1}`, `i_{2,2}`); ranks 1–4 reuse the
single-panel layer/beam maps (Tables -3, -2). Exact phase columns are **oracle-pinned**.

### Pipeline
- **config:** `SUPPORTED_NG_N1N2` + `n_panels`/`P` helpers; give `AntennaConfig` an optional
  `Ng` (default 1 ⇒ single-panel, so existing behaviour is untouched) with panel-major port
  ordering `port = pol·Ng·N1N2 + panel·N1N2 + n1·N2 + n2`.
- **precoder/select/overhead/serialize/validate:** mirror Phase 1 plus the inter-panel
  co-phase field(s); `select` adds the per-panel co-phase search to the rate maximization.

### Tests (`tests/test_type1_multipanel.py`)
- Oracle match for all `(Ng,N1,N2)` of Table 5.2.2.2.2-1, modes 1/2, ranks 1–4.
- Semi-unitarity/power/column-orthogonality; `Ng=1` degeneracy ≡ single-panel `precoder`
  (PMI-level equality where structures coincide).
- mode-2-requires-`Ng=2` and `N2=1`-no-`i_{1,2}` guards raise.
- RI/`ng-n1-n2` restriction honoured; overhead golden dicts; serialize round-trip;
  invariants-loop entry; harness integration on a `P=16` channel.

---

## Phase 3 — Further-enhanced Type II PS for predicted PMI (§5.2.2.2.11) — lightest

**New:** `codebooks/predicted_ps_r18.py` (thin subclass/wrapper over
[R17Type2Codebook](src/nr_csi/codebooks/fetype2_r17.py)); `tests/test_predicted_ps.py`.

Spec fixes **N4=1** and obtains the precoder "as in Table 5.2.2.2.7-3" — i.e. R17. Reuse the
R17 `precoder`/`select`/`overhead_bits`/`pack`/`validate` unchanged; the new class adds only
the predicted-PMI report framing: `paramCombination-Doppler-PS-r18` (reuses
[R17_PARAM_COMBOS](src/nr_csi/config.py#L170), Table 5.2.2.2.7-1), `valueOfN-Doppler-r18`
(N∈{2,4} when M=2), `R` handling, and `typeII-Doppler-PS-RI-Restriction-r18` (4-bit).

### Tests (`tests/test_predicted_ps.py`)
- Behavioural equivalence to R17 at N4=1: precoder/overhead/serialize equality.
- RI-restriction rejection; documented config exclusions raise `ValueError`.
- One harness run; serialize round-trip; invariants-loop entry.

---

## Cross-cutting wiring (every phase)

- Export new classes from [codebooks/__init__.py](src/nr_csi/codebooks/__init__.py); add
  `pack`/`unpack` dispatch branches in
  [serialize.py:413-450](src/nr_csi/codebooks/serialize.py#L413-L450).
- Add `validate_*` for each new family; `precoder()` calls it first (gNB rejects malformed
  reports), matching every existing codebook.
- Extend the README "Implemented codebooks" table, the traceability matrix, and the
  out-of-scope note (keep CJT listed as deferred). Add Type I ranks 3–8 to
  `fig_07_rank_adaptation` where meaningful.
- Update [memory: nr-csi-project-goal] once landed.

## Verification (end-to-end)

```bash
# fast spec/unit suite incl. new families (excludes oracle-regen + sionna)
pytest -m "not slow and not sionna and not oracle"
# external-oracle fixture comparisons (fixtures committed; no MATLAB needed)
pytest tests/test_type1_higher_ranks.py tests/test_type1_multipanel.py -k oracle
# property/invariants across every (N1,N2)[,Ng] and rank
pytest tests/test_invariants.py
# serialize round-trip + overhead golden dicts
pytest tests/test_serialize.py tests/test_overhead.py
# regenerate fixtures (developer-only; needs MATLAB 5G Toolbox / Sionna)
pytest -m oracle            # or: matlab -batch "run scripts/gen_oracle_fixtures.m"
scripts/check.sh            # ruff + fast + slow
```
Per-phase acceptance: (a) `precoder` matches the external/closed-form oracle phase-blind;
(b) semi-unitarity + `tr(WᴴW)=1`; (c) `unpack(pack)==pmi` and `len(pack)==total_overhead`;
(d) `validate_*` rejects out-of-range/strongest-coefficient violations; (e) degeneracy
identities hold (Ng=1 ≡ single-panel; predicted-PS N4=1 ≡ R17); (f) `evaluate` runs and
`subspace_sgcs∈[0,1]`. **Regression guard:** the full existing suite must stay green — no
existing test modified, skipped, or weakened.

## Risks / open questions

- **Oracle coverage.** MATLAB 5G Toolbox / Sionna DL Type I coverage is version-dependent.
  Mitigation: the independent in-repo closed-form oracle (`tests/oracle/closed_form.py`) is
  the fallback, with property tests; flag any config lacking an external golden as
  `@pytest.mark.xfail(reason="no external oracle")` rather than asserting on a self-derived
  value alone.
- **Exact Type I 5–8 beam-offset tables** (Tables 5.2.2.2.1-4 four-index variant and -9…-12)
  are not in the repo; they must come from the oracle / an authoritative TS 38.214 copy. The
  plan implements the confident `P<16` rank-3/4 forms directly and treats the rest as
  "match the committed fixture".
- **Effort ordering recommendation:** Phase 3 (cheap win) → Phase 1 (the explicit ask) →
  Phase 2, with Phase 0 fixtures produced alongside Phase 1. CJT (§5.2.2.2.8/9) remains a
  separate future effort and is intentionally excluded here.
```
