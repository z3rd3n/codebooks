# Fix the investigated surprises from the codebook comparison figures

## Context

While generating the comparison figure gallery (PR #2, branch
`claude/gallant-edison-guo47o`), six results looked wrong and were
investigated; each now has a root cause documented in `results/*.md`. The
user wants the surprises *fixed* — not just explained. Three are genuine
gaps in the framework (a crash-prone ZF baseline, a delay-oblivious and
measurement-unfair harness, a misleading rank>1 fidelity metric), two are
spec-faithful behaviors that need regression locks so they never resurface
as "bugs" (R18's Q-doubled K₀ budget, the f2 L=1 overhead inversion), and
one is a structural spec property whose deployment-side remedy should be
demonstrated (windowed port selection). A seventh item is the PEB
normalization foot-gun that already caused a 9 dB SE inflation bug once.

User decisions (confirmed):
- Fixes live at the **library/harness level** (backward-compatible
  additions; defaults preserve current behavior bit-for-bit) so future ML
  comparisons inherit them.
- Spec-faithful surprises get **regression tests + README errata notes**.
- **`.venv` removal** from the git index (~4600 committed files of a macOS
  homebrew venv, already gitignored) goes in as a separate commit.

## The surprises and their fixes (summary)

| # | Surprise (where seen) | Root cause (verified) | Fix |
|---|---|---|---|
| S1 | Type I rank-2 column-SGCS 0.26 vs SE still decent (fig_01) | Rigid spec pair spans a good subspace (overlap 0.52) but can't rotate within it; column-wise SGCS punishes the rotation | New rotation-invariant `subspace_sgcs` metric, reported by the harness alongside SGCS |
| S2 | R18 ≥ R16 on *static* channels (fig_10/08) | Spec budget K₀ = ⌈2βLM₁Q⌉ — idle Doppler bin donates its half | Lock-in test + README note (no behavior change) |
| S3 | R18 "noise-robust" at low measurement SNR (fig_08) | It observes N₄=4 slots (iid noise averages); others observe 1 — unfair harness comparison | `evaluate(..., measurement_slots=)` fairness knob |
| S4 | R18 ≈ R16 in the harness aging panel (fig_05 right) | Harness applies reports delay-obliviously: predicted interval s scored against slot s+d | `evaluate(..., delay_aware=)` knob |
| S5 | R15/R16-PS lose to regular codebooks even in the beam domain (fig_09) | Spec PS basis is a *consecutive port window*; DFT-PEB energy lands on non-adjacent ports | `BeamDomainChannel(sort_by_energy=)` tuned-PEB demo in figlib + fig_09 third bar group |
| S6 | Plain ZF crash on colinear reported directions (fig_06, ε-RZF workaround) | `baselines.zf`/`ezf(xi=0)` use `np.linalg.inv` on a singular Gram | Switch to `np.linalg.pinv`; revert the fig_06 ε hack |
| S7 | PEB normalization foot-gun (caused 9 dB SE inflation, fixed locally in figlib) | `dft.orthogonal_group` rows have norm √(N₁N₂) | `dft.unitary_peb` helper + docstring warning + unitarity test |
| S8 | f2 overhead inversion at L=1: R16 (744) > R15 (488) bits (f2.png) | Paper's own Table bit1/bit2 formulas; R16's fixed machinery not amortized at L=1 (erratum 4 territory) | Golden assertion + README erratum extension |

(Detailed design below.)

---

## Phase 0 — Repo hygiene (separate first commit)

`git rm -r --cached .venv` (~4600 files leave the index; `.gitignore` already
lists `.venv/` so it stays out). Working tree untouched. Commit message:
"Remove committed macOS virtualenv from the git index".

Note: the working environment must be recreated in a fresh container:
`python3.12 -m venv /tmp/venv && /tmp/venv/bin/pip install -e ".[dev]"`.

## Phase 1 — Library fixes (one commit with Phase 2)

### S6: singularity-robust ZF — `src/nr_csi/baselines/ideal.py`

* `zf(H)` (line ~35): replace `H.conj().T @ np.linalg.inv(H @ H.conj().T)`
  with `np.linalg.pinv(H)` — mathematically identical for full row rank
  (right pseudo-inverse), minimum-norm and finite when users report
  colinear directions (they then share the direction and fully interfere —
  the physically right outcome). Docstring updated accordingly.
* `ezf(...)` (line ~107): when `xi == 0`, return
  `np.linalg.pinv(V_eff.conj().T)`; keep the regularized-`inv` branch for
  `xi > 0`. Existing `tests/test_baselines_appendix.py::TestEZF::
  test_zero_eigen_leakage_without_regularization` and
  `TestBD::test_zero_interuser_leakage` must keep passing (pinv ≡ inv to
  ~1e-12 for well-conditioned inputs).

### S7: PEB helper — `src/nr_csi/utils/dft.py`

```python
def unitary_peb(cfg: AntennaConfig, q1: int = 0, q2: int = 0) -> np.ndarray:
    """Unitary full-connect PEB F (P/2 x P/2): orthogonal_group(...).T
    rescaled by 1/sqrt(N1*N2) so F^H F = I (power-preserving -- SE computed
    in the beam domain equals the physical domain)."""
```
Plus one docstring warning line on `orthogonal_group` ("rows have norm
sqrt(N1*N2), not 1"). Do NOT change `orthogonal_group` itself (9 call
sites incl. the `codebooks/_spatial.py` hot path).

### S1: rotation-invariant metric — `src/nr_csi/metrics/similarity.py`

```python
def subspace_sgcs(W_ref: np.ndarray, W_hat: np.ndarray) -> float:
    """Fraction of each reference column's energy captured by span(W_hat),
    mean over columns/leading axes. Equals sgcs at v=1; >= sgcs always
    (each w_hat column lies in the span); invariant to W_hat -> W_hat @ U
    (unitary U). Complements the 3GPP column-wise SGCS at rank > 1, which
    also penalizes rotations within the reported subspace that do not
    affect the log-det rate (e.g. Type I's rigid rank-2 pair)."""
```
Implementation: batched `Q, R = np.linalg.qr(W_hat)` over leading axes;
`num = sum_i |Q^H w_ref|^2` per reference column; divide by column energy;
mask Q columns whose `|diag(R)|` is ~0 (so zero/degenerate `W_hat` columns
cannot spuriously capture energy; all-zero `W_hat` → 0.0, matching `sgcs`).
Export from `metrics/__init__.py`.

### S3 + S4: harness knobs — `src/nr_csi/eval/harness.py`

New `evaluate()` parameters (defaults reproduce current behavior
bit-for-bit, same rng consumption):

```python
measurement_slots: int | None = None,   # S3: UE observation window m
delay_aware: bool = False,              # S4: gNB indexes predicted intervals
```

Per-drop timeline (replaces the current 3 lines; `m = measurement_slots or
n_slots`, validate `m >= n_slots` else ValueError):

```python
total  = m + feedback_delay_slots
H_full = channel.generate(n_slots=total, rng=rng)
H_meas = H_full[:m]                                  # UE observation
# (optional measurement noise as today, applied to H_meas)
H_in   = H_meas[m - n_slots:] if n_slots > 1 else H_meas.mean(0, keepdims=True)
H      = H_full[total - n_slots:]                    # scoring window
pmi    = scheme.select(H_in, rank=rank)
```

Semantics: the UE observes `m` consecutive intervals ending at the report
instant; multi-interval schemes (R18) consume the most recent `n_slots` of
the noisy observation, single-interval schemes its time-average (a longer
CSI-RS observation — exactly the advantage R18 silently enjoyed; static +
noiseless ⇒ identical to today for any `m`).

W application (absolute-slot bookkeeping: the report's interval `s` maps to
absolute slot `(m - n_slots) + s`; scoring index `j` is absolute slot
`(m - n_slots) + d + j`, i.e. scheme-relative interval `d + j`):

```python
if delay_aware:
    idx = np.minimum(np.arange(H.shape[0]) + feedback_delay_slots, W.shape[0] - 1)
    W_all = W[idx]
elif S_out == H.shape[0]:
    W_all = W                                        # current behavior
else:
    W_all = np.repeat(W[-1:], H.shape[0], axis=0)    # current behavior
```

(`delay_aware` with `S_out == 1` or `d == 0` degenerates to the oblivious
path — asserted by tests.)

Also in `evaluate()`: compute `subspace_sgcs(W_ref, W_all)` per drop and
add `subspace_sgcs: float = 0.0` field to `EvalResult` (named-arg
construction everywhere — verified backward compatible). Docstring gains
the two knob descriptions.

## Phase 2 — Tests (same commit as Phase 1)

* `tests/test_eval_spine.py` — new `TestSubspaceSGCS`:
  `test_rank1_equals_sgcs`, `test_right_unitary_invariance`,
  `test_at_least_columnwise_sgcs` (random precoders),
  `test_zero_precoder_is_zero`,
  `test_type1_rank2_subspace_gap` (seeded `RandomRayChannel` drop:
  `subspace_sgcs − sgcs ≥ 0.1` for Type I rank 2 — the S1 regression).
* `tests/test_restrictions_and_harness.py::TestHarnessKnobs` — add:
  `test_measurement_slots_noiseless_noop` (static, no noise: `m=4` results
  equal `m=None` for R16, same seed),
  `test_measurement_slots_averages_noise` (static, −5 dB measurement SNR:
  R16 SGCS with `m=4` > with `m=1`, seeded margin),
  `test_measurement_slots_validation` (`m < n_slots` raises),
  `test_delay_aware_zero_delay_noop`, `test_delay_aware_single_interval_noop`
  (R16 d=2: identical under both modes),
  `test_delay_aware_recovers_r18_prediction` (on-grid Doppler
  `SyntheticRayChannel`, R18 N4=4, d=2: delay-aware SGCS > oblivious + 0.05
  and within 0.05 of the d=0 value).
* `tests/test_baselines_appendix.py` — new `TestZFRobustness`:
  `test_pinv_matches_inv_when_well_conditioned` (rtol 1e-9),
  `test_colinear_users_graceful` (duplicated row: finite output, duplicate
  users share a direction),
  `test_mu_eval_many_type1_users_completes` (`evaluate_mu(Type1Codebook,
  ..., n_users=8, n_drops=5, regularization=None)` — the original crash
  repro — returns finite sum rates).
* `tests/test_dft_bases.py` — `test_unitary_peb_is_unitary` (FᴴF ≈ I for
  (4,2) and (8,1)).
* Lock-ins: `tests/test_compression_properties.py` — new
  `TestR18StaticBudget::test_static_r18_at_least_matched_r16` (R16 pc6 vs
  R18 pc7 N4=4 on ~20 static seeded drops: mean SGCS(R18) ≥ mean SGCS(R16)
  − 0.005 AND `total_overhead_bits` strictly larger — the S2 lock; the K₀
  formula itself is already tested in `test_etype2_r18.py`).
  `tests/test_overhead.py::TestF2Claims` — add
  `test_l1_inversion_r16_exceeds_r15` (`r16[0] > r15[0]` with an
  erratum-4 comment — the S8 lock).

## Phase 3 — Figure/doc updates (third commit)

* `scripts/figlib.py`: `BeamDomainChannel` uses `dft.unitary_peb`; gains
  `sort_by_energy: bool = False` — per-drop permutation of beam ports by
  descending energy (summed over slots, frequency, rx, and BOTH
  polarization halves; the SAME permutation applied to each half preserves
  the dual-pol structure; a permutation is unitary so comparability holds).
  Label in figures: "tuned PEB (per-drop sorted — upper bound of PEB
  tuning)".
* `scripts/fig_01_se_vs_snr.py`: store `subspace_sgcs` per scheme/rank in
  the JSON next to `sgcs`.
* `scripts/fig_05_mobility.py`: right panel adds "R18 … (delay-aware)"
  curve via `run_eval(..., delay_aware=True)` — the harness now shows the
  prediction gain the left panel showed by hand.
* `scripts/fig_06_mu_mimo.py`: remove `EPS_REG`, pass
  `regularization=None` (plain ZF is now pinv-robust); docstring note.
* `scripts/fig_08_channel_sensitivity.py`: right panel adds dashed
  "R16 eType II (4-slot meas.)" via `measurement_slots=4` — the fair
  comparison to R18's window.
* `scripts/fig_09_port_selection.py`: third bar per scheme on the
  energy-sorted beam domain — windowed R15/R16-PS recover; R17 ~unchanged
  (free selection is order-invariant — the control).
* Regenerate fig_01, 05, 06, 08, 09 (full quality) and update their
  `results/<name>.md` analyses (one short "fix landed" paragraph each).
* `README.md`: two errata/notes bullets (R18 K₀ = ⌈2βLM₁Q⌉ consequence;
  f2 L=1 inversion under the table formulas); mention `subspace_sgcs` in
  the metrics paragraph of "Comparing your ML CSI algorithm".

## Verification

1. `/tmp/venv/bin/ruff check src tests scripts` — clean.
2. `pytest -m "not slow and not sionna"` — all (~380 existing + ~15 new)
   green; existing EZF/BD leakage tests confirm pinv equivalence.
3. `pytest -m slow` — unaffected but run once as insurance.
4. `python scripts/make_all_figures.py --fast` — all 12 smoke-pass.
5. Regenerate the five affected figures at full quality; sanity-diff
   `results/fig_06_mu_mimo.json` against the committed ε-RZF version
   (sum rates within Monte-Carlo tolerance, ~±0.3 b/s/Hz) and confirm in
   `fig_05` that the delay-aware R18 aging curve now sits clearly above
   R16's (the gain the left panel already showed).
6. Three commits in order (hygiene / library+tests / figures+docs);
   `git push -u origin claude/gallant-edison-guo47o` — updates PR #2.