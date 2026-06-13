# Algorithmic Test Failure Remediation Report

Date: 2026-06-12

## Purpose

This report explains every currently failing algorithmic test and gives a
decision-complete production-code remediation plan for another LLM or engineer.

Do not delete, relax, skip, or mark these tests as `xfail`. They expose input
validation and malformed-wire-format defects. The test expansion was explicitly
implemented under a tests-only policy, so production fixes remain outstanding.

## Current Verification State

Commands:

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/pytest -q -m "not slow and not sionna"
.venv/bin/pytest -q -m slow
```

Current results:

- Ruff: passes.
- Fast suite: `8 failed, 974 passed, 542 deselected`.
- Slow suite: `534 passed, 990 deselected`.
- Total collection: `1524 tests`.

The eight failing nodes represent four root causes:

| Failing nodes | Root cause |
|---|---|
| 2 | `SubbandConfig` accepts zero/negative subband counts |
| 2 | R16 and R18 do not validate protocol parameter `R` |
| 1 | R17 permits `M=2` when only one frequency unit exists |
| 3 | Deserializers use `assert` for trailing-bit rejection |

## Failure 1: Zero Subband Count Is Accepted

### Test node

```text
tests/test_paper_exhaustive_algorithms.py::test_subband_configuration_rejects_nonpositive_count[0]
```

### Expected behavior

`SubbandConfig(n_subbands=0)` must raise `ValueError`.

`n_subbands` is a count used in:

```text
N3 = n_subbands * R
```

A zero count creates `N3=0`, which is not a valid PMI frequency-domain size.

### Actual behavior

Construction succeeds.

### Root cause

In `src/nr_csi/config.py`, `SubbandConfig.__post_init__()` enters the invalid
range branch, but only raises when `n_subbands > 19`:

```python
if not 1 <= self.n_subbands <= 19:
    if self.n_subbands > 19:
        raise ValueError(...)
```

For zero, the outer condition is true and the inner condition is false, so the
method returns normally.

### Required fix

Make the lower and upper bounds explicit:

```python
if self.n_subbands < 1:
    raise ValueError("n_subbands must be positive")
if self.n_subbands > 19:
    raise ValueError("at most 19 CQI subbands are configurable")
```

Keep the repository's deliberate allowance for small positive unit-test values
such as `1` and `2`. Do not change the minimum to the paper's operational value
of `3`, because existing tests intentionally exercise smaller positive domains.

## Failure 2: Negative Subband Count Is Accepted

### Test node

```text
tests/test_paper_exhaustive_algorithms.py::test_subband_configuration_rejects_nonpositive_count[-1]
```

### Expected behavior

`SubbandConfig(n_subbands=-1)` must raise `ValueError`.

### Actual behavior

Construction succeeds and the `N3` property becomes negative.

### Root cause

This is the same missing lower-bound raise described for Failure 1.

### Required fix

The `self.n_subbands < 1` guard above fixes both test nodes. Verify both
parameters independently.

## Failure 3: `R=0` Produces Division by Zero

### Test node

```text
tests/test_paper_exhaustive_algorithms.py::test_r16_and_r18_reject_r_outside_protocol_domain[0]
```

### Expected behavior

Both constructors must reject `R=0` with `ValueError`:

```python
R16Type2Codebook(antenna, N3=12, R=0)
R18Type2Codebook(antenna, N3=12, N4=4, R=0)
```

The paper defines `R` as one or two PMI matrices per CQI subband:

```text
R in {1, 2}
```

### Actual behavior

R16 reaches:

```python
M_v = ceil(p_v * N3 / R)
```

and raises:

```text
ZeroDivisionError: Fraction(1, 0)
```

The parametrized test stops at R16, so its R18 assertion is not reached.
Independent reproduction confirms R18 has the same `ZeroDivisionError`.

### Root cause

`R16Type2Codebook.__init__()` and `R18Type2Codebook.__init__()` assign `R`
without validating it. Their constructor-time `Mv()` checks call `m_v()`,
which divides by `R`.

Relevant code:

- `src/nr_csi/codebooks/etype2_r16.py`, constructor and `Mv()`
- `src/nr_csi/codebooks/etype2_r18.py`, constructor and `Mv()`
- `src/nr_csi/config.py`, `m_v()`

`SubbandConfig` and `n3_for_bwp()` already enforce `R in (1, 2)`, so the
codebook constructors are inconsistent with the rest of the public API.

### Required fix

Add the same early guard to both constructors, before any `Mv()` call:

```python
if R not in (1, 2):
    raise ValueError("R must be 1 or 2")
```

Also harden the public `m_v()` helper with the same guard. Constructor guards
give callers a clear failure boundary; the helper guard prevents direct calls
or future code paths from reintroducing division-by-zero behavior.

Do not catch `ZeroDivisionError`; reject the invalid domain before calculation.

## Failure 4: `R=3` Is Silently Accepted

### Test node

```text
tests/test_paper_exhaustive_algorithms.py::test_r16_and_r18_reject_r_outside_protocol_domain[3]
```

### Expected behavior

Both R16 and R18 constructors must raise `ValueError` because `R=3` is outside
the protocol domain `{1, 2}`.

### Actual behavior

R16 constructs successfully. Once R16 is fixed, the test proceeds to R18,
which currently also constructs successfully. Independent reproduction
confirmed both implementations accept `R=3`.

The resulting `Mv` and `K0` values are mathematically computable but represent
an unsupported configuration, so accepting them hides caller/configuration
errors.

### Root cause

Same missing constructor and helper validation as Failure 3.

### Required fix

The `R not in (1, 2)` guards described above must be applied to:

1. `R16Type2Codebook.__init__`
2. `R18Type2Codebook.__init__`
3. `config.m_v`

Add focused direct tests for `m_v(..., R=0)` and `m_v(..., R=3)` if the helper
is hardened.

## Failure 5: R17 Requests Two Taps From One Frequency Unit

### Test node

```text
tests/test_paper_exhaustive_algorithms.py::test_r17_rejects_two_taps_when_only_one_frequency_unit_exists
```

### Expected behavior

This constructor must raise `ValueError`:

```python
R17Type2Codebook(
    antenna,
    N3=1,
    param_combination=5,  # M=2
)
```

A codebook cannot select two distinct delay taps from an `N3=1` DFT domain.

### Actual behavior

Construction succeeds with:

```text
M=2, N3=1, N_window=4
```

Calling `select()` then fails later with:

```text
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
```

The late exception occurs because:

1. The selection condition for reporting `i16` is false when
   `min(N_window, N3) == 1`.
2. `pmi.i16` remains `None`.
3. `taps()` takes neither the `M == 1` branch nor the `min(...) == 2` branch.
4. It evaluates `pmi.i16 + 1`.

### Root cause

`R17Type2Codebook.__init__()` checks only that `N3` is positive. It does not
enforce the fundamental selection constraint:

```text
M <= N3
```

### Required fix

After `self.M` and `N3` are known, reject impossible configurations:

```python
if self.M > N3:
    raise ValueError(f"M={self.M} selected taps cannot exceed N3={N3}")
```

Do not silently degrade `M=2` to `M=1`; `M` is fixed by the selected R17
parameter-combination table row, so changing it would alter the configured
codebook semantics.

Optionally add a defensive assertion or `ValueError` in `taps()` for malformed
internal/PMI state, but constructor rejection is the primary fix required by
the test.

## Failure 6: One Trailing Zero Bit Raises `AssertionError`

### Test node

```text
tests/test_paper_metrics_serialization_baselines.py::test_unpack_rejects_trailing_bits_as_malformed_stream[0]
```

### Expected behavior

Appending one unused zero bit to a valid Type I bitstream must cause
`unpack()` to raise `ValueError`, identifying malformed external input.

### Actual behavior

`unpack_type1()` consumes the valid prefix and then executes:

```python
assert r.done()
```

This raises bare `AssertionError`.

### Root cause

The decoder uses an internal assertion for input validation. Assertions are
the wrong contract for malformed wire data and can be removed entirely when
Python runs with optimization (`python -O`).

### Required fix

Introduce one shared helper in `src/nr_csi/codebooks/serialize.py`:

```python
def _require_done(reader: BitReader) -> None:
    if not reader.done():
        remaining = len(reader.bits) - reader.pos
        raise ValueError(f"bitstream has {remaining} trailing bit(s)")
```

Replace every `assert r.done()` with `_require_done(r)`.

There are five affected unpackers:

1. `unpack_type1`
2. `unpack_r15`
3. `unpack_r16`
4. `unpack_r17`
5. `unpack_r18`

Do not fix only Type I. The test currently demonstrates Type I, but all family
decoders have the same defect.

## Failure 7: One Trailing One Bit Raises `AssertionError`

### Test node

```text
tests/test_paper_metrics_serialization_baselines.py::test_unpack_rejects_trailing_bits_as_malformed_stream[1]
```

### Expected behavior

Same as Failure 6: malformed trailing data must raise `ValueError`, regardless
of bit value.

### Actual behavior

Bare `AssertionError`.

### Root cause

Same `assert r.done()` misuse.

### Required fix

The shared `_require_done()` replacement fixes this node. The decoder must not
interpret or ignore the trailing bit.

## Failure 8: Multiple Trailing Bits Raise `AssertionError`

### Test node

```text
tests/test_paper_metrics_serialization_baselines.py::test_unpack_rejects_trailing_bits_as_malformed_stream[101]
```

### Expected behavior

Three trailing bits must raise `ValueError`.

### Actual behavior

Bare `AssertionError`.

### Root cause

Same `assert r.done()` misuse.

### Required fix

Use the shared `_require_done()` helper. Including the remaining-bit count in
the error message makes diagnosis deterministic:

```text
bitstream has 3 trailing bit(s)
```

## Recommended Additional Regression Coverage

The eight existing failures are sufficient to drive the required fixes. While
editing production code, add or extend tests for these adjacent contracts:

1. `SubbandConfig(n_subbands=1)` and `SubbandConfig(n_subbands=19)` remain valid.
2. `SubbandConfig(n_subbands=20)` remains invalid.
3. R16 and R18 accept `R=1` and `R=2`.
4. Direct `m_v()` calls reject `R=0` and `R=3` with `ValueError`.
5. R17 accepts `M=1, N3=1`.
6. R17 accepts `M=2, N3=2`.
7. Trailing bits are rejected with `ValueError` for all five serializer
   families, not only Type I.
8. Truncated streams must continue raising `ValueError("bitstream exhausted")`.

## Implementation Order

1. Fix `SubbandConfig` lower-bound validation in `config.py`.
2. Validate `R` in `m_v()`, R16 construction, and R18 construction.
3. Validate `M <= N3` in R17 construction.
4. Replace all five serializer completion assertions with `_require_done()`.
5. Run targeted failing tests.
6. Run the complete fast and slow suites.

## Targeted Verification

```bash
.venv/bin/pytest -q \
  tests/test_paper_exhaustive_algorithms.py::test_subband_configuration_rejects_nonpositive_count \
  tests/test_paper_exhaustive_algorithms.py::test_r16_and_r18_reject_r_outside_protocol_domain \
  tests/test_paper_exhaustive_algorithms.py::test_r17_rejects_two_taps_when_only_one_frequency_unit_exists \
  tests/test_paper_metrics_serialization_baselines.py::test_unpack_rejects_trailing_bits_as_malformed_stream
```

Expected result after production fixes:

```text
8 passed
```

Full acceptance:

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/pytest -q -m "not slow and not sionna"
.venv/bin/pytest -q -m slow
```

Expected acceptance criteria:

- No failing tests.
- No test is deleted, weakened, skipped, or marked `xfail`.
- Invalid public inputs fail early with `ValueError`, not incidental
  `ZeroDivisionError`, `TypeError`, or `AssertionError`.
- No supported valid configuration changes behavior.
