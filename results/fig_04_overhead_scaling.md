# fig_04_overhead_scaling.png — overhead scaling laws (Tables bit1/bit2)

**What it shows.** Pure formula evaluation (no Monte-Carlo) of the paper's
bit-overhead tables at the f2 operating point ((16,1), rank 2), swept along
the three axes that grow in deployments: frequency granularity N₃, spatial
bases L, and time coverage N₄. Log-scale y.

**Why it looks like this.**

* **Left (vs N₃):** R15 grows linearly — its i₂ block is re-sent per
  subband — reaching ~3000 b at N₃ = 72. R16/R18 grow only through
  M_v = ⌈N₃/4⌉ (bitmap and coefficient count ∝ M_v, tap indicator
  ~log C(2M_v−1, M_v−1)), i.e. sub-linearly with visible ceil() staircase
  steps. The R15/R16 ratio therefore *widens* with bandwidth: ~1.5× at
  N₃ = 4, ~3× at 18, ~4× at 72.
* **Middle (vs L):** all three families grow at the same rate in L (the
  paper's claim) because the dominant terms — R15's per-subband 2L−1
  coefficients, R16/R18's 2L·M_v bitmap + K^NZ ∝ βL coefficients — are all
  linear in L; the combinatorial i₁,₂ = log₂C(16, L) adds only a few bits.
  The vertical offsets are the per-generation fixed machinery.
* **Right (vs N₄):** the only panel where the *ordering inverts*. R15/R16
  must re-report every interval (lines ∝ N₄); R18 sends one predicted
  report whose size barely grows with N₄ (only i₁,₁₀ = ⌈log₂(N₄−1)⌉ per
  layer; the Q = 2 bitmap duplication is paid once). At N₄ = 1, R18
  degenerates to R16 (identical bits, Q = 1); by N₄ = 8 it is ~7× cheaper
  than R16 and ~20× cheaper than R15 for the same time coverage.

**Notable.** K^NZ is held at its budget K₀ = ⌈β·2L·M_v⌉ throughout, so the
curves are upper envelopes; a UE reporting fewer nonzero coefficients
shrinks only the i₂,₄/i₂,₅ terms. The same convention produced f2.png; the
absolute-bars discrepancy with the paper's figure (README erratum 4)
applies here too — trends and ratios are the reliable content.

**Config.** `scripts/fig_04_overhead_scaling.py`; p_v = 1/4, β = 1/2,
N_PSK = 4 (R15), Q = 2.
