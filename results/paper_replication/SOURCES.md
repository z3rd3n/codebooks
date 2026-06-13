# Paper sources for the `nr_csi` codebook-figure replication

Papers collected in `/Users/eranil/All/Useful Papers`, with arXiv id and the
figure (if any) this repo reproduces from them.  **Only the review (arXiv
2302.09222) is reproduced from; every other paper is library / reference
context.**

| paper | arXiv | role | reproduced here |
|---|---|---|---|
| Qin & Yin, *A Review of Codebooks for CSI Feedback in 5G New Radio and Beyond* | **2302.09222** | canonical review; the reproduction target | **Fig. 5** -> `repro_qin_fig05`; **Fig. 7** -> `repro_qin_fig07`; **Table I** -> `repro_qin_table1` |
| Ning, Yin, … Heath, Björnson, *PMI Codebooks in 3GPP* | **2601.05092** | this repo's own in-repo tutorial (`paper/main.tex`) | already covered by `scripts/fig_01..12_*.py` (no new work) |
| 3GPP TS 38.214 v19.2.0 | — (normative spec) | the codebook formulas implemented in `src/nr_csi/` | reference (overhead/precoder formulas) |
| *Mobility Enhancements in 3GPP Toward Rel-20 and Beyond* | — (survey) | Rel-20 mobility context | reference only |
| *Massive MIMO Evolution Towards 3GPP Release 18* | **2210.08218** | system-level R16->R18 codebook / throughput-gain framing | reference only (Fig-5 framing) |
| *Deep Learning Empowered Type-II Codebook* | **2305.08081** | its classical Type-II baseline curves (SGCS/throughput vs overhead) | reference cross-check for `repro_qin_fig05` (ML parts ignored) |
| *Time-domain Channel Property Feedback* | **2307.14998** | R18 Doppler context | reference only (`fig_05_mobility` context) |

## Reproduction status

- `repro_qin_fig05` — review **Fig. 5**, eType-II relative gain vs overhead,
  MU-MIMO, P=32, rank 1. *Trend / Pareto-shape reproduced* (ordering and
  frontier), not bit-exact % — see the figure's `.md` for the documented
  differences (channel model, scheduler/RU, normalization, (L,Mv) grid).
- `repro_qin_fig07` — review **Fig. 7**, port-selection relative gain vs
  overhead, P=32, rank 2. Evaluated **SU** (the harness `evaluate_mu` is
  rank-1-per-user only, and the eigen PS advantage is per-drop so it is
  incoherent under cross-user MU ZF); the eigen curve uses a literal
  covariance-eigenvector PEB. Ordering and overhead spread reproduced.
- `repro_qin_table1` — review **Table I**, self-checking overhead table:
  R15/R16/R18 bits from `metrics/overhead.py`, Type I/R17 from the codebooks.

No `src/` file, existing `scripts/fig_*.py`, or test was modified; these are
additive reproduction scripts only.
