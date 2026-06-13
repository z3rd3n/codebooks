# Repro: review Table I -- codebook comparison (self-checking overhead)

Source: Qin & Yin, *A Review of Codebooks for CSI Feedback in 5G New Radio and Beyond*, **arXiv 2302.09222**, Table I.

Operating point: **P = 32** ((N1,N2)=(8,2) dual-pol), N3 = 13, rank 1, R18 Doppler window N4 = 4.  The overhead column is computed (source in the last column), never hand-typed.

| family | spatial/delay beams | subband quantization | overhead [bits] | complexity (UE search) | bits source |
|---|---|---|---:|---|---|
| R15 Type I | 1 DFT beam + co-phasing | none (wideband PMI; per-subband co-phase only) | 34 | O(O1 O2 N1 N2) DFT-beam search | `codebook` |
| R15 Type II | L=4 beams/pol (2L=8 coeffs) | per-subband amplitude + n_psk phase (subbandAmplitude) | 351 | O(O1 O2 * C(N1 N2, L)) group+beam search, per-subband LS | `overhead.r15_bits` |
| R16 eType II | L=4 beams x Mv=7 delay taps (FD compression) | delay-domain coeffs, bitmap of K0 nonzeros | 221 | R15-II search + O(N3 log N3) delay FFT | `overhead.r16_bits` |
| R17 FeType II PS | free L=8=alpha*P/2 ports x M=2 taps | delay-domain coeffs over gNB-beamformed ports | 162 | O(C(P/2, L)) free port selection (no DFT-beam search) | `codebook` |
| R18 eType II Doppler | L=4 x Mv=7 x Q Doppler (one report covers N4=4) | delay + Doppler (time) DD-domain basis | 280 | R16 search + O(N4 log N4) Doppler DFT | `overhead.r18_bits` |

## Self-check (formula vs implementation)

- R15 Type II (L=4, n_psk=8, rank 1): overhead.py `r15_bits` = **351** == codebook `total_overhead_bits` = **351** (R15 reports all 2L coeffs every subband, so formula and realized counts coincide).
- R16/R18 formula bits use the spec worst-case K_nz; the codebook reports only the realized K_nz nonzeros, so its figure is lower (it is the per-drop cost, not the formula upper bound).

## Notes

- `overhead.py` (`r15_bits`/`r16_bits`/`r18_bits`) transcribes the spec per-information-element formulas; this table evaluates them at P = 32 so the bit counts match the paper's Table I overhead column for a 32-port array.
- Type I and R17 have no formula in `overhead.py`; their bits are the realized `total_overhead_bits` of a representative selected PMI.
- The complexity column is a short annotation of each codebook's documented reconstruction (module docstrings in `src/nr_csi/codebooks/`), not a verbatim quote of the paper's column.
