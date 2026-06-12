# fig_09_port_selection.png — regular vs port-selection on both channel domains

**What it shows.** Six variants (Type I, R15, R15-PS, R16, R16-PS, R17),
each evaluated on BOTH views of the same 2-ray drops: the raw antenna-domain
channel (blue) and the same drops through a unitary per-polarization DFT PEB
(orange) — the deployment model where the gNB beamforms the CSI-RS and the
UE sees a port-domain-sparse channel. Bars annotated with feedback bits.

**Why it looks like this.**

* **Each family wins on its home turf, and crashes off it.** Regular
  codebooks: R16 0.98 antenna / 0.81 beam; Type I 0.77 / 0.34. PS
  codebooks: R17 0.50 antenna / 0.85 beam. The PS basis vectors are
  *selection* vectors e_i (paper, Port-Selection appendix): they assume the
  channel is already sparse in the port index, which is false in the
  antenna domain; conversely Type I's DFT beams are maximally spread in
  the beam domain (a DFT of a DFT ≈ identity-like concentration is lost).
* **R17 is the best scheme in the beam domain at the lowest cost**
  (0.85 at 82 b vs R16's 0.81 at 131 b): once the PEB has concentrated the
  channel, free port selection + M = 2 taps is the right-sized model —
  the paper's "optimization of port selection in R17" claim, quantified.

**Notable / unexpected — explained.** R15-PS (0.73) and R16-PS (0.74) lose
to their *regular* counterparts (0.78/0.81) even in the beam domain. This
is structural, not a bug: the R15/R16 PS basis is a **consecutive window**
v_{i₁,₁·d+i} of L ports (paper eq. "vd"), and with 2 off-grid rays the PEB
concentrates energy on strong ports that are generally *not adjacent* —
the window covers some of them, free selection (R17, Algorithm 4) covers
all of them, and the regular codebook recovers obliquely by re-beamforming
the beam-domain ports with its own DFT combination. The R17-vs-window gap
(0.85 vs 0.73/0.74) is exactly the enhancement R17 standardized; the same
effect is asserted in `tests/test_equivalences.py::TestR17VsR16PS`.

The PEB here is the full orthogonal DFT group — unitary, so SGCS on the two
domains is directly comparable (same drops, rotated). A gNB with channel
statistics would use a *tuned* PEB (e.g. eigenbeams), concentrating energy
on even fewer, more nearly adjacent ports — the windowed PS codebooks fare
better the better the PEB is. The DFT PEB is the neutral choice.

**Config.** `scripts/fig_09_port_selection.py`, 60 drops, 2-ray channel,
rank 1, d = 1 for the PS variants, R17 pc5 (α = 1/2, M = 2).
