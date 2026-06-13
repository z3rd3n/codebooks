# fig_09_port_selection.png — regular vs port-selection across domains (CDL-C)

**What it shows.** Six variants (Type I, R15, R15-PS, R16, R16-PS, R17), each
scored on THREE views of the *same* CDL-C drops: the raw antenna-domain channel
(blue), the same drops through a unitary per-polarization DFT PEB (orange — the
port-selection deployment model), and the tuned PEB (green — beam domain with
ports re-sorted per drop by energy). Bars annotated with feedback bits.

**Why it looks like this.**

* **Each family wins on its home turf and collapses off it.** Regular codebooks:
  R16 0.95 antenna → 0.76 beam; Type I 0.64 → 0.26. PS codebooks: R17 0.62
  antenna → 0.90 beam. The PS basis vectors are *selection* vectors that assume a
  port-sparse channel — true after a PEB, false in the antenna domain; Type I's
  DFT beams are maximally spread in the beam domain, so they lose there.
* **R17 is the best beam-domain scheme at the lowest cost** (0.90 at 82 bits vs
  R16's 0.76 at 156): once the PEB has concentrated the channel, free port
  selection + M = 2 taps is the right-sized model.
* **The tuned PEB rescues the windowed PS codebooks.** R15-PS/R16-PS use a
  *consecutive* port window, which the DFT PEB scatters across non-adjacent
  strong ports; re-sorting ports by energy lifts them above their regular
  counterparts (R15-PS 0.90 → 0.93, R16-PS 0.91 → 0.94) while R17 is unchanged
  (0.90) — free selection is order-invariant, the control that confirms the
  mechanism.

**CDL vs synthetic.** The cross-over pattern is identical; CDL-C is slightly less
port-sparse after the DFT PEB than the 2-ray synthetic channel, so the beam-domain
numbers are a touch lower, but the "each codebook on its designed domain" story is
unchanged. The three domains are scored on the same drops (the CDL bank is reset
between views), exactly as the figure intends.

**Config.** `scripts/cdl_fig_09_port_selection.py`, CDL-C, (4,2) array, N₃ = 8,
rank 1, d = 1 for the PS variants, R17 pc5, 60 drops.
