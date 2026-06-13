# fig_07_rank_adaptation.png — rank (RI) adaptation, layers vs SNR (CDL-C, 4 rx)

**What it shows.** R16 eType II (pc6) on a 4-rx CDL-C channel. Left: SE vs SNR at
fixed ranks 1–4, the auto-RI envelope (`select_rank` re-decides the rank per drop
and SNR), and the rank-4 eigen upper bound. Right: the rank distribution auto-RI
actually selects vs SNR.

**Why it looks like this.**

* **Auto-RI rides the bound.** The orange envelope hugs the rank-4 eigen bound
  across the whole range (e.g. 17.8 b/s/Hz at 10 dB, bound ~18.4), by always
  picking the rank that maximizes SE for that drop and SNR — no fixed-rank curve
  does this everywhere.
* **The crossovers are the whole point.** At low SNR the fixed-rank-1 curve is
  best (beamforming gain beats multiplexing when noise dominates); each higher
  rank overtakes as SNR climbs. Auto-RI is the upper envelope of these crossing
  curves.
* **The rank histogram shifts cleanly with SNR.** At −5…0 dB auto-RI picks mostly
  **rank 2**; by 5 dB it is split rank 3/4; from 15 dB on it is **rank 4 almost
  always** — a 4-rx channel supports four streams once the SNR pays for them.
  CDL-C's spatial richness gives genuinely usable higher-order eigenmodes, so the
  transition to rank 4 is decisive.

**CDL vs synthetic.** The synthetic version uses a 6-ray channel to manufacture
rank; CDL-C provides the rank physically (4 rx into a correlated-but-rich cluster
set), so the auto-RI envelope and the rank-4 saturation are if anything cleaner.

**Config.** `scripts/cdl_fig_07_rank_adaptation.py`, CDL-C, R16 pc6, 4 rx,
N₃ = 8, 60 drops.
