# fig_07_rank_adaptation.png — rank adaptation (RI) for R16 eType II

**What it shows.** R16 pc6 on a 4-rx, 6-ray channel. Left: SE vs SNR at
fixed ranks 1–4, the auto-RI envelope (`eval.select_rank` re-decides the
rank per drop and per SNR), and the rank-4 eigen bound. Right: which rank
auto-RI actually picks, as stacked fractions of drops.

**Why it looks like this.**

* **The fixed-rank curves cross.** Rank 1 wins below ~−5 dB (all power on
  the strongest eigendirection), rank 2 around 0 dB, rank 3 near 5–10 dB,
  rank 4 only from ~15 dB. Splitting power across v layers costs ~10log₁₀v
  of per-layer SNR but multiplies the slope by v — the classic
  beamforming-vs-multiplexing trade, here with quantized precoders.
  At −5 dB fixed rank 4 (3.28) is *worse* than rank 2 (4.19): half the
  power goes to eigendirections the SNR cannot exploit.
* **Auto-RI rides the upper envelope and slightly beats it at the
  crossovers** (15 dB: 21.03 vs best fixed 20.94), because it adapts per
  drop: channels with a strong third eigenvalue get rank 3, others rank 2 —
  a selection gain no fixed rank can match. Where one rank dominates
  (25–30 dB: rank 4 chosen in 100% of drops) auto-RI coincides with the
  fixed curve exactly, as it must.
* The right panel shows the decision migrating 2 → 3 → 4 over ~25 dB of
  SNR; rank 1 is already rare at −5 dB because the 6-ray channel almost
  always offers a usable second eigendirection.
* The gap to the rank-4 eigen bound (~2.2 b/s/Hz at 30 dB) is the
  quantization cost of feeding back *four* directions with the same L = 4 /
  K₀ budget — per-layer fidelity drops as rank grows (p_v halves for
  v ∈ {3,4} per the param-combination table).

**Notable.** This figure is also the harness demo for `select_rank`: it
honors codebook rank limits and rank-restriction bitmaps, so an ML scheme
implementing `select(H, rank)` inherits auto-RI evaluation for free.

**Config.** `scripts/fig_07_rank_adaptation.py`, 60 drops, n_rx = 4,
n_paths = 6. Type I/R15 stop at rank 2 by spec and are shown in fig_01.
