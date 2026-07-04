"""Exhaustive(-ish) sweep of the NR PMI codebook catalog via the webapp's
own ``run_playground`` call, one JSON + one PNG per (codebook, antenna,
params, rank, channel, seed) combination.

Reuses ``webapp/server/runner.py::run_playground`` directly (no HTTP, no
reimplemented codebook instantiation), so every result is exactly what the
webapp's Playground would produce for that request.

Spec-valid parameter grid: for each (codebook, antenna pair, N3), the raw
boolean/choice parameter enumeration is pre-filtered with a bare
``runner.instantiate()`` dry run (no channel, no evaluation -- a few
microseconds each) *before* entering the expensive rank x channel loop, so
the attempted set is (almost) exactly the spec-valid grid rather than a
broad cartesian product padded with predictable skips. Only entries whose
still-valid combo count is very large (currently just ``cjt-r18``) are
further capped, documented in SUMMARY.md.

Every successful run additionally re-derives the exact-precision precoder
W (same seed, same instantiate/channel/select/precoder call sequence
``run_playground`` itself uses internally -- not a reimplementation) and
checks the fundamental per-subband unit-power invariant every codebook's
own formula is normalized to satisfy: each layer's column has norm
1/sqrt(rank) (equivalently tr(W_t^H W_t) = 1 summed over all layers). This
is exactly the invariant a prior L>P/2 port-aliasing bug in the R16 eType
II port-selection codebook violated (tr(W^H W) = 2.0 instead of 1.0); this
check catches this whole bug class automatically going forward.

Scope tiers:

* Tier 1 (default) -- up to 3 representative antenna pairs per codebook
  (smallest / middle / largest), every rank in the codebook's supported
  range, the spec-valid parameter grid (capped only where huge, see
  ``--param-combo-cap``), all 4 synthetic channel presets, N3 fixed at 8,
  one seed. CDL is skipped (slow, needs TensorFlow/Sionna).
* Tier 2 (opt-in, ``--tier 2``) -- all antenna pairs, N3 in {4, 8, 16},
  multiple seeds (``--seeds``), plus one CDL channel config for codebooks
  that support it. Much slower; not run unless asked for.

Usage::

    .venv/bin/python scripts/dataset/exhaustive_sweep.py --tier 1
    .venv/bin/python scripts/dataset/exhaustive_sweep.py --tier 2 --seeds 0,1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF if CDL ever loads it

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webapp.server import catalog, runner  # noqa: E402

# --------------------------------------------------------------------- config

SNR_DB = [-10, -5, 0, 10, 20, 30]
DROPS = 6

SYNTHETIC_CHANNELS = [
    ("sparse-urban", {"type": "synthetic", "preset": "sparse-urban", "n_rx": 2}),
    ("rich-scattering", {"type": "synthetic", "preset": "rich-scattering", "n_rx": 2}),
    ("near-los", {"type": "synthetic", "preset": "near-los", "n_rx": 2}),
    ("mobile-user", {"type": "synthetic", "preset": "mobile-user", "n_rx": 2}),
]
CDL_CHANNEL = (
    "cdl-C",
    {"type": "cdl", "cdl_model": "C", "cdl_speed_kmh": 3, "cdl_delay_spread_ns": 100.0, "n_rx": 2},
)

PARAM_ABBR = {
    "codebook_mode": "cm",
    "L": "L",
    "subband_amplitude": "sa",
    "port_selection": "ps",
    "d": "d",
    "param_combination": "pc",
    "R": "R",
    "N_window": "Nw",
    "N4": "N4",
    "n_trp": "ntrp",
    "param_combination_L": "pcL",
    "param_combination_alpha": "pcA",
    "mode": "md",
    "O3": "O3",
    "restricted_cmr": "rcmr",
    "variant": "var",
}

MANIFEST_FIELDS = [
    "index", "codebook_id", "name", "release", "n1", "n2", "ng", "ports", "rank",
    "params_json", "channel", "n3", "seed", "ok", "skip_reason",
    "sgcs", "subspace_sgcs", "total_bits",
    "se_at_10db", "bound_at_10db", "capacity_at_10db",
    "invariant_ok", "invariant_max_dev",
    "seconds", "json_path", "png_path",
]


# --------------------------------------------------------------- enumeration


def enumerate_params(param_specs: list) -> list[dict]:
    combos = [{p.key: p.default for p in param_specs}]
    for p in param_specs:
        values = [c.value for c in p.choices] if p.type == "choice" else [True, False]
        next_combos = []
        for combo in combos:
            if p.visible_if and str(combo.get(p.visible_if["key"])) != str(p.visible_if["value"]):
                next_combos.append(combo)  # hidden here; keep default, don't fan out
                continue
            for v in values:
                next_combos.append({**combo, p.key: v})
        combos = next_combos
    seen, unique = set(), []
    for c in combos:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def cap_combos(combos: list[dict], cap: int, param_specs: list, seed: int = 0) -> list[dict]:
    """Deterministically subsample to ``cap`` entries, always keeping the
    all-defaults combination. A no-op if ``combos`` already fits."""
    if cap <= 0 or len(combos) <= cap:
        return combos
    default = {p.key: p.default for p in param_specs}
    default_key = tuple(sorted(default.items()))
    pool = [c for c in combos if tuple(sorted(c.items())) != default_key]
    rng = random.Random(seed)
    sample = rng.sample(pool, min(cap - 1, len(pool)))
    return [default] + sample


def representative_pairs(pairs: list, k: int = 3) -> list:
    """``k`` evenly-spread antenna pairs spanning smallest to largest (both
    endpoints always included for k>=2), deduped. k=3 is smallest/middle/largest."""
    if len(pairs) <= k or k <= 1:
        return list(pairs)
    idxs = [round(i * (len(pairs) - 1) / (k - 1)) for i in range(k)]
    seen = set()
    out = []
    for i in idxs:
        p = pairs[i]
        key = (p.n1, p.n2, p.ports, p.ng)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


# ---------------------------------------------------------- spec-valid grid


def filter_valid_params(
    entry: catalog.CatalogEntry, pair, n3: int, params_list: list[dict]
) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Bare ``instantiate()`` dry run (no channel, no eval) per param combo --
    a few microseconds each -- splitting the raw enumeration into the
    spec-valid subset and the (params, reason) pairs pruned before ever
    entering the timed rank x channel loop."""
    valid, pruned = [], []
    pre_req_base = {
        "codebook_id": entry.id,
        "antenna": {"n1": pair.n1, "n2": pair.n2, "ng": pair.ng},
        "n3": n3,
    }
    for params in params_list:
        try:
            runner.instantiate(entry, {**pre_req_base, "params": params})
            valid.append(params)
        except (runner.ValidationError, ValueError) as exc:
            pruned.append((params, str(exc)))
    return valid, pruned


# -------------------------------------------------------------------- naming


def fmt_value(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    return re.sub(r"[^A-Za-z0-9]", "", str(v))


def slug_params(params: dict, param_specs: list) -> str:
    parts = []
    for p in param_specs:
        if p.key not in params:
            continue
        abbr = PARAM_ABBR.get(p.key, p.key[:4])
        parts.append(f"{abbr}{fmt_value(params[p.key])}")
    return "_".join(parts) if parts else "default"


def fmt_antenna(n1: int, n2: int, ng: int) -> str:
    s = f"ant{n1}x{n2}"
    if ng != 1:
        s += f"_ng{ng}"
    return s


_DIGIT_RE = re.compile(r"\d+")


def normalize_skip_reason(msg: str) -> str:
    return _DIGIT_RE.sub("#", msg)[:120]


# --------------------------------------------------------------- invariant


def independent_precoder(entry: catalog.CatalogEntry, req: dict) -> np.ndarray:
    """Re-derive the exact-precision W using the same call sequence
    ``run_playground`` uses internally for its own PMI/viz (same seed,
    same instantiate/resolve_channel_cfg/_channel_for/select/precoder) --
    not a reimplementation, just replicated to get full precision instead
    of the JSON payload's decimated, 4-decimal-rounded viz copy."""
    scheme, antenna, n3 = runner.instantiate(entry, req)
    chan_cfg = runner.resolve_channel_cfg(req.get("channel"), req["codebook_id"])
    channel = runner._channel_for(entry, scheme, antenna, n3, chan_cfg)
    n_slots = runner._n_slots(scheme)
    rng = np.random.default_rng(int(req.get("seed", 0)))
    H = channel.generate(n_slots=n_slots, rng=rng)
    pmi = scheme.select(H, rank=int(req["rank"]))
    return scheme.precoder(pmi)


def check_power_invariant(W: np.ndarray, rank: int, atol: float = 1e-6) -> tuple[bool, float]:
    """Every codebook's own formula normalizes each layer's column to norm
    1/sqrt(rank) (equivalently tr(W_t^H W_t) = 1 summed over all `rank`
    layers) per subband/interval -- this is exactly what the L>P/2
    port-aliasing bug broke (tr = 2.0 instead of 1.0)."""
    if not np.all(np.isfinite(W)):
        return False, float("inf")
    col_norms = np.linalg.norm(W, axis=-2)  # (S, N3, rank)
    max_dev = float(np.max(np.abs(col_norms - 1.0 / np.sqrt(rank))))
    return max_dev < atol, max_dev


# -------------------------------------------------------------------- output


def make_plot(result: dict, path: Path, title: str) -> None:
    m = result["metrics"]
    fig, ax = plt.subplots(figsize=(4, 3), dpi=110)
    ax.plot(m["snr_db"], m["se"], marker="o", ms=3, lw=1.3, label="SE (achieved)")
    ax.plot(m["snr_db"], m["se_upper_bound"], "--", marker="x", ms=3, lw=1.1,
            label="Eigen bound (equal power)")
    if "capacity_upper_bound" in m:
        ax.plot(m["snr_db"], m["capacity_upper_bound"], ":", marker="^", ms=3, lw=1.1,
                label="Capacity bound (waterfill)")
    ax.set_xlabel("SNR (dB)", fontsize=8)
    ax.set_ylabel("SE (bits/s/Hz)", fontsize=8)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_channel_list(tier: int, codebook_id: str) -> list[tuple[str, dict]]:
    chans = list(SYNTHETIC_CHANNELS)
    if tier == 2 and codebook_id != "cjt-r18":  # CDL unsupported for CJT (runner.py)
        chans.append(CDL_CHANNEL)
    return chans


def build_n3_list(tier: int) -> list[int]:
    return [8] if tier == 1 else [4, 8, 16]


def build_seed_list(tier: int, seeds_arg: str | None) -> list[int]:
    if seeds_arg:
        return [int(s) for s in seeds_arg.split(",")]
    return [0] if tier == 1 else [0, 1]


# cjt-r18 (huge raw param space) and type1-mp (a codebook_mode=2 candidate
# search that has shown wide, hard-to-reproduce wall-clock variance on
# larger multi-panel arrays) are ordered last for tier 2, so a time-budgeted
# run (--max-seconds) still gets full coverage of the other 10 families
# before touching the two least predictable ones.
_TIER2_SLOW_LAST = ("type1-mp", "cjt-r18")


def ordered_catalog(tier: int) -> list:
    if tier == 1:
        return list(catalog.CATALOG)
    fast = [e for e in catalog.CATALOG if e.id not in _TIER2_SLOW_LAST]
    slow = sorted(
        (e for e in catalog.CATALOG if e.id in _TIER2_SLOW_LAST),
        key=lambda e: _TIER2_SLOW_LAST.index(e.id),
    )
    return fast + slow


# ---------------------------------------------------------------------- main


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", type=int, choices=(1, 2), default=1)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "exhaustive_sweep"))
    p.add_argument("--param-combo-cap", type=int, default=60,
                   help="max SPEC-VALID param combinations per (codebook, antenna pair, N3) "
                        "(0 disables); only cjt-r18 has enough valid combos to hit this")
    p.add_argument("--pairs-per-entry", type=int, default=None,
                   help="antenna pairs sampled per codebook (smallest/.../largest, deduped); "
                        "default is 3 for tier 1 and ALL pairs for tier 2 -- the literal "
                        "'all pairs' tier-2 grid is ~287k attempts (~10h); pass e.g. 6 for a "
                        "still-much-broader-than-tier-1 but session-practical run (~1.5-2h "
                        "with the matching --param-combo-cap)")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seed list (default: '0' for tier 1, '0,1' for tier 2)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many attempted combinations (smoke testing)")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="stop gracefully (write manifest/SUMMARY with partial coverage) after "
                        "this much wall-clock time; for tier 2, the two least predictable "
                        "families (type1-mp, cjt-r18) are processed last so a budget cutoff "
                        "still leaves full coverage of the other 10")
    return p.parse_args()


def main() -> None:
    args = cli()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n3_list = build_n3_list(args.tier)
    seed_list = build_seed_list(args.tier, args.seeds)

    manifest_rows: list[dict] = []
    index = 0
    total_attempted = 0
    total_ok = 0
    total_pruned = 0
    total_invariant_violations = 0
    skip_reasons: Counter = Counter()
    pruned_reasons: Counter = Counter()
    per_codebook = defaultdict(lambda: {"attempted": 0, "ok": 0, "skipped": 0, "pruned": 0,
                                          "invariant_violations": 0})
    per_codebook_skip_reasons: dict[str, Counter] = defaultdict(Counter)
    raw_combo_counts: dict[str, int] = {}
    valid_combo_counts: dict[str, int] = {}
    capped_combo_counts: dict[str, int] = {}
    invariant_violation_examples: list[str] = []

    start_all = time.time()
    stop = False

    for entry in ordered_catalog(args.tier):
        if stop:
            break
        n_pairs = args.pairs_per_entry if args.pairs_per_entry is not None else (None if args.tier == 2 else 3)
        pairs = entry.antenna.pairs if n_pairs is None else representative_pairs(entry.antenna.pairs, n_pairs)
        channel_list = build_channel_list(args.tier, entry.id)
        all_params = enumerate_params(entry.params)
        raw_combo_counts[entry.id] = len(all_params)
        ranks = range(entry.ranks[0], entry.ranks[1] + 1)
        entry_valid_total = 0
        entry_capped_total = 0

        for pair in pairs:
            if stop:
                break
            for n3 in n3_list:
                if stop:
                    break
                valid_params, pruned = filter_valid_params(entry, pair, n3, all_params)
                total_pruned += len(pruned)
                per_codebook[entry.id]["pruned"] += len(pruned)
                for _params, reason in pruned:
                    normalized = normalize_skip_reason(reason)
                    pruned_reasons[normalized] += 1
                entry_valid_total += len(valid_params)
                valid_params = cap_combos(valid_params, args.param_combo_cap, entry.params, seed=0)
                entry_capped_total += len(valid_params)

                for params in valid_params:
                    if stop:
                        break
                    for rank in ranks:
                        if stop:
                            break
                        for channel_name, chan_cfg in channel_list:
                            if stop:
                                break
                            # CDL is expensive (Sionna/TF); test it once per (entry, pair,
                            # params, rank) rather than crossed with every N3 x seed, or
                            # tier 2's N3-x-seed multiplier (3x x 2x) would make CDL alone
                            # ~6x more expensive than the synthetic channels for no added
                            # signal (CDL's own model/speed/delay-spread aren't swept here).
                            if chan_cfg.get("type") == "cdl" and n3 != 8:
                                continue
                            for seed in ([seed_list[0]] if chan_cfg.get("type") == "cdl" else seed_list):
                                if args.limit is not None and total_attempted >= args.limit:
                                    stop = True
                                    break
                                if args.max_seconds is not None and time.time() - start_all > args.max_seconds:
                                    stop = True
                                    break
                                index += 1
                                total_attempted += 1
                                per_codebook[entry.id]["attempted"] += 1

                                # eigen_precoder's reference caps rank at n_rx (a real
                                # MIMO constraint: layers can't exceed receive antennas),
                                # so a fixed n_rx=2 would spuriously fail every rank>2
                                # request. Scale n_rx with rank so every rank actually runs.
                                req_chan_cfg = {**chan_cfg, "n_rx": max(chan_cfg.get("n_rx", 2), rank)}
                                req = {
                                    "codebook_id": entry.id,
                                    "params": params,
                                    "antenna": {"n1": pair.n1, "n2": pair.n2, "ng": pair.ng},
                                    "n3": n3,
                                    "rank": rank,
                                    "channel": req_chan_cfg,
                                    "snr_db": SNR_DB,
                                    "drops": DROPS,
                                    "seed": seed,
                                }
                                row = {
                                    "index": index,
                                    "codebook_id": entry.id,
                                    "name": entry.name,
                                    "release": entry.release,
                                    "n1": pair.n1,
                                    "n2": pair.n2,
                                    "ng": pair.ng,
                                    "ports": pair.ports,
                                    "rank": rank,
                                    "params_json": json.dumps(params, sort_keys=True),
                                    "channel": channel_name,
                                    "n3": n3,
                                    "seed": seed,
                                    "ok": False,
                                    "skip_reason": "",
                                    "sgcs": None,
                                    "subspace_sgcs": None,
                                    "total_bits": None,
                                    "se_at_10db": None,
                                    "bound_at_10db": None,
                                    "capacity_at_10db": None,
                                    "invariant_ok": None,
                                    "invariant_max_dev": None,
                                    "seconds": None,
                                    "json_path": "",
                                    "png_path": "",
                                }

                                try:
                                    result = runner.run_playground(req)
                                except (runner.ValidationError, ValueError) as exc:
                                    row["skip_reason"] = str(exc)[:300]
                                    normalized = normalize_skip_reason(str(exc))
                                    skip_reasons[normalized] += 1
                                    per_codebook_skip_reasons[entry.id][normalized] += 1
                                    per_codebook[entry.id]["skipped"] += 1
                                    manifest_rows.append(row)
                                    continue

                                inv_ok, inv_dev = None, None
                                try:
                                    W = independent_precoder(entry, req)
                                    inv_ok, inv_dev = check_power_invariant(W, rank)
                                except Exception as exc:  # noqa: BLE001 -- never let a
                                    # verification-only side channel crash the sweep
                                    inv_ok, inv_dev = False, float("inf")
                                    print(f"  [invariant check errored] {entry.id}: {exc}", flush=True)
                                if not inv_ok:
                                    total_invariant_violations += 1
                                    per_codebook[entry.id]["invariant_violations"] += 1
                                    if len(invariant_violation_examples) < 20:
                                        invariant_violation_examples.append(
                                            f"{entry.id} ant{pair.n1}x{pair.n2} rank{rank} "
                                            f"params={params} max_dev={inv_dev:.4g}"
                                        )

                                stem = (
                                    f"{index:06d}__{entry.id}__{fmt_antenna(pair.n1, pair.n2, pair.ng)}"
                                    f"__{slug_params(params, entry.params)}__rank{rank}"
                                    f"__ch-{channel_name}__seed{seed}"
                                )
                                json_path = out_dir / f"{stem}.json"
                                png_path = out_dir / f"{stem}.png"
                                json_path.write_text(json.dumps(result))
                                title = (f"{entry.short_name} {fmt_antenna(pair.n1, pair.n2, pair.ng)} "
                                         f"r{rank}\n{channel_name}")
                                make_plot(result, png_path, title)

                                m = result["metrics"]
                                row.update({
                                    "ok": True,
                                    "sgcs": m["sgcs"],
                                    "subspace_sgcs": m["subspace_sgcs"],
                                    "total_bits": m["total_bits"],
                                    "se_at_10db": m["se_at_10db"],
                                    "bound_at_10db": m["bound_at_10db"],
                                    "capacity_at_10db": m.get("capacity_at_10db"),
                                    "invariant_ok": inv_ok,
                                    "invariant_max_dev": round(inv_dev, 8) if inv_dev is not None else None,
                                    "seconds": result["seconds"],
                                    "json_path": json_path.name,
                                    "png_path": png_path.name,
                                })
                                manifest_rows.append(row)
                                total_ok += 1
                                per_codebook[entry.id]["ok"] += 1

                                if index % 50 == 0:
                                    elapsed = time.time() - start_all
                                    print(f"[{index}] attempted={total_attempted} ok={total_ok} "
                                          f"invariant_violations={total_invariant_violations} "
                                          f"elapsed={elapsed:.1f}s ({entry.id})", flush=True)

        valid_combo_counts[entry.id] = entry_valid_total
        capped_combo_counts[entry.id] = entry_capped_total

    total_seconds = time.time() - start_all

    # ---------------------------------------------------------------- manifest

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, restval="")
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in MANIFEST_FIELDS})

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest_rows, f)

    # ------------------------------------------------------------------ summary

    lines = []
    lines.append("# Exhaustive sweep summary\n")
    lines.append(f"- Tier: **{args.tier}**")
    lines.append(f"- Total attempted (spec-valid grid, entered the timed rank x channel x seed loop): "
                 f"**{total_attempted}**")
    lines.append(f"- Succeeded: **{total_ok}**")
    lines.append(f"- Skipped at runtime (passed the dry-run filter but still failed -- worth "
                 f"investigating): **{total_attempted - total_ok}**")
    lines.append(f"- Pruned before the timed loop (failed the spec-valid dry-run filter, near-zero "
                 f"cost, NOT counted in 'attempted' above): **{total_pruned}**")
    lines.append(f"- **Power invariant violations (tr(W_t^H W_t) != 1): {total_invariant_violations}**"
                 + (" — investigate immediately." if total_invariant_violations else " (none)."))
    lines.append(f"- Wall-clock time: **{total_seconds:.1f}s** ({total_seconds/60:.1f} min)")
    lines.append("")
    lines.append("## Scope / tier choices")
    lines.append("")
    _default_n_pairs = None if args.tier == 2 else 3
    _n_pairs = args.pairs_per_entry if args.pairs_per_entry is not None else _default_n_pairs
    if _n_pairs is None:
        lines.append("- Antenna: **all** pairs in `entry.antenna.pairs` per codebook.")
    else:
        lines.append(f"- Antenna: up to **{_n_pairs}** representative pairs per codebook (smallest / "
                      f"...(evenly spread).../ largest of `entry.antenna.pairs`, deduped)"
                      + (" -- the literal 'all pairs' tier-2 grid is ~287k attempts (~10h); this "
                         "run traded pair coverage for N3/seed/CDL coverage to stay session-practical."
                         if args.tier == 2 else "."))
    lines.append("- Rank: every rank in `entry.ranks` (inclusive).")
    lines.append(
        "- Params: the **spec-valid grid** -- `enumerate_params()`'s full boolean/choice enumeration, "
        "pre-filtered per (codebook, antenna pair, N3) with a bare `runner.instantiate()` dry run "
        f"(no channel, no eval) so only combinations that actually construct are attempted, capped at "
        f"**{args.param_combo_cap}** per (pair, N3) only when the valid set itself is still large "
        "(in practice only `cjt-r18`; every other entry's valid set already fits under the cap)."
    )
    lines.append(f"- Channels: {', '.join(name for name, _ in SYNTHETIC_CHANNELS)}"
                 + (" + one CDL config (skipped for cjt-r18, unsupported; run once per "
                    "(entry, pair, params, rank) at N3=8 and the first seed only -- not "
                    "crossed with the full N3 x seed grid, since Sionna/TF makes it "
                    "~10x slower per call than the synthetic channels for no added signal "
                    "(its own model/speed/delay-spread aren't swept here))." if args.tier == 2
                    else " (CDL skipped — slow, opt-in via --tier 2)."))
    lines.append(f"- N3: {n3_list}.")
    lines.append(f"- Seeds: {seed_list}.")
    lines.append(f"- `snr_db={SNR_DB}`, `drops={DROPS}`.")
    if args.tier == 2:
        lines.append(
            f"- Processing order: `{'`, `'.join(e.id for e in ordered_catalog(2))}` -- "
            f"`type1-mp` and `cjt-r18` (the two families with the largest or least "
            f"predictable per-run cost) are deliberately processed **last**, so a "
            f"`--max-seconds` budget cutoff still leaves full coverage of the other 10 "
            f"families rather than an arbitrary catalog-order prefix."
        )
    if args.max_seconds is not None:
        hit_budget = total_seconds >= args.max_seconds - 1.0
        lines.append(
            f"- Wall-clock budget: **{args.max_seconds:.0f}s** "
            + (f"-- **hit**; this run stopped early with partial coverage of the last "
               f"family or two (see per-codebook counts below)." if hit_budget
               else "-- not hit; the full requested grid completed within budget.")
        )
    lines.append("")
    lines.append("### Spec-valid grid vs. raw enumeration, per codebook")
    lines.append("")
    lines.append(
        "`raw` = every boolean/choice combination `enumerate_params()` can produce, ignoring antenna "
        "size entirely. `valid` = the subset that actually constructs for at least one (pair, N3) in "
        "this sweep (summed across all pairs/N3 -- so it can slightly exceed `raw` when a combo is "
        "valid for multiple pairs). `capped` = what was actually attempted after the per-(pair,N3) cap. "
        "`cjt-r18`'s raw count (3024) is inflated because its `param_combination_L`/`_alpha` choice "
        "lists are fixed for `n_trp=2` at catalog-definition time and don't narrow for other `n_trp` "
        "values; even after filtering to genuinely valid combos it stays the largest family by a wide "
        "margin, hence the cap."
    )
    lines.append("")
    lines.append("| codebook | raw | valid (all pairs/N3) | attempted after cap |")
    lines.append("|---|---|---|---|")
    for entry in catalog.CATALOG:
        raw = raw_combo_counts.get(entry.id, "-")
        valid = valid_combo_counts.get(entry.id, "-")
        capped = capped_combo_counts.get(entry.id, "-")
        lines.append(f"| {entry.id} | {raw} | {valid} | {capped} |")
    lines.append("")

    lines.append("## Per-codebook counts")
    lines.append("")
    lines.append("| codebook | name | release | attempted | ok | runtime-skipped | pruned (pre-filter) | invariant violations |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for entry in catalog.CATALOG:
        c = per_codebook[entry.id]
        lines.append(f"| {entry.id} | {entry.name} | {entry.release} | {c['attempted']} | {c['ok']} | "
                     f"{c['skipped']} | {c['pruned']} | {c['invariant_violations']} |")
    lines.append("")

    lines.append("## Runtime skip reasons (passed the dry-run filter, still failed)")
    lines.append("")
    if skip_reasons:
        lines.append("| count | reason (digits normalized) |")
        lines.append("|---|---|")
        for reason, count in skip_reasons.most_common():
            lines.append(f"| {count} | {reason} |")
    else:
        lines.append("(none -- every combination that passed the spec-valid pre-filter also ran "
                      "to completion)")
    lines.append("")

    lines.append("## Pre-filter prune reasons (near-zero cost, never entered the timed loop)")
    lines.append("")
    if pruned_reasons:
        lines.append("| count | reason (digits normalized) |")
        lines.append("|---|---|")
        for reason, count in pruned_reasons.most_common(25):
            lines.append(f"| {count} | {reason} |")
    else:
        lines.append("(none)")
    lines.append("")

    if total_invariant_violations:
        lines.append("## Power invariant violations (examples)")
        lines.append("")
        lines.append(
            "Every codebook's own reconstruction formula normalizes each layer to norm 1/sqrt(rank) "
            "per subband (tr(W_t^H W_t) = 1 summed over all layers). A violation here means the "
            "precoder is transmitting the wrong total power -- exactly the class of bug found in the "
            "R16 eType II port-selection codebook (L > P/2 aliased distinct beam indices onto the "
            "same physical port). Investigate immediately; do not attribute to metric/bound choice."
        )
        lines.append("")
        for line in invariant_violation_examples:
            lines.append(f"- {line}")
        lines.append("")

    zero_success = [
        e for e in catalog.CATALOG
        if per_codebook[e.id]["attempted"] > 0 and per_codebook[e.id]["ok"] == 0
    ]
    if zero_success:
        lines.append("## Zero-success families (needs attention)")
        lines.append("")
        lines.append(
            "These codebooks were attempted but never succeeded once. Before assuming a sweep "
            "bug, check the dominant skip reason below against the codebook's own antenna/rank "
            "constraints."
        )
        lines.append("")
        for e in zero_success:
            top = per_codebook_skip_reasons[e.id].most_common(3)
            reasons = "; ".join(f"{r} (x{c})" for r, c in top)
            lines.append(f"- **{e.id}** ({per_codebook[e.id]['attempted']} attempted): {reasons}")
        lines.append("")

    lines.append("## Metrics glossary")
    lines.append("")
    lines.append(
        "- `bound_at_10db` -- equal-power SVD (eigen) beamforming on the same rank subspace: a valid "
        "*achievable* rate, not a true supremum. A codebook precoder with non-orthogonal or "
        "unequal-power layers can occasionally beat it without any bug.\n"
        "- `capacity_at_10db` -- true waterfilling capacity over the top-`rank` singular values: the "
        "provable supremum over *any* linear precoder with tr(W^H W)=1 and rank(W)<=rank. `se_at_10db` "
        "should never exceed this one; if it does, that's a genuine bug (unlike the equal-power bound, "
        "which is fine to occasionally beat).\n"
        "- `invariant_ok` / `invariant_max_dev` -- per-run check that every layer's column has norm "
        "1/sqrt(rank) in every subband, independently re-derived (not read from the rounded/decimated "
        "JSON viz payload)."
    )
    lines.append("")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    print(f"\nDone: {total_ok}/{total_attempted} succeeded in {total_seconds:.1f}s "
          f"({total_pruned} pruned pre-filter, {total_invariant_violations} invariant violations).")
    print(f"Manifest: {out_dir / 'manifest.csv'}, {out_dir / 'manifest.json'}")
    print(f"Summary: {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
