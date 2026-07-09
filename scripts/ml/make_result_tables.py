"""Emit the markdown result tables for docs/ml/glimpse.md from the eval JSONs.

Reads ``results/ml/frontier_cdl*.json`` (+ optional generalization) and prints:
* a rate-distortion comparison at representative bit budgets,
* the overhead-at-matched-fidelity table,
* the ablation table,
* the cross-profile generalization table.

    .venv/bin/python scripts/ml/make_result_tables.py --results results/ml --cdl C
"""

from __future__ import annotations

import argparse
import json
import pathlib

CODEBOOK = ["R15 Type I", "R15 Type II", "R16 eType II", "R17 FeType II PS"]


def load(p: pathlib.Path) -> list[dict]:
    return json.loads(p.read_text())["points"]


def nearest(rows, family, bits):
    c = [r for r in rows if r["family"] == family]
    return min(c, key=lambda r: abs(r["bits"] - bits)) if c else None


def best_codebook_at(rows, bits, tol=1.12):
    c = [r for r in rows if r["family"] in CODEBOOK and r["bits"] <= bits * tol]
    return max(c, key=lambda r: r["sgcs"]) if c else None


def best_codebook_for_sgcs(rows, target):
    c = [r for r in rows if r["family"] in CODEBOOK and r["sgcs"] >= target]
    return min(c, key=lambda r: r["bits"]) if c else None


def best_glimpse_for_sgcs(rows, target):
    c = [r for r in rows if r["family"] == "GLIMPSE (learned)" and r["sgcs"] >= target]
    return min(c, key=lambda r: r["bits"]) if c else None


def frontier_table(rows) -> str:
    budgets = [48, 96, 144, 192]
    out = ["| ~bits | GLIMPSE SGCS | best codebook SGCS (config) | "
           "GLIMPSE SE@10 | codebook SE@10 |",
           "|---|---|---|---|---|"]
    for bb in budgets:
        g = nearest(rows, "GLIMPSE (learned)", bb)
        c = best_codebook_at(rows, bb)
        if not g or not c:
            continue
        gse, cse = g["se"]["10.0"], c["se"]["10.0"]
        win = "**" if g["sgcs"] >= c["sgcs"] else ""
        out.append(f"| {bb} | {win}{g['sgcs']:.3f}{win} (m={g['config']}) | "
                   f"{c['sgcs']:.3f} ({c['family'].split()[0]} {c['config']}, {c['bits']:.0f}b) | "
                   f"{gse:.2f} | {cse:.2f} |")
    return "\n".join(out)


def gain_table(rows) -> str:
    out = ["| target SGCS | codebook bits | GLIMPSE bits | reduction |",
           "|---|---|---|---|"]
    for t in (0.70, 0.80, 0.85, 0.90, 0.92):
        c = best_codebook_for_sgcs(rows, t)
        g = best_glimpse_for_sgcs(rows, t)
        if c and g:
            out.append(f"| {t:.2f} | {c['bits']:.0f} | {g['bits']:.0f} | "
                       f"**{100 * (1 - g['bits'] / c['bits']):.0f}%** |")
    return "\n".join(out)


def ablation_table(rows) -> str:
    fams = [("GLIMPSE (learned)", "KLT + water-fill + learned (full)"),
            ("GLIMPSE (learned, uniform-B)", "KLT + uniform-B + learned"),
            ("GLIMPSE (LS)", "KLT + water-fill + LS (linear)"),
            ("GLIMPSE (OMP)", "KLT + water-fill + OMP"),
            ("GLIMPSE-random (LS)", "random basis + LS"),
            ("GLIMPSE-random (OMP)", "random basis + OMP")]
    budgets = [48, 96, 144, 192]
    header = "| variant | " + " | ".join(f"~{b}b" for b in budgets) + " |"
    out = [header, "|" + "---|" * (len(budgets) + 1)]
    for fam, label in fams:
        cells = []
        for bb in budgets:
            r = nearest(rows, fam, bb)
            cells.append(f"{r['sgcs']:.3f}" if r and abs(r["bits"] - bb) < bb * 0.3 else "–")
        if any(c != "–" for c in cells):
            out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def generalization_table(results: pathlib.Path, models: str, bits: int) -> str | None:
    have = []
    for m in models.split(","):
        p = results / f"frontier_cdl{m}.json"
        if p.exists():
            have.append((m, load(p)))
    if len(have) < 2:
        return None
    out = [f"| profile | GLIMPSE SGCS @~{bits}b | best codebook SGCS | in training mix? |",
           "|---|---|---|---|"]
    for m, rows in have:
        g = nearest(rows, "GLIMPSE (learned)", bits)
        c = best_codebook_at(rows, bits)
        inmix = "yes" if m in ("A", "B", "C") else "no (near-LoS)"
        out.append(f"| CDL-{m} | {g['sgcs']:.3f} | {c['sgcs']:.3f} | {inmix} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results/ml"))
    ap.add_argument("--cdl", default="C")
    ap.add_argument("--gen-models", default="A,B,C,D,E")
    ap.add_argument("--gen-bits", type=int, default=144)
    args = ap.parse_args()
    rows = load(args.results / f"frontier_cdl{args.cdl}.json")
    print("### 4.1 frontier\n")
    print(frontier_table(rows))
    print("\n### 4.2 gain\n")
    print(gain_table(rows))
    print("\n### 4.3 ablation\n")
    print(ablation_table(rows))
    gen = generalization_table(args.results, args.gen_models, args.gen_bits)
    if gen:
        print("\n### 4.4 generalization\n")
        print(gen)


if __name__ == "__main__":
    main()
