"""Run every CDL figure script (cdl_fig_*.py) into results/sionna_cdl_gallery/.

Full gallery:   python scripts/make_cdl_gallery.py
Smoke test:     python scripts/make_cdl_gallery.py --fast
Pick a model:   python scripts/make_cdl_gallery.py --model A
Extra args (--drops, --seed) are forwarded to every script.

Each cdl_fig_NN.py runs the matching fig_NN logic (scripts/figlib.py) on a
Sionna 3GPP TR 38.901 CDL channel; see scripts/cdllib.py for the channel and
results/sionna_cdl_gallery/fig_NN_*.md for the per-figure analysis.
"""

import pathlib
import subprocess
import sys
import time

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent


def main() -> int:
    scripts = sorted(SCRIPTS_DIR.glob("cdl_fig_*.py"))
    forwarded = sys.argv[1:]
    failures = []
    t_total = time.time()
    for script in scripts:
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(script), *forwarded], cwd=SCRIPTS_DIR)
        dt = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
        print(f"[{status:>11}] {script.name}  {dt:6.1f}s")
        if proc.returncode != 0:
            failures.append(script.name)
    print(f"\n{len(scripts) - len(failures)}/{len(scripts)} figures in "
          f"{time.time() - t_total:.1f}s")
    if failures:
        print("failed:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
