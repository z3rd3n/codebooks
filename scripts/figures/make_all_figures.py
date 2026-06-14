"""Run every comparison figure script (fig_*.py) and report timing/status.

Full gallery:   python scripts/make_all_figures.py
Smoke test:     python scripts/make_all_figures.py --fast
Extra args (e.g. --drops 200 --seed 1 --out results) are forwarded to
every script.

Outputs land in results/ as fig_NN_<name>.png + .json (see
plans/plan_figures.md for what each figure compares).
"""

import pathlib
import subprocess
import sys
import time

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent


def main() -> int:
    scripts = sorted(SCRIPTS_DIR.glob("fig_*.py"))
    forwarded = sys.argv[1:]
    failures = []
    t_total = time.time()
    for script in scripts:
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(script), *forwarded],
                              cwd=SCRIPTS_DIR)
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
