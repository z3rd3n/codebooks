"""Smoke test for scripts/compare/mu_pareto.py (--quick mode)."""

import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "compare" / "mu_pareto.py"


def test_quick_run_produces_csv_and_png(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    csv_path = tmp_path / "mu_pareto.csv"
    png_path = tmp_path / "mu_pareto.png"
    assert csv_path.exists() and png_path.exists()
    assert png_path.stat().st_size > 0

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 4  # Type I, Type II, 2x eType II, beamformed FeType II
    expected_cols = {"label", "bits", "sum_rate", "sum_rate_ci95",
                     "edge_rate_p5", "sum_rate_full_csi"}
    assert expected_cols <= set(rows[0].keys())
    for r in rows:
        bits, rate = float(r["bits"]), float(r["sum_rate"])
        assert bits > 0 and rate > 0
        # quantized feedback can never beat the matched full-CSI reference
        assert rate <= float(r["sum_rate_full_csi"]) + 1e-6
        assert 0 <= float(r["edge_rate_p5"]) <= rate
