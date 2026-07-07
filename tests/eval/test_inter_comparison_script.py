"""Smoke test for scripts/compare/inter_comparison.py (--quick mode)."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "compare" / "inter_comparison.py"


def test_quick_run_produces_all_figures_and_summary(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr

    for name in ("fig1_mu_pareto.png", "fig2_delay.png",
                 "fig3_rate_distortion.png", "fig4_receiver.png"):
        f = tmp_path / name
        assert f.exists() and f.stat().st_size > 0, name

    summary = json.loads((tmp_path / "summary.json").read_text())
    # fig1: every ladder point sits at or below the full-CSI reference
    fig1 = summary["fig1_mu_pareto"]
    for pts in fig1["families"].values():
        for p in pts:
            assert 0 < p["sum_rate"] <= fig1["full_csi"] + 1e-6
            assert p["bits"] > 0
    # fig3: metrics are within their physical ranges
    for pts in summary["fig3_rate_distortion"].values():
        for p in pts:
            assert 0 <= p["sgcs"] <= 1
            assert 0 <= p["se_frac"] <= 1 + 1e-9
    # fig4: per-layer MMSE never exceeds joint decoding
    for row in summary["fig4_receiver"].values():
        assert row["mmse"] <= row["joint"] + 1e-9
        assert max(row["rank1"], row["rank2"]) <= row["auto"] + 1e-9
