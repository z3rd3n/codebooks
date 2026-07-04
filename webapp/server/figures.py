"""Figure-script subprocess jobs — a faithful port of ``webapp/registry.py``.

Runs the existing ``scripts/figures/fig_*.py`` (or its Sionna ``cdl_*`` twin)
as a subprocess so the web process stays TensorFlow-free, then reads back the
PNG/JSON the script writes.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "scripts" / "figures"
SYNTH_OUT = REPO_ROOT / "results" / "webapp"
# CDL scripts force their output here (cdllib.run_original -> GALLERY); it
# cannot be redirected, so CDL results are read back from this fixed location.
CDL_OUT = REPO_ROOT / "results" / "sionna_cdl_gallery"

# Display name -> NRCSI_FAMILIES token (must equal the codebook .name prefix
# figlib.select_families / style() match on). Identical to registry.py.
FAMILIES: dict[str, str] = {
    "Type I (R15)": "R15 Type I",
    "Type II (R15)": "R15 Type II",
    "eType II (R16)": "R16 eType II",
    "feType II PS (R17)": "R17 FeType II PS",
    "eType II Doppler (R18)": "R18 eType II Doppler",
}


@dataclass(frozen=True)
class Figure:
    slug: str
    title: str
    blurb: str
    honors_families: bool = True
    swept: tuple[str, ...] = field(default_factory=tuple)
    est_seconds: int = 40


FIGURES: dict[str, Figure] = {
    f.slug: f
    for f in [
        Figure("fig_01_se_vs_snr", "SE vs SNR",
               "SU spectral efficiency vs SNR (rank 1 & 2) against the eigen bound.",
               est_seconds=45),
        Figure("fig_02_rate_distortion", "Rate–distortion",
               "Feedback bits vs fidelity (SGCS / SE@10dB): the Pareto frontier.",
               est_seconds=55),
        Figure("fig_03_overhead_breakdown", "Overhead breakdown",
               "Per-element PMI bit anatomy (spatial / delay / Doppler / phase ...).",
               est_seconds=3),
        Figure("fig_04_overhead_scaling", "Overhead scaling laws",
               "Analytic feedback-bit growth vs N3, L, N4 (formula-based).",
               honors_families=False, swept=("families", "antenna", "n3"), est_seconds=2),
        Figure("fig_05_mobility", "Mobility / CSI aging",
               "CSI aging vs R18 predicted PMI under Doppler and feedback delay.",
               est_seconds=50),
        Figure("fig_06_mu_mimo", "MU-MIMO sum rate",
               "Zero-forcing sum rate from reported PMIs vs SNR and user count.",
               est_seconds=45),
        Figure("fig_07_rank_adaptation", "Rank adaptation",
               "SE vs SNR at fixed ranks + auto-RI envelope (fixed Type I vs R16).",
               honors_families=False, swept=("families",), est_seconds=50),
        Figure("fig_08_channel_sensitivity", "Channel sensitivity",
               "Robustness of SGCS to channel sparsity and estimation noise.",
               est_seconds=55),
        Figure("fig_09_port_selection", "Port selection",
               "Regular vs port-selection codebooks across antenna/beam domains.",
               honors_families=False, swept=("families",), est_seconds=50),
        Figure("fig_10_array_scaling", "Antenna array scaling",
               "SE / gap / SGCS / bits vs port count (sweeps the array itself).",
               swept=("antenna",), est_seconds=50),
        Figure("fig_11_frequency_granularity", "Frequency granularity",
               "Fidelity and cost vs N3 (sweeps the frequency granularity itself).",
               swept=("n3",), est_seconds=45),
        Figure("fig_12_summary", "Summary scorecard",
               "Radar chart: 5 normalized axes (SE, SGCS, compactness, mobility).",
               est_seconds=70),
        Figure("fig_13_new_codebooks", "Spec-completion codebooks",
               "2-port Type I, R19 refined Type I (modeA/modeB/multi-panel), "
               "R18 CJT mode1 vs mode2.",
               honors_families=False, swept=("families", "antenna", "n3"),
               est_seconds=10),
    ]
}


def build_env(cfg: dict) -> dict[str, str]:
    """Translate the figures-run config into ``NRCSI_*`` environment overrides."""
    env = os.environ.copy()
    env["NRCSI_N1"] = str(cfg["n1"])
    env["NRCSI_N2"] = str(cfg["n2"])
    env["NRCSI_N3"] = str(cfg["n3"])
    env["NRCSI_N_RX"] = str(cfg["n_rx"])

    tokens = [FAMILIES[f] for f in cfg.get("families", []) if f in FAMILIES]
    if tokens:
        env["NRCSI_FAMILIES"] = ",".join(tokens)

    if cfg["channel"] == "synthetic":
        env["NRCSI_N_PATHS"] = str(cfg["n_paths"])
        env["NRCSI_MAX_DELAY"] = str(cfg["max_delay"])
    else:  # cdl
        env["NRCSI_CDL_MODEL"] = str(cfg["cdl_model"])
        env["NRCSI_CDL_SPEED"] = str(cfg["cdl_speed"])
        env["NRCSI_CDL_SPEED_FAST"] = str(max(float(cfg["cdl_speed"]), 30.0))
        env["NRCSI_CDL_DS"] = repr(float(cfg["cdl_delay_spread_ns"]) * 1e-9)
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    return env


def _script(slug: str, channel: str) -> pathlib.Path:
    return FIG_DIR / (f"cdl_{slug}.py" if channel == "cdl" else f"{slug}.py")


def cdl_twin_exists(slug: str) -> bool:
    """Whether the figure has a Sionna/CDL variant (``cdl_<slug>.py``)."""
    return (FIG_DIR / f"cdl_{slug}.py").exists()


def run_figure(slug: str, cfg: dict, run: dict) -> dict:
    """Run one figure as a subprocess; return paths, parsed data, log, status."""
    channel = cfg["channel"]
    out_dir = CDL_OUT if channel == "cdl" else SYNTH_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{slug}.png"
    js = out_dir / f"{slug}.json"

    argv = [sys.executable, str(_script(slug, channel)),
            "--drops", str(run["drops"]), "--seed", str(run["seed"]),
            "--out", str(out_dir)]
    if run.get("fast"):
        argv.append("--fast")

    env = build_env(cfg)
    started = time.time()
    try:
        proc = subprocess.run(
            argv, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
            timeout=run.get("timeout", 1800),
        )
        log = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired as e:
        log = f"TIMEOUT after {run.get('timeout', 1800)}s\n" + (e.stdout or "") + (e.stderr or "")
        ok = False
    seconds = time.time() - started

    fresh = png.exists() and png.stat().st_mtime >= started - 1
    data = None
    if js.exists() and js.stat().st_mtime >= started - 1:
        try:
            data = json.loads(js.read_text())
        except json.JSONDecodeError:
            data = None
    return {
        "ok": ok and fresh,
        "png": png if fresh else None,
        "json_path": js if (fresh and data is not None) else None,
        "data": data,
        "log": log,
        "seconds": seconds,
        "cmd": " ".join(argv),
    }


def artifact_url(path: pathlib.Path | None) -> str | None:
    if path is None:
        return None
    try:
        rel = path.resolve().relative_to(REPO_ROOT / "results")
    except ValueError:
        return None
    return f"/artifacts/{rel.as_posix()}"


def run_figures_job(job, req: dict) -> None:
    """Executed on the job worker thread: run every requested slug in order,
    appending a result dict to ``job.results`` as each finishes."""
    slugs = list(req.get("slugs") or [])
    channel = req.get("channel", "synthetic")
    antenna = req.get("antenna") or {"n1": 4, "n2": 2}
    cfg = {
        "channel": channel,
        "n_rx": int(req.get("n_rx", 2)),
        "n1": int(antenna.get("n1", 4)),
        "n2": int(antenna.get("n2", 2)),
        "n3": int(req.get("n3", 8)),
        "families": req.get("families") or list(FAMILIES),
        "n_paths": req.get("n_paths", 4),
        "max_delay": req.get("max_delay", 3.0),
        "cdl_model": req.get("cdl_model", "C"),
        "cdl_speed": req.get("cdl_speed", 3.0),
        "cdl_delay_spread_ns": req.get("cdl_delay_spread_ns", 100.0),
    }
    fast = bool(req.get("fast", True))
    run = {
        "drops": int(req.get("drops", 100)),
        "seed": int(req.get("seed", 0)),
        "fast": fast,
        "timeout": 600 if fast else 2400,
    }

    total = max(len(slugs), 1)
    for i, slug in enumerate(slugs):
        fig = FIGURES.get(slug)
        title = fig.title if fig else slug
        job.message = f"Running {title} ({i + 1}/{len(slugs)})…"
        job.progress = i / total
        res = run_figure(slug, cfg, run)
        job.results.append({
            "slug": slug,
            "ok": res["ok"],
            "png_url": artifact_url(res["png"]),
            "json_url": artifact_url(res["json_path"]),
            "data": res["data"],
            "log": res["log"],
            "seconds": round(res["seconds"], 2),
        })
        job.progress = (i + 1) / total
    job.message = "Done"
