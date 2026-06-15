"""Backend for the Streamlit codebook explorer (``webapp/app.py``).

The web UI is a thin driver over the existing figure scripts: it turns the form
inputs into ``NRCSI_*`` environment variables plus ``--drops/--seed/--fast`` CLI
flags, runs the selected ``scripts/figures/fig_*.py`` (or its Sionna ``cdl_*``
twin) as a subprocess, then reads back the ``<slug>.png`` + ``<slug>.json`` the
script writes.  Keeping the figure code untouched means the web view always
matches what ``make_all_figures.py`` produces.

The ``NRCSI_*`` knobs are consumed in ``src/nr_csi/figtools/figlib.py`` and
``cdllib.py`` (antenna geometry, ``N3``, ``n_rx``, ray richness, family filter,
CDL model/speed/delay-spread).  Unset == the original benchmark defaults.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "scripts" / "figures"
SYNTH_OUT = REPO_ROOT / "results" / "webapp"
# CDL scripts force their output here (cdllib.run_original -> GALLERY); we cannot
# redirect it, so we read CDL results from this fixed location.
CDL_OUT = REPO_ROOT / "results" / "sionna_cdl_gallery"

# Display name -> NRCSI_FAMILIES token (must equal the codebook .name prefix that
# figlib.select_families / style() match on).
FAMILIES: dict[str, str] = {
    "Type I (R15)": "R15 Type I",
    "Type II (R15)": "R15 Type II",
    "eType II (R16)": "R16 eType II",
    "feType II PS (R17)": "R17 FeType II PS",
    "eType II Doppler (R18)": "R18 eType II Doppler",
}


@dataclass(frozen=True)
class Figure:
    slug: str  # filename stem written by the script (same for synthetic & CDL)
    title: str
    blurb: str
    honors_families: bool = True  # does the family filter change the curves?
    swept: tuple[str, ...] = field(default_factory=tuple)  # config fields it ignores
    est_seconds: int = 40  # rough full-quality runtime, for the UI hint


# Order mirrors the gallery; ``swept`` lists config fields the figure varies
# internally (so the corresponding form input does not apply to it).
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
    ]
}


def build_env(cfg: dict) -> dict[str, str]:
    """Translate the form config into ``NRCSI_*`` environment overrides.

    ``cfg`` keys: families (list of display names), n1, n2, n3, n_rx,
    channel ('synthetic'|'cdl'), and channel-specific knobs (n_paths/max_delay
    or cdl_model/cdl_speed/cdl_delay_spread_ns).
    """
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
        # mobility figures use the "fast" regime; keep it >= the chosen speed so
        # a user-requested high speed is honoured there too.
        env["NRCSI_CDL_SPEED_FAST"] = str(max(float(cfg["cdl_speed"]), 30.0))
        env["NRCSI_CDL_DS"] = repr(float(cfg["cdl_delay_spread_ns"]) * 1e-9)
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    return env


def _script(slug: str, channel: str) -> pathlib.Path:
    return FIG_DIR / (f"cdl_{slug}.py" if channel == "cdl" else f"{slug}.py")


def run_figure(slug: str, cfg: dict, run: dict) -> dict:
    """Run one figure as a subprocess; return paths, parsed data, log, status.

    ``run`` keys: drops (int), seed (int), fast (bool), timeout (int seconds).
    Returns a dict with: ok, png (Path|None), data (dict|None), log (str),
    seconds (float), cmd (str).
    """
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

    # Trust the artifact only if it was (re)written by this run.
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
        "data": data,
        "log": log,
        "seconds": seconds,
        "cmd": " ".join(argv),
    }


# ----------------------------------------------------------------- channel compare

CHANNEL_COMPARE = FIG_DIR / "channel_compare.py"
COMPARE_OUT = REPO_ROOT / "results" / "webapp"
CDL_MODELS = ["A", "B", "C", "D", "E"]


def run_channel_compare(spec: dict, run: dict) -> dict:
    """Run the channel-comparison script on a spec (see channel_compare.py).

    Writes the spec to a temp JSON, runs the script as a subprocess (so the
    Sionna/TensorFlow import stays out of the Streamlit process), and reads back
    the overlay PNG + the per-channel diagnostics/validation JSON.
    """
    COMPARE_OUT.mkdir(parents=True, exist_ok=True)
    png = COMPARE_OUT / "channel_compare.png"
    js = COMPARE_OUT / "channel_compare.json"
    spec_path = COMPARE_OUT / "channel_compare_spec.json"
    spec_path.write_text(json.dumps(spec))

    argv = [sys.executable, str(CHANNEL_COMPARE), "--spec", str(spec_path),
            "--out", str(COMPARE_OUT), "--seed", str(run.get("seed", 0))]
    if run.get("fast"):
        argv.append("--fast")

    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    started = time.time()
    try:
        proc = subprocess.run(argv, cwd=str(REPO_ROOT), env=env, capture_output=True,
                              text=True, timeout=run.get("timeout", 2400))
        log = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired as e:
        log = f"TIMEOUT\n{e.stdout or ''}{e.stderr or ''}"
        ok = False
    seconds = time.time() - started

    fresh = png.exists() and png.stat().st_mtime >= started - 1
    data = None
    if js.exists() and js.stat().st_mtime >= started - 1:
        try:
            data = json.loads(js.read_text())
        except json.JSONDecodeError:
            data = None
    return {"ok": ok and fresh, "png": png if fresh else None, "data": data,
            "log": log, "seconds": seconds, "cmd": " ".join(argv)}
