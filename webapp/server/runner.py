"""Playground runs: channel + select/precoder + metrics + viz.

Implements SPEC.md §4: ``evaluate()`` for the metrics block, plus one extra
single drop with the request seed to produce the PMI/viz payloads.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import threading
from typing import Any

import numpy as np

from nr_csi.channel.synthetic import RandomRayChannel
from nr_csi.config import AntennaConfig
from nr_csi.eval.harness import evaluate

from . import catalog

# --------------------------------------------------------------- channel presets

CHANNEL_PRESETS: dict[str, dict] = {
    "sparse-urban": {"n_paths": 4, "max_delay": 3.0, "max_doppler": 0.0},
    "rich-scattering": {"n_paths": 12, "max_delay": 6.0, "max_doppler": 0.0},
    "near-los": {"n_paths": 2, "max_delay": 1.0, "max_doppler": 0.0},
    "mobile-user": {"n_paths": 4, "max_delay": 3.0, "max_doppler": 0.5},
}

DOPPLER_IDS = {"etype2-doppler-r18", "predicted-ps-r18"}

CDL_NO_SIONNA_MSG = (
    "3GPP CDL channels need the optional `sionna` extra, which isn't installed on "
    'this server. Use the synthetic channel instead, or install it with '
    '`pip install -e ".[sionna]"`.'
)
CDL_CJT_UNSUPPORTED_MSG = (
    "3GPP CDL isn't available yet for the CJT codebook — it needs per-transmission-"
    "point CDL channels that aren't implemented in this app. Use the synthetic "
    "channel instead."
)


class ValidationError(ValueError):
    """A friendly, already-formatted error to surface as a 400."""


def resolve_channel_cfg(channel_req: dict | None, codebook_id: str) -> dict:
    """Merge a preset with explicit overrides; explicit fields win."""
    channel_req = dict(channel_req or {})
    preset = channel_req.get("preset")
    default_doppler = 0.5 if codebook_id in DOPPLER_IDS else 0.0
    # Doppler/predicted codebooks need genuine movement to show anything; match
    # the "fast" regime the Doppler figure scripts already use for CDL speed.
    default_cdl_speed = 30.0 if codebook_id in DOPPLER_IDS else 3.0
    cfg = {
        "type": "synthetic",
        "n_rx": 2,
        "n_paths": 4,
        "max_delay": 3.0,
        "max_doppler": default_doppler,
        "cdl_model": "C",
        "cdl_speed_kmh": default_cdl_speed,
        "cdl_delay_spread_ns": 100.0,
    }
    if preset and preset in CHANNEL_PRESETS:
        cfg.update(CHANNEL_PRESETS[preset])
    for key in ("type", "n_rx", "n_paths", "max_delay", "max_doppler",
                "cdl_model", "cdl_speed_kmh", "cdl_delay_spread_ns"):
        if key in channel_req and channel_req[key] is not None:
            cfg[key] = channel_req[key]
    cfg["inter_trp_delay"] = float(channel_req.get("inter_trp_delay", 0.5))
    return cfg


# ------------------------------------------------------------------------ CDL

# Playground/Compare run in the same process (unlike Figure Lab, which isolates
# each run in its own subprocess), so concurrent requests could otherwise drive
# TensorFlow from multiple threads at once. This serializes all CDL channel
# construction/generation to keep that safe.
_SIONNA_LOCK = threading.Lock()


class _LockedChannel:
    """Wraps a channel so every ``generate()`` call is serialized."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.N3 = inner.N3
        self.n_rx = inner.n_rx
        self.n_ports = inner.n_ports

    def generate(self, n_slots: int = 1, rng: np.random.Generator | None = None):
        with _SIONNA_LOCK:
            return self._inner.generate(n_slots=n_slots, rng=rng)


def _build_cdl_channel(antenna: AntennaConfig, n3: int, chan_cfg: dict) -> "_LockedChannel":
    if importlib.util.find_spec("sionna") is None:
        raise ValidationError(CDL_NO_SIONNA_MSG)
    from nr_csi.channel.sionna_adapter import SionnaCDLChannel

    n_rx = int(chan_cfg["n_rx"])
    model = str(chan_cfg.get("cdl_model", "C"))
    speed = float(chan_cfg.get("cdl_speed_kmh", 3.0))
    ds_ns = float(chan_cfg.get("cdl_delay_spread_ns", 100.0))
    try:
        with _SIONNA_LOCK:
            inner = SionnaCDLChannel(
                antenna, N3=n3, model=model, n_rx=n_rx,
                ue_speed_kmh=speed, delay_spread=ds_ns * 1e-9,
            )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return _LockedChannel(inner)


def _build_antenna(entry: catalog.CatalogEntry, req: dict) -> AntennaConfig:
    return catalog.build_antenna(entry, req.get("antenna") or {})


def _n_slots(scheme) -> int:
    return int(getattr(scheme, "N4", 1))


def _cjt_channel(scheme, antenna: AntennaConfig, n3: int, chan_cfg: dict, inter_trp_delay: float):
    """Per-TRP RandomRayChannel concatenated on the port axis with a delay
    offset per TRP (fig_13's ``panel_cjt_delay`` construction, generalized to
    N_TRP resources)."""
    n_trp = scheme.n_trp
    chan_n_rx = int(chan_cfg["n_rx"])
    n_paths = int(chan_cfg["n_paths"])
    max_delay = float(chan_cfg["max_delay"])
    max_doppler = float(chan_cfg["max_doppler"])
    base_chan = RandomRayChannel(
        antenna, N3=n3, n_rx=chan_n_rx, n_paths=n_paths, max_delay=max_delay,
        max_doppler=max_doppler,
    )

    class _CJTChannel:
        N3 = n3
        n_rx = chan_n_rx
        n_ports = antenna.P * n_trp

        def generate(self, n_slots: int = 1, rng: np.random.Generator | None = None):
            rng = rng or np.random.default_rng()
            base = base_chan.generate(n_slots=n_slots, rng=rng)
            parts = [base]
            for j in range(1, n_trp):
                ramp = np.exp(2j * np.pi * np.arange(n3) * (inter_trp_delay * j) / n3)
                parts.append(base * ramp[None, :, None, None])
            return np.concatenate(parts, axis=-1)

    return _CJTChannel()


def _rank_bounds(entry: catalog.CatalogEntry) -> tuple[int, int]:
    return entry.ranks


def instantiate(entry: catalog.CatalogEntry, req: dict):
    """Build (scheme, antenna, n3, channel_cfg) from a resolved RunRequest,
    translating constructor ``ValueError``s into ``ValidationError``."""
    antenna = _build_antenna(entry, req)
    n3 = int(req.get("n3", 8))
    params = dict(req.get("params") or {})
    try:
        scheme = entry.factory(antenna, n3, params)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return scheme, antenna, n3


def _channel_for(
    entry: catalog.CatalogEntry, scheme, antenna: AntennaConfig, n3: int, chan_cfg: dict
):
    if chan_cfg.get("type") == "cdl":
        if entry.id == "cjt-r18":
            raise ValidationError(CDL_CJT_UNSUPPORTED_MSG)
        return _build_cdl_channel(antenna, n3, chan_cfg)
    if entry.id == "cjt-r18":
        return _cjt_channel(scheme, antenna, n3, chan_cfg, chan_cfg["inter_trp_delay"])
    n_rx = int(chan_cfg["n_rx"])
    n_paths = int(chan_cfg["n_paths"])
    max_delay = float(chan_cfg["max_delay"])
    max_doppler = float(chan_cfg["max_doppler"])
    doppler_period = 8 if max_doppler > 0 else 1
    return RandomRayChannel(
        antenna, N3=n3, n_rx=n_rx, n_paths=n_paths, max_delay=max_delay,
        max_doppler=max_doppler, doppler_period=doppler_period,
    )


def validate_request(req: dict) -> None:
    """Instantiate the scheme (no channel/eval) to check the config is legal."""
    codebook_id = req.get("codebook_id")
    entry = catalog.get_entry(codebook_id)
    if entry is None:
        raise ValidationError(f"unknown codebook id '{codebook_id}'")
    scheme, antenna, n3 = instantiate(entry, req)
    rank = int(req.get("rank", 1))
    lo, hi = _rank_bounds(entry)
    if not lo <= rank <= hi:
        raise ValidationError(f"rank {rank} outside this codebook's supported range [{lo}, {hi}]")

    # Cheap CDL checks (no TensorFlow import, no model construction) so the
    # debounced auto-validate can still catch these instantly.
    if (req.get("channel") or {}).get("type") == "cdl":
        if importlib.util.find_spec("sionna") is None:
            raise ValidationError(CDL_NO_SIONNA_MSG)
        if entry.id == "cjt-r18":
            raise ValidationError(CDL_CJT_UNSUPPORTED_MSG)


# ------------------------------------------------------------------- decimate


def _decimate_axis(arr: np.ndarray, axis: int, cap: int = 64) -> np.ndarray:
    n = arr.shape[axis]
    if n <= cap:
        return arr
    idx = np.linspace(0, n - 1, cap).round().astype(int)
    idx = np.unique(idx)
    return np.take(arr, idx, axis=axis)


def _decimate_matrix(arr: np.ndarray, cap: int = 64) -> np.ndarray:
    out = arr
    for axis in range(out.ndim):
        out = _decimate_axis(out, axis, cap)
    return out


def _round_list(arr: np.ndarray, decimals: int = 4):
    return np.round(arr.astype(float), decimals).tolist()


def _matrix_payload(complex_mat: np.ndarray) -> dict:
    dec = _decimate_matrix(np.asarray(complex_mat))
    return {"abs": _round_list(np.abs(dec)), "rows": dec.shape[0], "cols": dec.shape[1]}


# ------------------------------------------------------------------- PMI summary


def _summarize_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        arr = value
        flat_preview = arr.flatten()[:6]
        preview = ", ".join(_fmt_scalar(x) for x in flat_preview)
        suffix = ", ..." if arr.size > 6 else ""
        return f"shape {tuple(arr.shape)}, e.g. [{preview}{suffix}]"
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            preview = ", ".join(_fmt_scalar(x) for x in value[:6])
            return f"{len(value)} entries, e.g. [{preview}, ...]"
        return "[" + ", ".join(_fmt_scalar(x) for x in value) + "]"
    return _fmt_scalar(value)


def _fmt_scalar(x: Any) -> str:
    if isinstance(x, (np.floating, float)):
        return f"{float(x):.4g}"
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, np.bool_):
        return str(bool(x))
    return str(x)


def summarize_pmi(scheme, pmi) -> dict:
    """PMI field summaries: overhead_bits(pmi) gives per-field bits; values
    are summarized compactly from the PMI dataclass."""
    bits = scheme.overhead_bits(pmi)
    field_values: dict[str, Any] = {}
    if dataclasses.is_dataclass(pmi):
        for f in dataclasses.fields(pmi):
            field_values[f.name] = getattr(pmi, f.name)
    fields = []
    for name, b in bits.items():
        value = field_values.get(name)
        if value is None:
            # some overhead keys are synthetic (e.g. "cmr"); fall back to a
            # best-effort match against the PMI's own field names
            value = field_values.get(name.split("_")[0])
        fields.append({
            "name": name,
            "value": _summarize_value(value) if value is not None else "(derived)",
            "bits": int(b),
            "description": "",
        })
    return {"fields": fields}


# ------------------------------------------------------------------- beam grid


def _beam_grid(entry: catalog.CatalogEntry, scheme, pmi, antenna: AntennaConfig) -> dict | None:
    """Derive selected beam (l, m) coordinates on the oversampled DFT grid for
    Type I single-panel and the Type II family, from the reported PMI."""
    try:
        g1, g2 = antenna.n_beams
    except Exception:
        return None

    def grid_payload(points: list[tuple[int, int]]) -> dict:
        return {"g1": int(g1), "g2": int(g2), "selected": [[int(a), int(b)] for a, b in points]}

    cid = entry.id
    try:
        if cid == "type1-sp":
            l0, m0 = pmi.i11, pmi.i12
            pts = [(l0, m0)]
            return grid_payload(pts)
        if cid in ("type2-r15", "etype2-r16") and not getattr(scheme, "port_selection", False):
            from nr_csi.utils import combinatorics as cb

            L = scheme.L
            n1, n2 = cb.decode_beam_combination(pmi.i12, antenna.N1, antenna.N2, L)
            q1, q2 = pmi.q1, pmi.q2
            pts = [(antenna.O1 * a + q1, antenna.O2 * b + q2) for a, b in zip(n1, n2)]
            return grid_payload(pts)
        if cid == "fetype2-r17":
            return None  # free port selection, no DFT-grid beams to plot
        if cid == "refined-type1-r19" and getattr(scheme, "mode", None) == "modeA":
            pts = [(b[0], b[1]) for b in scheme._beams(pmi)]
            return grid_payload(pts)
    except Exception:
        return None
    return None


# ------------------------------------------------------------------- snippet


def build_python_snippet(
    entry: catalog.CatalogEntry, req: dict, antenna: AntennaConfig, n3: int,
    rank: int, chan_cfg: dict,
) -> str:
    params = dict(req.get("params") or {})
    snr_db = req.get("snr_db") or [-10, -5, 0, 5, 10, 15, 20, 25, 30]
    drops = int(req.get("drops", 8))
    seed = int(req.get("seed", 0))

    ant_line = f"antenna = AntennaConfig.standard(N1={antenna.N1}, N2={antenna.N2}"
    if antenna.Ng != 1:
        ant_line += f", Ng={antenna.Ng}"
    ant_line += ")"

    factory_name = entry.factory.__name__

    if chan_cfg.get("type") == "cdl":
        channel_import = "from nr_csi.channel.sionna_adapter import SionnaCDLChannel"
        channel_line = (
            f"channel = SionnaCDLChannel(antenna, N3={n3}, model={chan_cfg.get('cdl_model', 'C')!r}, "
            f"n_rx={int(chan_cfg['n_rx'])}, ue_speed_kmh={chan_cfg.get('cdl_speed_kmh', 3.0)}, "
            f"delay_spread={chan_cfg.get('cdl_delay_spread_ns', 100.0)}e-9)  # needs the `sionna` extra"
        )
    else:
        channel_import = "from nr_csi.channel.synthetic import RandomRayChannel"
        channel_line = (
            f"channel = RandomRayChannel(antenna, N3={n3}, n_rx={int(chan_cfg['n_rx'])}, "
            f"n_paths={int(chan_cfg['n_paths'])}, max_delay={chan_cfg['max_delay']}, "
            f"max_doppler={chan_cfg['max_doppler']})"
        )

    lines = [
        "from nr_csi.config import AntennaConfig",
        channel_import,
        "from nr_csi.eval.harness import evaluate",
        f"# codebook: {entry.id} ({entry.name})",
        "# (constructor call mirrors webapp/server/catalog.py's factory for this entry)",
        "",
        ant_line,
        channel_line,
        f"# params = {params!r}",
        f"# scheme = <{entry.id} factory>(antenna, N3={n3}, params)"
        f"  # see catalog.py:{factory_name}",
        "",
        f"result = evaluate(scheme, channel, snr_db={list(snr_db)}, rank={rank}, "
        f"n_drops={drops}, rng=__import__('numpy').random.default_rng({seed}))",
        "print(result.se, result.sgcs, result.overhead_bits)",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------- main run


def run_playground(req: dict) -> dict:
    import time

    started = time.time()
    codebook_id = req.get("codebook_id")
    entry = catalog.get_entry(codebook_id)
    if entry is None:
        raise ValidationError(f"unknown codebook id '{codebook_id}'")

    scheme, antenna, n3 = instantiate(entry, req)
    rank = int(req.get("rank", 1))
    lo, hi = _rank_bounds(entry)
    if not lo <= rank <= hi:
        raise ValidationError(f"rank {rank} outside this codebook's supported range [{lo}, {hi}]")

    drops = min(int(req.get("drops", 8)), 64)
    snr_db = list(req.get("snr_db") or [-10, -5, 0, 5, 10, 15, 20, 25, 30])[:12]
    seed = int(req.get("seed", 0))
    chan_cfg = resolve_channel_cfg(req.get("channel"), codebook_id)

    channel = _channel_for(entry, scheme, antenna, n3, chan_cfg)
    n_slots = _n_slots(scheme)

    rng = np.random.default_rng(seed)
    try:
        result = evaluate(
            scheme, channel, snr_db=snr_db, rank=rank, n_drops=drops,
            n_slots=n_slots, rng=rng,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    # one extra single drop for pmi/viz, fresh channel, same seed family
    rng2 = np.random.default_rng(seed)
    H = channel.generate(n_slots=n_slots, rng=rng2)
    try:
        pmi = scheme.select(H, rank=rank)
        W = scheme.precoder(pmi)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    overhead = scheme.overhead_bits(pmi)
    total_bits = int(sum(overhead.values()))

    # viz: channel |H| at slot 0, rx 0; eigen spectrum; precoder per layer
    H0 = H[0, :, 0, :]  # (N3, P)
    channel_viz = _matrix_payload(H0)
    channel_viz["rows"] = "N3 frequency units"
    channel_viz["cols"] = "P ports"

    # eigen spectrum: mean singular values of H across N3 (slot 0)
    Hs = H[0]  # (N3, Nr, P)
    svals = np.linalg.svd(Hs, compute_uv=False)  # (N3, min(Nr,P))
    eigen_spectrum = _round_list(np.mean(svals, axis=0))

    precoder_viz = []
    W0 = W[0]  # (N3, P, rank)
    for li in range(W0.shape[-1]):
        layer = W0[:, :, li]
        mat = _decimate_matrix(layer)
        precoder_viz.append({
            "layer": li + 1,
            "abs": _round_list(np.abs(mat)),
            "phase": _round_list(np.angle(mat)),
        })

    beam_grid = _beam_grid(entry, scheme, pmi, antenna)

    snippet = build_python_snippet(entry, req, antenna, n3, rank, chan_cfg)

    config_echo = {
        "codebook_id": codebook_id,
        "params": dict(req.get("params") or {}),
        "antenna": {"n1": antenna.N1, "n2": antenna.N2, "ng": antenna.Ng},
        "n3": n3,
        "rank": rank,
        "channel": chan_cfg,
        "snr_db": snr_db,
        "drops": drops,
        "seed": seed,
    }

    se_at_10 = _interp_at(snr_db, result.se, 10.0)
    bound_at_10 = _interp_at(snr_db, result.se_upper_bound, 10.0)

    return {
        "ok": True,
        "seconds": round(time.time() - started, 3),
        "scheme_name": scheme.name,
        "config_echo": config_echo,
        "python_snippet": snippet,
        "metrics": {
            "sgcs": round(result.sgcs, 4),
            "subspace_sgcs": round(result.subspace_sgcs, 4),
            "snr_db": snr_db,
            "se": _round_list(np.array(result.se)),
            "se_upper_bound": _round_list(np.array(result.se_upper_bound)),
            "overhead_bits": {k: int(v) for k, v in overhead.items()},
            "total_bits": total_bits,
            "se_at_10db": se_at_10,
            "bound_at_10db": bound_at_10,
        },
        "pmi": summarize_pmi(scheme, pmi),
        "viz": {
            "channel": channel_viz,
            "eigen_spectrum": eigen_spectrum,
            "precoder": precoder_viz,
            **({"beam_grid": beam_grid} if beam_grid is not None else {}),
        },
    }


def _interp_at(xs: list[float], ys: list[float], target: float) -> float | None:
    if not xs:
        return None
    if target in xs:
        return round(float(ys[xs.index(target)]), 4)
    if target < xs[0] or target > xs[-1]:
        return None
    return round(float(np.interp(target, xs, ys)), 4)
