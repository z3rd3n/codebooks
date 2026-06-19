"""On-disk format for CDL datasets: sharded HDF5 + a JSON manifest.

Layout::

    <dataset_dir>/
      manifest.json                 # full recipe + provenance + per-config index
      <tag>/shard_00000.h5          # one config's samples, fixed port count P
      <tag>/shard_00001.h5
      ...

Each shard holds the clean, energy-normalized complex channel plus the per-sample
metadata needed to reproduce provenance and add noise at train time.  Splits are
recorded as *parameters* in the manifest (see :mod:`.splits`) rather than as
index lists, and reproduced on load.

``h5py`` is an optional dependency (the ``dataset`` extra); importing this module
is fine without it, only the read/write helpers require it.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import subprocess

import numpy as np

from .splits import split_indices

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "h5py is required for dataset I/O; install the optional extra: "
            'pip install -e ".[dataset]"'
        ) from exc
    return h5py


def git_commit() -> str | None:
    """Best-effort short commit hash of the repo, or None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # pragma: no cover
        return None


def sionna_version() -> str | None:
    try:
        import sionna
        return getattr(sionna, "__version__", None)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Shards
# --------------------------------------------------------------------------- #
def write_shard(path: pathlib.Path, H: np.ndarray, meta: dict, attrs: dict) -> None:
    """Write one shard.

    ``H``: (n, n_rx, P, n_freq) complex.  ``meta`` keys: ``norm`` (n,) float,
    ``cdl_model`` (n,) str, ``delay_spread_ns`` (n,) float, ``ue_speed_kmh`` (n,)
    float.  ``attrs``: scalar provenance written as HDF5 attributes."""
    h5py = _h5py()
    path.parent.mkdir(parents=True, exist_ok=True)
    n = H.shape[0]
    with h5py.File(path, "w") as f:
        f.create_dataset("H", data=H.astype(np.complex64), compression="gzip",
                         compression_opts=4)
        f.create_dataset("norm", data=np.asarray(meta["norm"], dtype=np.float32))
        f.create_dataset("delay_spread_ns",
                         data=np.asarray(meta["delay_spread_ns"], dtype=np.float32))
        f.create_dataset("ue_speed_kmh",
                         data=np.asarray(meta["ue_speed_kmh"], dtype=np.float32))
        f.create_dataset("cdl_model",
                         data=np.asarray(meta["cdl_model"], dtype="S1"))
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs["n_samples"] = n
        f.attrs["schema_version"] = SCHEMA_VERSION


def read_shard(path: pathlib.Path) -> dict:
    """Read a shard into a dict of numpy arrays + an ``attrs`` dict."""
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        out = {
            "H": f["H"][()],
            "norm": f["norm"][()],
            "delay_spread_ns": f["delay_spread_ns"][()],
            "ue_speed_kmh": f["ue_speed_kmh"][()],
            "cdl_model": np.array([s.decode() for s in f["cdl_model"][()]]),
            "attrs": dict(f.attrs),
        }
    return out


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def write_manifest(out_dir: pathlib.Path, cfg, config_entries: list[dict]) -> dict:
    """Write ``manifest.json`` = the config recipe + provenance + per-config index."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "sionna_version": sionna_version(),
        "config": cfg.to_dict(),
        "configs": config_entries,
        "total_samples": sum(e["n_samples"] for e in config_entries),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest(dataset_dir: str | pathlib.Path) -> dict:
    path = pathlib.Path(dataset_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {dataset_dir}")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Loading / iteration
# --------------------------------------------------------------------------- #
def _split_mask(n: int, manifest: dict, split: str | None) -> np.ndarray | None:
    """Boolean keep-mask of length ``n`` for ``split`` ('train'|'val'|'test'),
    or None to keep all."""
    if split is None:
        return None
    cfg = manifest["config"]
    idx = split_indices(n, tuple(cfg["splits"]), cfg["seed"], cfg["split_method"])
    mask = np.zeros(n, dtype=bool)
    mask[idx[split]] = True
    return mask


def load_config_array(
    dataset_dir: str | pathlib.Path, config_tag: str, split: str | None = None
) -> dict:
    """Load one config's samples (all shards concatenated) into memory.

    Returns ``{'H', 'norm', 'delay_spread_ns', 'ue_speed_kmh', 'cdl_model',
    'attrs'}``.  Use for audit / small configs; stream with :func:`iter_samples`
    for large ones."""
    dataset_dir = pathlib.Path(dataset_dir)
    manifest = read_manifest(dataset_dir)
    entry = next((e for e in manifest["configs"] if e["tag"] == config_tag), None)
    if entry is None:
        raise KeyError(f"config tag {config_tag!r} not in manifest")
    parts = [read_shard(dataset_dir / s) for s in entry["shards"]]
    out = {
        "H": np.concatenate([p["H"] for p in parts]),
        "norm": np.concatenate([p["norm"] for p in parts]),
        "delay_spread_ns": np.concatenate([p["delay_spread_ns"] for p in parts]),
        "ue_speed_kmh": np.concatenate([p["ue_speed_kmh"] for p in parts]),
        "cdl_model": np.concatenate([p["cdl_model"] for p in parts]),
        "attrs": parts[0]["attrs"] if parts else {},
    }
    mask = _split_mask(out["H"].shape[0], manifest, split)
    if mask is not None:
        for k in ("H", "norm", "delay_spread_ns", "ue_speed_kmh", "cdl_model"):
            out[k] = out[k][mask]
    return out


def iter_samples(
    dataset_dir: str | pathlib.Path, config_tag: str | None = None, split: str | None = None
):
    """Yield ``(H_sample, meta)`` one sample at a time across (optionally one)
    config and split, without holding the whole dataset in memory."""
    dataset_dir = pathlib.Path(dataset_dir)
    manifest = read_manifest(dataset_dir)
    for entry in manifest["configs"]:
        if config_tag is not None and entry["tag"] != config_tag:
            continue
        # split is per-config over that config's sample order
        offset = 0
        masks = None
        if split is not None:
            masks = _split_mask(entry["n_samples"], manifest, split)
        for shard_rel in entry["shards"]:
            shard = read_shard(dataset_dir / shard_rel)
            n = shard["H"].shape[0]
            for i in range(n):
                gi = offset + i
                if masks is not None and not masks[gi]:
                    continue
                yield shard["H"][i], {
                    "config_tag": entry["tag"],
                    "norm": float(shard["norm"][i]),
                    "delay_spread_ns": float(shard["delay_spread_ns"][i]),
                    "ue_speed_kmh": float(shard["ue_speed_kmh"][i]),
                    "cdl_model": str(shard["cdl_model"][i]),
                }
            offset += n
