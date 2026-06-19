"""Audit / QA a generated CDL dataset.

Reuses ``nr_csi.channel.diagnostics`` to verify a dataset is physically sane
before any training: realized delay spread tracks the configured value, NLOS
channels have rich (flat) spatial spectra, and the angular-delay representation
is sparse (energy concentrated in early taps).

Writes ``<dataset_dir>/audit.png`` + ``<dataset_dir>/audit.json`` (artifact-on-
disk, so the webapp Dataset tab can read them back) and prints a console summary.

    python scripts/dataset/audit_dataset.py data/cdl_smoke
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nr_csi.channel.diagnostics import (  # noqa: E402
    power_delay_profile,
    rms_delay_spread,
    singular_value_spectrum,
    spatial_covariance_spectrum,
    taps_to_seconds,
)
from nr_csi.dataset import io  # noqa: E402


def _load_capped(dataset_dir: pathlib.Path, entry: dict, cap: int) -> dict:
    """Read shards of one config until ``cap`` samples are gathered."""
    parts: list[dict] = []
    got = 0
    for shard_rel in entry["shards"]:
        s = io.read_shard(dataset_dir / shard_rel)
        parts.append(s)
        got += s["H"].shape[0]
        if got >= cap:
            break
    H = np.concatenate([p["H"] for p in parts])[:cap]
    ds = np.concatenate([p["delay_spread_ns"] for p in parts])[:cap]
    model = np.concatenate([p["cdl_model"] for p in parts])[:cap]
    return {"H": H, "delay_spread_ns": ds, "cdl_model": model,
            "attrs": parts[0]["attrs"]}


def _effective_rank(spectrum: np.ndarray) -> float:
    """Participation ratio (1 / sum p_i^2) of a normalized spectrum -- the
    number of 'active' modes; near 1 for LoS, large for rich NLOS."""
    p = spectrum / spectrum.sum()
    return float(1.0 / np.sum(p ** 2))


def _ds_monotonicity(H_diag: np.ndarray, ds_ns: np.ndarray, fft_size: int, scs: float,
                     n_bins: int = 5) -> list[dict]:
    """Bin samples by configured DS; report measured RMS delay spread per bin."""
    edges = np.quantile(ds_ns, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (ds_ns >= lo) & (ds_ns <= hi if i == n_bins - 1 else ds_ns < hi)
        if sel.sum() < 2:
            continue
        taps = rms_delay_spread(H_diag[sel])
        rows.append({
            "configured_ns_mid": float((lo + hi) / 2),
            "measured_ns": float(taps_to_seconds(taps, fft_size, scs) * 1e9),
            "n": int(sel.sum()),
        })
    return rows


def audit(dataset_dir: pathlib.Path, cap: int) -> dict:
    manifest = io.read_manifest(dataset_dir)
    cfg = manifest["config"]
    fft_size, scs = cfg["fft_size"], cfg["subcarrier_spacing"]
    configs = manifest["configs"]

    summary = {"dataset_dir": str(dataset_dir), "total_samples": manifest["total_samples"],
               "configs": []}
    fig, axes = plt.subplots(len(configs), 4, figsize=(18, 3.6 * len(configs)),
                             squeeze=False)
    for r, entry in enumerate(configs):
        data = _load_capped(dataset_dir, entry, cap)
        # stored H is (n, rx, port, n_freq) -> diagnostics want (n, N3, rx, port)
        H_diag = data["H"].transpose(0, 3, 1, 2)
        pdp = power_delay_profile(H_diag)
        cum = np.cumsum(pdp)
        sv = singular_value_spectrum(H_diag)
        cov = spatial_covariance_spectrum(H_diag)
        ds_taps = rms_delay_spread(H_diag)
        n_freq = pdp.size
        k16 = min(16, n_freq)
        cs = {
            "tag": entry["tag"], "P": entry["P"], "n_audited": int(data["H"].shape[0]),
            "rms_delay_spread_taps": float(ds_taps),
            "rms_delay_spread_ns": float(taps_to_seconds(ds_taps, fft_size, scs) * 1e9),
            "energy_first_16_taps": float(cum[k16 - 1]),
            "spatial_effective_rank": _effective_rank(cov),
            "mimo_effective_rank": _effective_rank(sv),
            "models_present": sorted(set(data["cdl_model"].tolist())),
            "ds_monotonicity": _ds_monotonicity(H_diag, data["delay_spread_ns"],
                                                 fft_size, scs),
        }
        summary["configs"].append(cs)

        ax = axes[r]
        ax[0].semilogy(np.arange(n_freq), pdp + 1e-12)
        ax[0].set(title=f"{entry['tag']}  PDP", xlabel="delay tap", ylabel="norm. power")
        ax[0].grid(alpha=0.3)
        ax[1].plot(np.arange(n_freq), cum)
        ax[1].axvline(k16, color="r", ls="--", lw=0.8)
        ax[1].set(title=f"cum. delay energy ({cs['energy_first_16_taps']:.2f} @16)",
                  xlabel="delay tap", ylim=(0, 1.02))
        ax[1].grid(alpha=0.3)
        ax[2].plot(np.arange(sv.size), sv, marker="o", ms=3)
        ax[2].set(title=f"MIMO sing. spectrum (rank≈{cs['mimo_effective_rank']:.1f})",
                  xlabel="mode")
        ax[2].grid(alpha=0.3)
        ax[3].semilogy(np.arange(cov.size), cov + 1e-12)
        ax[3].set(title=f"spatial cov. (eff. rank≈{cs['spatial_effective_rank']:.1f})",
                  xlabel="eigenmode")
        ax[3].grid(alpha=0.3)

    fig.suptitle(f"CDL dataset audit — {dataset_dir.name} "
                 f"({manifest['total_samples']} samples, profiles {cfg['profiles']})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = dataset_dir / "audit.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    (dataset_dir / "audit.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_dir", type=pathlib.Path)
    p.add_argument("--max-samples", type=int, default=3000,
                   help="samples per config to audit (caps load time)")
    args = p.parse_args()

    summary = audit(args.dataset_dir, args.max_samples)
    print(f"Audited {summary['total_samples']} samples in {args.dataset_dir}")
    for c in summary["configs"]:
        print(f"  {c['tag']:>12} (P={c['P']:>3}): "
              f"DS≈{c['rms_delay_spread_ns']:.0f} ns, "
              f"delay energy@16={c['energy_first_16_taps']:.2f}, "
              f"spatial rank≈{c['spatial_effective_rank']:.1f}, "
              f"models={c['models_present']}")
    print(f"Wrote {args.dataset_dir}/audit.png and audit.json")


if __name__ == "__main__":
    main()
