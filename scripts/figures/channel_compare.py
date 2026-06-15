"""Compare channel models on the PMI grid -- delay, Doppler, and spatial views.

Driven by a JSON spec (``--spec``) so the web UI can request an arbitrary set of
channels to overlay; also runnable standalone.  All channels share one antenna /
N3 / n_rx / slot grid so their diagnostics are directly comparable.

    python scripts/figures/channel_compare.py --spec spec.json --out results/

Spec schema::

    {
      "antenna": {"n1": 4, "n2": 2}, "n3": 16, "n_rx": 2,
      "n_slots": 16, "interval_ms": 2.0, "drops": 30,
      "channels": [
        {"label": "CDL-C 100ns 3km/h", "type": "cdl",
         "model": "C", "speed": 3.0, "delay_spread_ns": 100.0},
        {"label": "synthetic 4-ray",   "type": "synthetic",
         "n_paths": 4, "max_delay": 3.0, "max_doppler": 0.5}
      ]
    }

The four panels (power-delay profile, frequency correlation, temporal
correlation, singular-value spectrum) plus the validation table characterise the
three axes the codebooks compress, and let an obvious adapter bug (a flat PDP, a
delay spread that ignores its setting, a Doppler that ignores UE speed) show up
at a glance.  See ``nr_csi.channel.diagnostics`` for the metric definitions.
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
    coherence_lag,
    freq_correlation,
    power_delay_profile,
    rms_delay_spread,
    spatial_covariance_spectrum,
    taps_to_seconds,
    time_correlation,
)
from nr_csi.config import AntennaConfig  # noqa: E402

FFT_SIZE = 256
SCS = 30e3  # subcarrier spacing (Hz); together they set the delay-tap scale
CARRIER = 3.5e9
LIGHT = 3e8
COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#A2142F"]


def _build_channel(spec: dict, ant: AntennaConfig, n3: int, n_rx: int, interval_s: float):
    if spec["type"] == "synthetic":
        from nr_csi.channel import RandomRayChannel

        return RandomRayChannel(
            ant, N3=n3, n_rx=n_rx,
            n_paths=int(spec.get("n_paths", 4)),
            max_delay=float(spec.get("max_delay", 3.0)),
            max_doppler=float(spec.get("max_doppler", 0.0)),
            doppler_period=int(spec.get("doppler_period", max(n3, 1))),
        )
    if spec["type"] == "cdl":
        from nr_csi.channel.sionna_adapter import SionnaCDLChannel

        return SionnaCDLChannel(
            ant, N3=n3, model=str(spec.get("model", "C")), n_rx=n_rx,
            ue_speed_kmh=float(spec.get("speed", 3.0)),
            delay_spread=float(spec.get("delay_spread_ns", 100.0)) * 1e-9,
            interval_duration=interval_s, fft_size=FFT_SIZE, subcarrier_spacing=SCS,
        )
    raise ValueError(f"unknown channel type: {spec['type']!r}")


def _collect(channel, n_slots: int, drops: int, seed: int) -> np.ndarray:
    """Stack ``drops`` realisations into (drops, n_slots, N3, rx, port)."""
    rng = np.random.default_rng(seed)
    return np.stack([channel.generate(n_slots=n_slots, rng=rng) for _ in range(drops)])


def _validate(spec: dict, H: np.ndarray, interval_s: float) -> dict:
    """Per-channel correctness summary (measured vs configured where defined)."""
    rho_f = freq_correlation(H)
    rho_t = time_correlation(H)
    ds_taps = rms_delay_spread(H)
    row = {
        "label": spec["label"],
        "type": spec["type"],
        "finite": bool(np.all(np.isfinite(H))),
        "rms_ds_taps": round(ds_taps, 3),
        "coh_bw_units": round(coherence_lag(rho_f), 2),
        "coh_time_slots": round(coherence_lag(rho_t), 2),
        "lambda1_frac": round(float(spatial_covariance_spectrum(H)[0]), 3),
    }
    if spec["type"] == "cdl":
        meas_ns = taps_to_seconds(ds_taps, FFT_SIZE, SCS) * 1e9
        nominal = float(spec.get("delay_spread_ns", 100.0))
        row["rms_ds_ns_measured"] = round(meas_ns, 1)
        row["rms_ds_ns_nominal"] = nominal
        f_d = float(spec.get("speed", 3.0)) / 3.6 * CARRIER / LIGHT
        row["doppler_hz"] = round(f_d, 2)
        row["coh_time_ms"] = round(coherence_lag(rho_t) * interval_s * 1e3, 2)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=pathlib.Path, required=True, help="channel comparison spec JSON")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"))
    p.add_argument("--drops", type=int, default=None, help="override spec drops")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fast", action="store_true", help="few drops (smoke test)")
    args = p.parse_args()

    spec = json.loads(args.spec.read_text())
    ant = AntennaConfig.standard(int(spec["antenna"]["n1"]), int(spec["antenna"]["n2"]))
    n3 = int(spec.get("n3", 16))
    n_rx = int(spec.get("n_rx", 2))
    n_slots = int(spec.get("n_slots", 16))
    interval_s = float(spec.get("interval_ms", 2.0)) * 1e-3
    drops = args.drops if args.drops is not None else int(spec.get("drops", 30))
    if args.fast:
        drops = min(drops, 4)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    data: dict = {"channels": [], "validation": [], "n3": n3, "n_slots": n_slots}

    for i, ch in enumerate(spec["channels"]):
        color = COLORS[i % len(COLORS)]
        try:
            channel = _build_channel(ch, ant, n3, n_rx, interval_s)
            H = _collect(channel, n_slots, drops, args.seed + i)
        except Exception as e:  # surface adapter/sionna errors per channel
            data["validation"].append({"label": ch.get("label", f"ch{i}"), "error": str(e)})
            print(f"FAILED {ch.get('label')!r}: {e}")
            continue

        pdp = power_delay_profile(H)
        rho_f = freq_correlation(H)
        rho_t = time_correlation(H)
        spec_sv = spatial_covariance_spectrum(H)
        row = _validate(ch, H, interval_s)
        data["validation"].append(row)
        data["channels"].append({
            "label": ch["label"], "pdp": pdp.tolist(), "freq_corr": rho_f.tolist(),
            "time_corr": rho_t.tolist(), "sv_spectrum": spec_sv.tolist(),
        })

        lbl = ch["label"]
        taps = np.fft.fftshift(np.fft.fftfreq(n3) * n3)  # centered: -N3/2 .. N3/2-1
        axes[0, 0].plot(taps, 10 * np.log10(np.fft.fftshift(pdp) + 1e-6),
                        "-o", ms=3, color=color, label=lbl)
        axes[0, 1].plot(np.arange(len(rho_f)), rho_f, "-o", ms=3, color=color, label=lbl)
        axes[1, 0].plot(np.arange(len(rho_t)), rho_t, "-o", ms=3, color=color, label=lbl)
        axes[1, 1].plot(np.arange(1, len(spec_sv) + 1), spec_sv, "-o", ms=4, color=color, label=lbl)

    axes[0, 0].set(title="Power-delay profile", xlabel="delay (taps, circular)",
                   ylabel="normalized power (dB)")
    axes[0, 1].set(title="Frequency correlation", xlabel="frequency-unit lag",
                   ylabel=r"$|\rho|$", ylim=(0, 1.02))
    axes[0, 1].axhline(0.5, color="0.6", ls="--", lw=0.8)
    axes[1, 0].set(title="Temporal (Doppler) correlation", xlabel="slot lag",
                   ylabel=r"$|\rho|$", ylim=(0, 1.02))
    axes[1, 0].axhline(0.5, color="0.6", ls="--", lw=0.8)
    axes[1, 1].set(title="Spatial covariance spectrum (angular richness / LoS)",
                   xlabel="eigenvalue index", ylabel="normalized power")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Channel comparison -- ({ant.N1},{ant.N2}) array, N3={n3}, "
                 f"n_rx={n_rx}, {n_slots} slots, {drops} drops", fontsize=12)
    fig.tight_layout()

    args.out.mkdir(parents=True, exist_ok=True)
    png = args.out / "channel_compare.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    (args.out / "channel_compare.json").write_text(json.dumps(data, indent=1))
    print(f"saved {png}")
    for row in data["validation"]:
        print(row)


if __name__ == "__main__":
    main()
