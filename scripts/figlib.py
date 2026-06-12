"""Shared helpers for the codebook-comparison figure scripts (fig_*.py).

Conventions (see plans/plan_figures.md):

* Default benchmark: dual-pol (4,2) array (P = 16), N3 = 8, sparse
  multipath ``RandomRayChannel``, rank 1, paired seeds across schemes.
* Standard scheme set at matched L = 4 spatial bases per polarization.
* Port-selection codebooks are evaluated through ``BeamDomainChannel``, a
  unitary per-polarization DFT PEB view of the same physical drops -- their
  deployment model (the gNB beamforms the CSI-RS first).  Unitarity keeps
  SE/SGCS directly comparable with antenna-domain schemes.
* Every figure saves ``results/<name>.png`` plus ``results/<name>.json``
  with the exact plotted numbers.
* Common CLI: --drops, --seed, --out, --fast (smoke-test sizes).
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

from nr_csi.channel import RandomRayChannel  # noqa: E402
from nr_csi.channel.base import ChannelSource  # noqa: E402
from nr_csi.codebooks import (  # noqa: E402
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18Type2Codebook,
    Type1Codebook,
)
from nr_csi.config import AntennaConfig  # noqa: E402
from nr_csi.eval import evaluate  # noqa: E402
from nr_csi.utils import dft  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

# Default benchmark configuration shared by the Monte-Carlo figures.
ANT = AntennaConfig.standard(4, 2)  # P = 16 CSI-RS ports
N3 = 8

# One fixed color/marker per codebook family across every figure.
STYLE: dict[str, dict] = {
    "R15 Type I": dict(color="#0072BD", marker="o"),
    "R15 Type II": dict(color="#D95319", marker="s"),
    "R15 Type II PS": dict(color="#D95319", marker="s", linestyle="--"),
    "R16 eType II": dict(color="#EDB120", marker="^"),
    "R16 eType II PS": dict(color="#EDB120", marker="^", linestyle="--"),
    "R17 FeType II PS": dict(color="#7E2F8E", marker="D"),
    "R18 eType II Doppler": dict(color="#77AC30", marker="v"),
    "eigen upper bound": dict(color="0.35", linestyle="--", marker=""),
    "full CSI ZF": dict(color="0.35", linestyle="--", marker=""),
}


def style(name: str) -> dict:
    """Plot kwargs for a scheme name (longest STYLE key that prefixes it)."""
    best = max((k for k in STYLE if name.startswith(k)), key=len, default=None)
    return dict(STYLE[best]) if best else {}


def standard_schemes(ant: AntennaConfig = ANT, n3: int = N3, N4: int = 4) -> list[tuple]:
    """The five-family comparison set at matched L = 4, as (scheme, domain)
    pairs with domain in {"antenna", "beam"} (beam = evaluate through the
    DFT PEB, the port-selection deployment model)."""
    return [
        (Type1Codebook(ant, N3=n3), "antenna"),
        (R15Type2Codebook(ant, N3=n3, L=4), "antenna"),
        (R16Type2Codebook(ant, N3=n3, param_combination=6), "antenna"),
        (R17Type2Codebook(ant, N3=n3, param_combination=5), "beam"),
        (R18Type2Codebook(ant, N3=n3, N4=N4, param_combination=7), "antenna"),
    ]


class BeamDomainChannel(ChannelSource):
    """Unitary per-polarization DFT PEB view of a wrapped channel source.

    Port-selection codebooks assume the gNB has already applied a port
    external beamformer to the CSI-RS; feeding them the raw antenna-domain
    channel understates them (see scripts/compare_schemes.py).  The PEB
    used here is the full orthogonal DFT group, which is unitary per
    polarization, so SE, SGCS, and the eigen upper bound computed in this
    domain equal their physical-domain values exactly.
    """

    def __init__(self, inner: ChannelSource, antenna: AntennaConfig) -> None:
        self.inner = inner
        # orthogonal_group rows have norm sqrt(N1*N2); rescale so F is
        # unitary and the PEB preserves channel power (honest SE numbers).
        self.F = dft.orthogonal_group(antenna, 0, 0).T / np.sqrt(antenna.n_ports_per_pol)
        self.half = antenna.P // 2

    def generate(self, n_slots: int = 1, rng: np.random.Generator | None = None):
        H = self.inner.generate(n_slots=n_slots, rng=rng)
        return np.concatenate(
            [H[..., : self.half] @ self.F, H[..., self.half :] @ self.F], axis=-1
        )


def run_eval(
    scheme,
    channel: ChannelSource,
    domain: str = "antenna",
    *,
    seed: int = 0,
    antenna: AntennaConfig = ANT,
    **kwargs,
):
    """``evaluate`` with paired seeding and the scheme's natural slot count.

    Every call builds a fresh ``default_rng(seed)``, so schemes evaluated
    with the same seed see the *same* channel drops (paired comparison).
    ``domain="beam"`` wraps the channel in the DFT PEB first.
    """
    if domain == "beam":
        channel = BeamDomainChannel(channel, antenna)
    kwargs.setdefault("n_slots", getattr(scheme, "N4", 1))
    return evaluate(scheme, channel, rng=np.random.default_rng(seed), **kwargs)


def default_channel(ant: AntennaConfig = ANT, n3: int = N3, **kwargs) -> RandomRayChannel:
    """The benchmark channel: sparse multipath, Type II's design regime but
    honest for Type I as well (off-grid angles/delays)."""
    kwargs.setdefault("n_rx", 2)
    kwargs.setdefault("n_paths", 4)
    kwargs.setdefault("max_delay", 3.0)
    return RandomRayChannel(ant, N3=n3, **kwargs)


def cli(description: str, drops: int = 100) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--drops", type=int, default=drops, help="Monte-Carlo drops per point")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed (paired across schemes)")
    p.add_argument("--out", type=pathlib.Path, default=RESULTS, help="output directory")
    p.add_argument("--fast", action="store_true", help="smoke-test sizes (few drops)")
    args = p.parse_args()
    if args.fast:
        args.drops = min(args.drops, 8)
    return args


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serializable: {type(obj)}")


def save(fig, out_dir: pathlib.Path, name: str, data: dict | None = None) -> None:
    """Save the figure as <name>.png and its plotted numbers as <name>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if data is not None:
        (out_dir / f"{name}.json").write_text(json.dumps(data, indent=1, default=_jsonable))
    print(f"saved {png}")
