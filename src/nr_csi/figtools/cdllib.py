"""CDL companion to ``figlib``: run the existing figure scripts on Sionna CDL.

The 12 ``fig_*.py`` scripts are written against ``figlib.default_channel`` (a
synthetic ``RandomRayChannel``) and ``figlib.run_eval``.  ``run_original`` swaps
those names in a figure module for Sionna 3GPP TR 38.901 CDL equivalents and
runs the script unchanged into ``results/sionna_cdl_gallery/`` -- so each
``cdl_fig_NN.py`` stays a thin, standalone wrapper while the (battle-tested)
plotting logic lives in one place.

Why a replay channel?  ``SionnaCDLChannel.generate`` is TensorFlow-driven and
ignores the NumPy ``rng``, so the figures' paired-seed comparisons would break.
``CDLReplay`` seeds once, caches drops as they are drawn, and ``reset()`` rewinds
to drop 0 -- the reset-aware ``cdl_run_eval`` calls it before every scheme, so
all schemes (and all channel-domain views) see the *same* CDL drops, paired and
reproducible.

Channel knobs that have no CDL analog are mapped honestly:
* ``n_paths`` / ``max_delay`` (random-ray richness)  -> ignored; CDL's cluster
  structure is fixed by the model, so e.g. fig_08's sparsity sweep is flat.
* ``max_doppler`` > 0 (mobility)  -> a faster UE speed + longer slot interval.

Environment overrides: ``NRCSI_CDL_MODEL`` (default C), ``NRCSI_CDL_SPEED``,
``NRCSI_CDL_SPEED_FAST``, ``NRCSI_CDL_DS`` (delay spread, s), ``NRCSI_CDL_SEED``,
``NRCSI_CDL_SLOTS`` (time steps per cached drop).
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF before any import

import sys  # noqa: E402

import figlib  # noqa: E402
import numpy as np  # noqa: E402

from nr_csi.channel.base import ChannelSource  # noqa: E402
from nr_csi.eval import evaluate  # noqa: E402

GALLERY = figlib.RESULTS / "sionna_cdl_gallery"

MODEL = os.environ.get("NRCSI_CDL_MODEL", "C")
SPEED = float(os.environ.get("NRCSI_CDL_SPEED", "3.0"))  # km/h, static figures
SPEED_FAST = float(os.environ.get("NRCSI_CDL_SPEED_FAST", "30.0"))  # km/h, mobility
DS = float(os.environ.get("NRCSI_CDL_DS", "100e-9"))  # delay spread (s)
INTERVAL = float(os.environ.get("NRCSI_CDL_INTERVAL", "0.5e-3"))  # slot interval (s)
INTERVAL_FAST = float(os.environ.get("NRCSI_CDL_INTERVAL_FAST", "2.0e-3"))
SEED = int(os.environ.get("NRCSI_CDL_SEED", "0"))
SLOTS = int(os.environ.get("NRCSI_CDL_SLOTS", "4"))  # >= max n_slots any script needs


class CDLReplay(ChannelSource):
    """A Sionna CDL channel with a replayable, reproducible drop sequence.

    Drops are generated lazily (each with ``slots`` time steps so any
    ``[:n_slots]`` slice is valid) and cached; ``reset()`` rewinds to drop 0 so
    the next consumer replays the same sequence -- this is what keeps the
    figure comparisons paired and reproducible despite Sionna's TF-driven RNG.
    """

    def __init__(self, antenna, n3, *, n_rx, speed, delay_spread, interval, model, seed, slots):
        self.antenna = antenna
        self.n3 = n3
        self.n_rx = n_rx
        self.speed = speed
        self.delay_spread = delay_spread
        self.interval = interval
        self.model = model
        self.seed = seed
        self.slots = slots
        self._cdl = None
        self._bank: list[np.ndarray] = []
        self._i = 0

    def _ensure(self):
        if self._cdl is not None:
            return
        import tensorflow as tf

        from nr_csi.channel.sionna_adapter import SionnaCDLChannel

        tf.random.set_seed(self.seed)
        try:
            from sionna.phy import config as sionna_config

            sionna_config.seed = self.seed
        except Exception:  # pragma: no cover - older/newer Sionna layouts
            pass
        self._cdl = SionnaCDLChannel(
            self.antenna, N3=self.n3, model=self.model, n_rx=self.n_rx,
            ue_speed_kmh=self.speed, delay_spread=self.delay_spread,
            interval_duration=self.interval,
        )

    def generate(self, n_slots: int = 1, rng=None) -> np.ndarray:
        self._ensure()
        while self._i >= len(self._bank):
            self._bank.append(self._cdl.generate(n_slots=self.slots))
        H = self._bank[self._i]
        self._i += 1
        if H.shape[0] < n_slots:
            raise ValueError(
                f"CDL drop has {H.shape[0]} slots, need {n_slots}; raise NRCSI_CDL_SLOTS"
            )
        return H[:n_slots]

    def reset(self):
        self._i = 0


class CDLBeamDomainChannel(figlib.BeamDomainChannel):
    """``figlib.BeamDomainChannel`` that forwards ``reset()`` to its inner CDL
    channel (so beam-domain views replay the same drops as antenna-domain)."""

    def reset(self):
        if hasattr(self.inner, "reset"):
            self.inner.reset()


_CACHE: dict = {}


def cdl_channel(ant=figlib.ANT, n3=figlib.N3, **kwargs) -> CDLReplay:
    """Drop-in replacement for ``figlib.default_channel`` backed by Sionna CDL.

    ``n_paths`` / ``max_delay`` are ignored (no CDL analog); ``max_doppler`` > 0
    selects the mobility regime (faster UE, longer interval).  Channels are
    cached by configuration so repeated sweep points reuse one drop bank.
    """
    mobile = float(kwargs.get("max_doppler", 0.0) or 0.0) > 0.0
    speed = SPEED_FAST if mobile else SPEED
    interval = INTERVAL_FAST if mobile else INTERVAL
    n_rx = int(kwargs.get("n_rx", 2))
    key = (MODEL, ant, n3, n_rx, speed, DS, interval, SLOTS, SEED)
    chan = _CACHE.get(key)
    if chan is None:
        chan = CDLReplay(ant, n3, n_rx=n_rx, speed=speed, delay_spread=DS,
                         interval=interval, model=MODEL, seed=SEED, slots=SLOTS)
        _CACHE[key] = chan
    chan.reset()
    return chan


def cdl_run_eval(scheme, channel, domain: str = "antenna", *, seed: int = 0,
                 antenna=figlib.ANT, **kwargs):
    """Reset-aware ``figlib.run_eval``: rewinds the CDL bank so every scheme and
    every domain view is scored on the same drops."""
    if hasattr(channel, "reset"):
        channel.reset()
    if domain == "beam":
        channel = CDLBeamDomainChannel(channel, antenna)
    kwargs.setdefault("n_slots", getattr(scheme, "N4", 1))
    return evaluate(scheme, channel, rng=np.random.default_rng(seed), **kwargs)


def _set_model(model: str) -> None:
    global MODEL
    MODEL = model
    _CACHE.clear()


def run_original(mod, *, default_model: str | None = None) -> None:
    """Run a ``fig_*.py`` module's ``main()`` on CDL into the gallery folder.

    Swaps the channel/eval names the module imported from ``figlib`` for their
    CDL equivalents, injects ``--out <gallery>``, and forwards the rest of the
    CLI.  A leading ``--model X`` (not a figure CLI arg) is consumed here.
    """
    argv = list(sys.argv[1:])
    if "--model" in argv:
        k = argv.index("--model")
        _set_model(argv[k + 1])
        del argv[k:k + 2]
    elif default_model is not None:
        _set_model(default_model)

    GALLERY.mkdir(parents=True, exist_ok=True)
    for name, repl in (("default_channel", cdl_channel),
                       ("run_eval", cdl_run_eval),
                       ("BeamDomainChannel", CDLBeamDomainChannel)):
        if hasattr(mod, name):
            setattr(mod, name, repl)

    old_argv = sys.argv
    sys.argv = [mod.__name__, "--out", str(GALLERY), *argv]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
