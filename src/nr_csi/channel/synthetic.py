"""Deterministic ray-based synthetic channel for unit tests.

Each ray contributes a rank-one term

    H_ray[s, t] = gain * e^{j 2 pi t d / N3} * e^{j 2 pi s nu / N4_ref}
                  * a_rx  (v_tx)^H

where (m1, m2) place the transmit beam on the (possibly oversampled) DFT
grid of the antenna config, ``d`` is the delay in DFT tap units of N3 and
``nu`` the Doppler in DFT shift units of ``doppler_period``.  Because the
transmit response enters conjugated, a codebook beam that matches (m1, m2)
is exactly the optimal precoder -- which makes ground-truth assertions in
tests trivial (on-grid rays => exact PMI recovery).

Dual polarization: the ray illuminates both polarizations of the dual-pol
array with co-phase ``pol_phase`` (v_dual = [v; e^{j*pol_phase} v]), matching
the Type I model H = [H1, H2] with H2 = e^{-j*pol_phase} H1.

Multi-panel (Ng > 1): the same physical ray direction ``v`` illuminates every
panel (co-located panels, common AoD/AoA), but each panel p picks up an
independent wideband co-phase ``panel_phase[p]`` -- matching the multi-panel
codebooks (``Type1MultiPanelCodebook``, ``RefinedType1MultiPanelCodebook``),
which treat inter-panel phase as a free per-panel parameter (i14) rather than
a geometrically predictable DFT-grid extension. Ports are ordered panel-major
then polarization-major, i.e. port = panel * 2*N1*N2 + pol * N1*N2 + (n1,n2),
matching the ``2*Ng*N1*N2``-length concatenation those codebooks build.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AntennaConfig
from .base import ChannelSource


@dataclass
class Ray:
    gain: complex
    m1: float  # horizontal index on the oversampled beam grid (float = off-grid)
    m2: float = 0.0
    delay: float = 0.0  # delay tap, in units of the N3-point DFT grid
    doppler: float = 0.0  # Doppler shift, in units of the doppler_period-point DFT grid
    pol_phase: float = 0.0  # co-phase between polarizations
    a_rx: np.ndarray | None = None  # receive signature (defaults to all-ones)
    panel_phase: np.ndarray | None = None  # per-panel co-phase, length Ng (None = single panel / all zeros)


class SyntheticRayChannel(ChannelSource):
    def __init__(
        self,
        antenna: AntennaConfig,
        rays: list[Ray],
        N3: int = 1,
        n_rx: int = 1,
        doppler_period: int = 1,
    ) -> None:
        self.antenna = antenna
        self.rays = rays
        self.N3 = N3
        self.n_rx = n_rx
        self.n_ports = antenna.P
        self.doppler_period = doppler_period

    def _beam(self, m1: float, m2: float) -> np.ndarray:
        a = self.antenna
        k1, k2 = np.arange(a.N1), np.arange(a.N2)
        ah = np.exp(2j * np.pi * m1 * k1 / (a.O1 * a.N1))
        av = np.exp(2j * np.pi * m2 * k2 / (a.O2 * a.N2))
        return np.kron(ah, av)

    def _port_vector(self, ray: Ray) -> np.ndarray:
        Ng = self.antenna.Ng
        v = self._beam(ray.m1, ray.m2)
        pol = np.array([1.0, np.exp(1j * ray.pol_phase)])
        if ray.panel_phase is None:
            panel = np.ones(Ng, dtype=complex)
        else:
            panel_phase = np.asarray(ray.panel_phase, dtype=float)
            if panel_phase.shape != (Ng,):
                raise ValueError(f"panel_phase must have {Ng} entries, got {panel_phase.shape}")
            panel = np.exp(1j * panel_phase)
        return np.einsum("g,c,n->gcn", panel, pol, v).reshape(-1)

    def generate(self, n_slots: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        H = np.zeros((n_slots, self.N3, self.n_rx, self.n_ports), dtype=complex)
        t = np.arange(self.N3)
        s = np.arange(n_slots)
        for ray in self.rays:
            v_full = self._port_vector(ray)
            a_rx = np.ones(self.n_rx) if ray.a_rx is None else np.asarray(ray.a_rx)
            f_freq = np.exp(2j * np.pi * t * ray.delay / self.N3)  # (N3,)
            f_time = np.exp(2j * np.pi * s * ray.doppler / self.doppler_period)  # (n_slots,)
            H += ray.gain * np.einsum("s,t,r,p->strp", f_time, f_freq, a_rx, v_full.conj())
        return H


class RandomRayChannel(SyntheticRayChannel):
    """Fresh random sparse rays per drop: the deterministic test channel
    re-randomized so Monte-Carlo evaluation is meaningful.

    ``max_delay``/``max_doppler`` bound the uniform draws (in DFT tap/shift
    units); rays are off-grid in angle, delay, and Doppler in general.
    """

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int = 1,
        n_rx: int = 1,
        n_paths: int = 4,
        max_delay: float = 3.0,
        max_doppler: float = 0.0,
        doppler_period: int = 1,
    ) -> None:
        super().__init__(antenna, [], N3=N3, n_rx=n_rx, doppler_period=doppler_period)
        self.n_paths = n_paths
        self.max_delay = max_delay
        self.max_doppler = max_doppler

    def generate(self, n_slots: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        G1, G2 = self.antenna.n_beams
        Ng = self.antenna.Ng
        self.rays = [
            Ray(
                gain=(rng.standard_normal() + 1j * rng.standard_normal())
                / np.sqrt(2 * self.n_paths),
                m1=rng.uniform(0, G1),
                m2=rng.uniform(0, G2),
                delay=rng.uniform(0, self.max_delay),
                doppler=rng.uniform(0, self.max_doppler),
                pol_phase=rng.uniform(0, 2 * np.pi),
                a_rx=(rng.standard_normal(self.n_rx) + 1j * rng.standard_normal(self.n_rx))
                / np.sqrt(2),
                panel_phase=(rng.uniform(0, 2 * np.pi, size=Ng) if Ng > 1 else None),
            )
            for _ in range(self.n_paths)
        ]
        return super().generate(n_slots, rng)
