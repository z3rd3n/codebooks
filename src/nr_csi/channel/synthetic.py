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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AntennaConfig
from ..utils import dft
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

    def generate(self, n_slots: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        H = np.zeros((n_slots, self.N3, self.n_rx, self.n_ports), dtype=complex)
        t = np.arange(self.N3)
        s = np.arange(n_slots)
        for ray in self.rays:
            v = self._beam(ray.m1, ray.m2)
            v_dual = np.concatenate([v, np.exp(1j * ray.pol_phase) * v])
            a_rx = np.ones(self.n_rx) if ray.a_rx is None else np.asarray(ray.a_rx)
            f_freq = np.exp(2j * np.pi * t * ray.delay / self.N3)  # (N3,)
            f_time = np.exp(2j * np.pi * s * ray.doppler / self.doppler_period)  # (n_slots,)
            H += ray.gain * np.einsum("s,t,r,p->strp", f_time, f_freq, a_rx, v_dual.conj())
        return H
