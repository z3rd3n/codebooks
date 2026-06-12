"""3GPP 38.901 CDL channels via Sionna (optional dependency).

Maps a Sionna CDL downlink channel onto the framework's port/subband grid:

    H[slot, t, rx, port],  port = pol * N1*N2 + n1 * N2 + n2

with n1 horizontal (columns), n2 vertical (rows) -- the ordering assumed by
the spatial DFT bases in :mod:`nr_csi.utils.dft`.  The permutation from
Sionna's internal antenna numbering is derived from the panel's antenna
positions, so it does not depend on Sionna's enumeration conventions.

Time axis: ``generate(n_slots)`` samples one channel snapshot per PMI slot
interval (duration ``interval_duration`` seconds), which is the granularity
the R18 Doppler codebook operates on.
"""

from __future__ import annotations

import numpy as np

from ..config import AntennaConfig, n3_for_bwp
from .base import ChannelSource

RE_PER_RB = 12  # subcarriers per resource block


def _import_sionna():
    try:  # Sionna >= 1.0
        from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
        from sionna.phy.channel.tr38901 import CDL, Antenna, PanelArray

        return CDL, Antenna, PanelArray, cir_to_ofdm_channel, subcarrier_frequencies
    except ImportError:  # Sionna 0.x
        from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies
        from sionna.channel.tr38901 import CDL, Antenna, PanelArray

        return CDL, Antenna, PanelArray, cir_to_ofdm_channel, subcarrier_frequencies


class SionnaCDLChannel(ChannelSource):
    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int | None = None,
        model: str = "A",
        carrier_frequency: float = 3.5e9,
        delay_spread: float = 100e-9,
        n_rx: int = 2,
        ue_speed_kmh: float = 3.0,
        fft_size: int = 256,
        subcarrier_spacing: float = 30e3,
        interval_duration: float = 0.5e-3,
        n_rb: int | None = None,
        subband_size: int | None = None,
        R: int = 1,
    ) -> None:
        """Either give ``N3`` directly (the grid is split into N3 equal
        chunks), or give the BWP size ``n_rb`` and a ``subband_size`` from
        Table tabCSS: then N3 = ceil(n_rb / subband_size) * R and each PMI
        frequency unit averages the subcarriers of its RB-aligned subband
        (the last one may be short)."""
        CDL, Antenna, PanelArray, cir_to_ofdm, subc_freq = _import_sionna()
        self._cir_to_ofdm = cir_to_ofdm
        self.antenna = antenna
        if n_rb is not None:
            if subband_size is None:
                raise ValueError("subband_size is required together with n_rb")
            if n_rb * RE_PER_RB > fft_size:
                raise ValueError(
                    f"{n_rb} RB = {n_rb * RE_PER_RB} subcarriers exceed fft_size={fft_size}"
                )
            self.N3 = n3_for_bwp(n_rb, subband_size, R)
            n_sb = self.N3 // R
            # subcarrier slice per PMI unit; R units share one CQI subband
            edges = [min(i * subband_size * RE_PER_RB, n_rb * RE_PER_RB)
                     for i in range(n_sb + 1)]
            self._unit_slices = [
                slice(edges[i], edges[i + 1]) for i in range(n_sb) for _ in range(R)
            ]
        elif N3 is not None:
            self.N3 = N3
            sc = fft_size // N3
            self._unit_slices = [slice(i * sc, (i + 1) * sc) for i in range(N3)]
        else:
            raise ValueError("give either N3 or (n_rb, subband_size)")
        self.n_rx = n_rx
        self.n_ports = antenna.P
        self.fft_size = fft_size
        self.interval_duration = interval_duration
        self.frequencies = subc_freq(fft_size, subcarrier_spacing)

        bs_array = PanelArray(
            num_rows_per_panel=antenna.N2,
            num_cols_per_panel=antenna.N1,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency,
        )
        ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=max(n_rx // 2, 1),
            polarization="dual" if n_rx > 1 else "single",
            polarization_type="cross" if n_rx > 1 else "V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
        )
        speed = ue_speed_kmh / 3.6
        self._cdl = CDL(
            model=model,
            delay_spread=delay_spread,
            carrier_frequency=carrier_frequency,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            min_speed=speed,
            max_speed=speed,
        )
        self.bs_array = bs_array
        self._port_perm = self._port_permutation(bs_array)

    def _port_permutation(self, bs_array) -> np.ndarray:
        """Permutation p such that H[..., sionna_ant[p[k]]] is our port k.

        Our port k = pol * N1*N2 + n1 * N2 + n2; antennas of one polarization
        are ordered by horizontal position (y) first, vertical (z) fastest.
        """
        pos = np.asarray(bs_array.ant_pos)  # (num_ant, 3), columns (x, y, z)
        ind_p1 = np.asarray(bs_array.ant_ind_pol1).ravel()
        ind_p2 = np.asarray(bs_array.ant_ind_pol2).ravel()
        perm = []
        for ind in (ind_p1, ind_p2):
            p = pos[ind]
            # sort by horizontal (y) then vertical (z): vertical index fastest
            order = np.lexsort((p[:, 2], p[:, 1]))
            perm.extend(ind[order].tolist())
        return np.asarray(perm)

    def generate(self, n_slots: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Note: randomness is driven by Sionna/TensorFlow, not by ``rng``."""
        a, tau = self._cdl(
            batch_size=1,
            num_time_steps=n_slots,
            sampling_frequency=1.0 / self.interval_duration,
        )
        h = self._cir_to_ofdm(self.frequencies, a, tau, normalize=True)
        # (batch, num_rx, num_rx_ant, num_tx, num_tx_ant, time, subcarriers)
        h = np.asarray(h)[0, 0, :, 0, :, :, :]  # (rx_ant, tx_ant, time, sc)
        h = h[:, self._port_perm]  # our port ordering
        # average the subcarriers of each PMI frequency unit
        units = [h[..., s].mean(axis=-1) for s in self._unit_slices]
        h = np.stack(units, axis=-1)  # (rx, port, time, N3)
        return h.transpose(2, 3, 0, 1)  # (slot, t, rx, port)
