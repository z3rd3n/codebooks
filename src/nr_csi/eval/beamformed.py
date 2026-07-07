"""Beamformed CSI-RS adapter for port-selection codebooks.

Port-selection codebooks (R15 Type II PS, eType II PS, FeType II R17,
predicted PS R18) are specified for *beamformed* CSI-RS: the gNB precodes
each CSI-RS port with a spatial beam (derived from SRS or wideband
statistics) and the UE selects among those ports.  Evaluating them on raw
antenna-element channels -- where "port selection" degenerates to picking
bare array elements with SGCS ~ 1/P -- says nothing about the codebook.

``BeamformedPortsScheme`` restores the intended physics while keeping the
comparison on the same field as spatial-DFT codebooks:

* gNB side: wideband per-polarization covariance of the measured channel
  (a genie SRS estimate), common top-``n_beams_per_pol`` eigenbeams applied
  to both polarizations -- the same beams-per-pol structure as W1.
* UE side: the wrapped codebook sees the effective channel ``H @ B`` with
  ``2 * n_beams_per_pol`` ports and reports its normal PMI.
* Scoring: ``precoder`` maps the reported precoder back to physical
  antenna ports (``W_phys = B @ W_eff``), so SGCS/SE are measured against
  the *raw* channel, directly comparable with non-PS codebooks.

``B`` has orthonormal columns (eigenvectors, block-diagonal per pol), so
column norms -- and the tr(W^H W) = 1 power convention -- are preserved
exactly.  The beams are gNB-side and cost no feedback bits; they travel
inside the returned ``BeamformedPMI`` only so that ``precoder`` stays a
pure function of the PMI.

The polarization split assumes the framework's single-panel dual-pol port
order (first P/2 ports = pol 0, last P/2 = pol 1); multi-panel (Ng > 1)
channels are panel-major and are not supported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..codebooks.base import CodebookScheme


@dataclass
class BeamformedPMI:
    """Wrapped PMI: the inner codebook's report plus the gNB-side beams."""

    inner: Any
    B: np.ndarray  # (P_raw, P_eff), orthonormal columns; NOT fed back


class BeamformedPortsScheme(CodebookScheme):
    """Wrap a port-selection ``CodebookScheme`` behind eigen-beamformed ports.

    ``inner`` must be configured for ``2 * n_beams_per_pol`` CSI-RS ports;
    ``n_raw_ports`` is the physical antenna-port count of the channel this
    adapter will be evaluated on.
    """

    def __init__(
        self,
        inner: CodebookScheme,
        n_raw_ports: int,
        n_beams_per_pol: int,
    ) -> None:
        if n_raw_ports % 2 != 0 or n_raw_ports < 2:
            raise ValueError(f"n_raw_ports must be a positive even number, got {n_raw_ports}")
        if not 1 <= n_beams_per_pol <= n_raw_ports // 2:
            raise ValueError(
                f"n_beams_per_pol must be in [1, {n_raw_ports // 2}], got {n_beams_per_pol}"
            )
        inner_p = getattr(getattr(inner, "antenna", None), "P", None)
        if inner_p is not None and inner_p != 2 * n_beams_per_pol:
            raise ValueError(
                f"inner scheme is configured for {inner_p} CSI-RS ports but the "
                f"adapter produces 2*{n_beams_per_pol} = {2 * n_beams_per_pol}"
            )
        self.inner = inner
        self.n_raw_ports = int(n_raw_ports)
        self.n_beams_per_pol = int(n_beams_per_pol)
        self.name = f"{inner.name} @ {n_beams_per_pol}-eigenbeam CSI-RS"

    # ------------------------------------------------------------- gNB beams

    def beams(self, H: np.ndarray) -> np.ndarray:
        """Wideband eigenbeam matrix B (P_raw, 2*Kb) from measured channel H.

        Per-pol covariances are summed (common beams for both pols, as in
        W1); columns are orthonormal by construction, block-diagonal per pol.
        """
        H = np.asarray(H)
        P = H.shape[-1]
        if P != self.n_raw_ports:
            raise ValueError(f"channel has {P} ports, adapter expects {self.n_raw_ports}")
        half = P // 2
        kb = self.n_beams_per_pol
        X = H.reshape(-1, P)
        R = X[:, :half].conj().T @ X[:, :half] + X[:, half:].conj().T @ X[:, half:]
        _, vecs = np.linalg.eigh(R)  # ascending eigenvalues
        Bc = vecs[:, ::-1][:, :kb]  # top-kb eigenbeams, orthonormal
        B = np.zeros((P, 2 * kb), dtype=complex)
        B[:half, :kb] = Bc
        B[half:, kb:] = Bc
        return B

    # ---------------------------------------------------------- scheme API

    def select(self, H: np.ndarray, rank: int = 1) -> BeamformedPMI:
        B = self.beams(H)
        return BeamformedPMI(inner=self.inner.select(np.asarray(H) @ B, rank=rank), B=B)

    def precoder(self, pmi: BeamformedPMI) -> np.ndarray:
        W_eff = self.inner.precoder(pmi.inner)  # (S, N3, P_eff, v)
        return pmi.B @ W_eff  # (S, N3, P_raw, v); B^H B = I keeps column norms

    def overhead_bits(self, pmi: BeamformedPMI) -> dict[str, int]:
        return self.inner.overhead_bits(pmi.inner)

    def total_overhead_bits(self, pmi: BeamformedPMI) -> int:
        return self.inner.total_overhead_bits(pmi.inner)

    def __getattr__(self, item: str):
        # transparent passthrough (N4, L, antenna, ...) for harness/runner
        # introspection; __getattr__ fires only when normal lookup fails
        if item.startswith("__") or "inner" not in self.__dict__:
            raise AttributeError(item)
        return getattr(self.inner, item)
