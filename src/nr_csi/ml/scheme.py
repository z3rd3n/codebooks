"""GLIMPSE as a :class:`~nr_csi.codebooks.base.CodebookScheme`.

``select`` (UE) is the fixed projection + quantization pipeline -- zero
learned parameters, one small matrix multiply beyond the eigen targets every
PMI scheme computes.  ``precoder`` (gNB) delegates reconstruction to a
pluggable decoder:

* :class:`LeastSquaresDecoder` -- minimum-norm inverse, no prior (floor);
* :class:`OMPDecoder` -- orthogonal matching pursuit, classical sparsity
  prior (what pre-DL compressed-sensing feedback would do);
* ``nr_csi.ml.decoder.KerasDecoder`` -- the trained network (gNB side).

The decoder choice never changes the report: all three consume the *same*
bits, which is the point of one-sided learning -- the network is a pure
gNB-side upgrade.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ..codebooks._spatial import aligned_eigen_targets
from ..codebooks.base import CodebookScheme
from ..config import AntennaConfig
from .projection import GlimpseCodec
from .quantizer import allocate_bits, dequantize_alloc, quantize_alloc


class GlimpseDecoder(Protocol):
    """Reconstructs angle-delay vectors from physical (destandardized)
    measurements ``y[..., m]`` -> ``g_hat[..., D]``."""

    def __call__(self, y: np.ndarray, m: int, bits: int = 3) -> np.ndarray:
        ...


class LeastSquaresDecoder:
    """Minimum-norm solution ``A_m^H y`` -- the prior-free floor."""

    name = "ls"

    def __init__(self, codec: GlimpseCodec) -> None:
        self.codec = codec

    def __call__(self, y: np.ndarray, m: int, bits: int = 3) -> np.ndarray:
        return self.codec.adjoint(y, m)


class OMPDecoder:
    """Orthogonal matching pursuit: the classical compressed-sensing baseline.

    Greedy support growth with an exact least-squares re-solve per step --
    the most robust of the classical greedy solvers on this ensemble (plain
    IHT and HTP stall in wrong-support fixed points near the phase
    transition).  Represents what pre-deep-learning CS feedback would do
    with the *same bits*: a generic sparsity prior instead of a learned
    channel prior.  ``s`` defaults to ``max(m // 3, 8)`` -- the usual
    m >= 3s greedy working regime, floored so tiny reports keep enough
    support to describe a dual-polarized path.
    """

    name = "omp"

    def __init__(self, codec: GlimpseCodec, sparsity: int | None = None) -> None:
        self.codec = codec
        self.sparsity = sparsity

    def __call__(self, y: np.ndarray, m: int, bits: int = 3) -> np.ndarray:
        s = min(self.sparsity or max(m // 3, 8), m)
        A = self.codec.A[:m]
        single = np.asarray(y).ndim == 1
        u = np.atleast_2d(y)
        out = np.zeros((u.shape[0], self.codec.D), complex)
        for i, ui in enumerate(u):
            residual = ui.copy()
            support: list[int] = []
            x_s = np.zeros(0, complex)
            for _ in range(s):
                corr = np.abs(residual @ A.conj())
                corr[support] = -np.inf
                support.append(int(np.argmax(corr)))
                x_s, *_ = np.linalg.lstsq(A[:, support], ui, rcond=None)
                residual = ui - A[:, support] @ x_s
            out[i, support] = x_s
        return out[0] if single else out


class GlimpseScheme(CodebookScheme):
    """One-sided learned CSI feedback with a codebook-compatible interface.

    Report: per layer, ``m`` complex KLT measurements standardized to unit
    variance and quantized under a total budget of ``2*m*bits`` bits.  With
    ``allocation="waterfill"`` (default) those bits are distributed across the
    ``2m`` real dimensions by reverse water-filling (optimal transform coding:
    high-variance coordinates get more bits); ``"uniform"`` gives every
    dimension ``bits`` bits (the ablation baseline).  Either way the report is
    ``rank * 2 * m * bits`` bits.  ``m`` is a *continuous* overhead knob (any
    value up to ``codec.m_max``), unlike the discrete codebook parameter
    ladders, and reports are prefix-decodable.
    """

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        decoder: GlimpseDecoder,
        m: int = 16,
        bits: int = 3,
        m_max: int = 48,
        seed: int = 0,
        codec: GlimpseCodec | None = None,
        allocation: str = "waterfill",
        name: str | None = None,
    ) -> None:
        self.antenna = antenna
        self.N3 = N3
        self.codec = codec or GlimpseCodec(antenna, N3, m_max=m_max, seed=seed)
        if not 1 <= m <= self.codec.m_max:
            raise ValueError(f"m must be in [1, {self.codec.m_max}], got {m}")
        if allocation not in ("waterfill", "uniform"):
            raise ValueError("allocation must be 'waterfill' or 'uniform'")
        self.m = m
        self.bits = bits
        self.allocation = allocation
        self.total_bits = 2 * m * bits  # per layer
        self.decoder = decoder
        # per-real-dim bit allocation (a published constant of sigma; identical
        # at UE and gNB, so no side information is signalled)
        self.bits_vec = (
            allocate_bits(self.codec._sigma, m, self.total_bits)
            if allocation == "waterfill"
            else np.full(2 * m, bits, dtype=np.uint8)
        )
        dec_name = getattr(decoder, "name", type(decoder).__name__)
        tag = "" if allocation == "waterfill" else "u"
        self.name = name or f"GLIMPSE {dec_name} m{m} B{bits}{tag}"

    # ------------------------------------------------------------------- UE
    def select(self, H: np.ndarray, rank: int = 1) -> dict[str, Any]:
        H = np.asarray(H)[-1]
        if H.shape[0] != self.N3:
            raise ValueError(f"channel has {H.shape[0]} frequency units, expected {self.N3}")
        V = aligned_eigen_targets(H, rank)  # (N3, P, v)
        g = self.codec.targets_to_vec(V)  # (v, D)
        y = self.codec.project(g, self.m)  # (v, m) physical measurements
        u = self.codec.standardize(y)  # unit-variance report
        streams = [quantize_alloc(u[layer], self.bits_vec) for layer in range(u.shape[0])]
        u_hat = np.stack([dequantize_alloc(s, self.bits_vec) for s in streams])  # (v, m)
        return {"u_hat": u_hat, "streams": streams, "m": self.m, "bits": self.bits,
                "rank": rank}

    # ------------------------------------------------------------------ gNB
    def precoder(self, pmi: dict[str, Any]) -> np.ndarray:
        m, rank = pmi["m"], pmi["rank"]
        y_hat = self.codec.destandardize(np.asarray(pmi["u_hat"]))  # (v, m) physical
        g_hat = self.decoder(y_hat, m, pmi["bits"])  # (v, D)
        V = self.codec.vec_to_targets(g_hat)  # (N3, P, v)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / np.where(norms == 0, 1.0, norms)
        return (V / np.sqrt(rank))[None]  # (1, N3, P, v), tr(W^H W) = 1

    # ------------------------------------------------------------- overhead
    def overhead_bits(self, pmi: dict[str, Any]) -> dict[str, int]:
        # the allocation sums to total_bits, so every layer's stream is exactly
        # total_bits long -- the report is rank * total_bits bits.
        return {"measurements": int(sum(len(s) for s in pmi["streams"]))}

    # ---------------------------------------------------------- serialization
    def pack(self, pmi: dict[str, Any]) -> str:
        """Actual feedback bitstream; ``len == total_overhead_bits`` by
        construction (the honesty convention of ``codebooks.serialize``)."""
        return "".join(pmi["streams"])

    def unpack(self, bits: str, rank: int) -> dict[str, Any]:
        n = self.total_bits
        streams = [bits[i * n : (i + 1) * n] for i in range(rank)]
        u_hat = np.stack([dequantize_alloc(s, self.bits_vec) for s in streams])
        return {"u_hat": u_hat, "streams": streams, "m": self.m, "bits": self.bits,
                "rank": rank}

    def truncate(self, pmi: dict[str, Any], m: int) -> dict[str, Any]:
        """Rateless rate adaptation: re-form the report at ``m <= m0`` KLT
        measurements.

        Because the KLT rows are variance-ordered, the first ``m`` coordinates
        are the most informative, so keeping them (and re-deriving the smaller
        budget's bit allocation) yields the report the UE would have sent at the
        reduced rate -- decodable by the *same* decoder, with graceful monotone
        fidelity loss.
        """
        m0 = int(pmi["m"])
        if not 1 <= m <= m0:
            raise ValueError(f"m must be in [1, {m0}], got {m}")
        u_top = np.asarray(pmi["u_hat"])[:, :m]  # top-m KLT coordinates
        sub = GlimpseScheme(self.antenna, self.N3, self.decoder, m=m, bits=self.bits,
                            codec=self.codec, allocation=self.allocation)
        streams = [quantize_alloc(u_top[layer], sub.bits_vec)
                   for layer in range(u_top.shape[0])]
        u_hat = np.stack([dequantize_alloc(s, sub.bits_vec) for s in streams])
        return {"u_hat": u_hat, "streams": streams, "m": m, "bits": self.bits,
                "rank": int(pmi["rank"])}
