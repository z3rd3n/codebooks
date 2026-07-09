"""PRISM as a :class:`~nr_csi.codebooks.base.CodebookScheme`.

``select`` (UE): angle-delay eigen targets -> basis choice by max captured
energy -> standardized, reverse-water-filled Lloyd-Max quantization under the
chosen basis's published stds.  Zero learned parameters; K small matrix
multiplies and an argmax beyond what GLIMPSE already does.

``precoder`` (gNB): least-squares inverse under the signalled basis.  Fully
linear -- PRISM deliberately has no neural network on either side, resting on
GLIMPSE's finding that LS matches a trained decoder in-distribution once the
sketch basis is statistically matched.

Report layout (per rank-``v`` report): ``ceil(log2 K)`` index bits (shared by
all layers) followed by ``v`` per-layer streams of exactly ``2 m B`` bits.
``len(pack(report)) == overhead bits`` by construction, as everywhere in this
framework.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..codebooks._spatial import aligned_eigen_targets
from ..codebooks.base import CodebookScheme
from ..config import AntennaConfig
from ..ml.quantizer import allocate_bits, dequantize_alloc, quantize_alloc
from .mixture import PrismCodec


class PrismScheme(CodebookScheme):
    """One-sided mixture-of-KLT-sketches CSI feedback (fully linear)."""

    def __init__(
        self,
        antenna: AntennaConfig,
        N3: int,
        codec: PrismCodec,
        m: int = 16,
        bits: int = 3,
        allocation: str = "waterfill",
        name: str | None = None,
    ) -> None:
        if not 1 <= m <= codec.m_max:
            raise ValueError(f"m must be in [1, {codec.m_max}], got {m}")
        if allocation not in ("waterfill", "uniform"):
            raise ValueError("allocation must be 'waterfill' or 'uniform'")
        self.antenna = antenna
        self.N3 = N3
        self.codec = codec
        self.m = m
        self.bits = bits
        self.allocation = allocation
        self.total_bits = 2 * m * bits  # per layer, excluding the shared index
        # one published allocation per basis (deterministic in the published
        # sigmas -- computed identically at UE and gNB, no side information)
        self.bits_vecs = [
            allocate_bits(codec.sigmas[k], m, self.total_bits)
            if allocation == "waterfill" else np.full(2 * m, bits, dtype=np.uint8)
            for k in range(codec.n_components)
        ]
        self.name = name or f"PRISM K{codec.n_components} m{m} B{bits}"

    # ------------------------------------------------------------------- UE
    def select(self, H: np.ndarray, rank: int = 1) -> dict[str, Any]:
        H = np.asarray(H)[-1]
        if H.shape[0] != self.N3:
            raise ValueError(
                f"channel has {H.shape[0]} frequency units, expected {self.N3}")
        V = aligned_eigen_targets(H, rank)  # (N3, P, v)
        g = self.codec.targets_to_vec(V)  # (v, D)
        k = self.codec.select_basis(g, self.m)  # max energy across layers
        y = self.codec.project(g, self.m, k)  # (v, m)
        u = self.codec.standardize(y, k)
        streams = [quantize_alloc(u[layer], self.bits_vecs[k])
                   for layer in range(u.shape[0])]
        u_hat = np.stack([dequantize_alloc(s, self.bits_vecs[k]) for s in streams])
        return {"u_hat": u_hat, "streams": streams, "basis": k, "m": self.m,
                "bits": self.bits, "rank": rank}

    # ------------------------------------------------------------------ gNB
    def precoder(self, pmi: dict[str, Any]) -> np.ndarray:
        k, m, rank = pmi["basis"], pmi["m"], pmi["rank"]
        y_hat = self.codec.destandardize(np.asarray(pmi["u_hat"]), k)
        g_hat = self.codec.adjoint(y_hat, m, k)  # least squares, (v, D)
        V = self.codec.vec_to_targets(g_hat)  # (N3, P, v)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / np.where(norms == 0, 1.0, norms)
        return (V / np.sqrt(rank))[None]  # (1, N3, P, v), tr(W^H W) = 1

    # ------------------------------------------------------------- overhead
    def overhead_bits(self, pmi: dict[str, Any]) -> dict[str, int]:
        return {"basis_index": self.codec.index_bits,
                "measurements": int(sum(len(s) for s in pmi["streams"]))}

    # ---------------------------------------------------------- serialization
    def pack(self, pmi: dict[str, Any]) -> str:
        """Feedback bitstream: index header then per-layer streams;
        ``len == total_overhead_bits`` by construction."""
        header = (format(pmi["basis"], f"0{self.codec.index_bits}b")
                  if self.codec.index_bits else "")
        return header + "".join(pmi["streams"])

    def unpack(self, bits: str, rank: int) -> dict[str, Any]:
        nb = self.codec.index_bits
        k = int(bits[:nb], 2) if nb else 0
        n = self.total_bits
        streams = [bits[nb + i * n : nb + (i + 1) * n] for i in range(rank)]
        u_hat = np.stack([dequantize_alloc(s, self.bits_vecs[k]) for s in streams])
        return {"u_hat": u_hat, "streams": streams, "basis": k, "m": self.m,
                "bits": self.bits, "rank": rank}

    def truncate(self, pmi: dict[str, Any], m: int) -> dict[str, Any]:
        """Rateless rate adaptation: re-form the report at ``m <= m0``
        measurements, keeping the signalled basis (its rows are
        variance-ordered, so the first ``m`` coordinates are the most
        informative ones under that basis)."""
        m0 = int(pmi["m"])
        if not 1 <= m <= m0:
            raise ValueError(f"m must be in [1, {m0}], got {m}")
        k = pmi["basis"]
        sub = PrismScheme(self.antenna, self.N3, self.codec, m=m, bits=self.bits,
                          allocation=self.allocation)
        u_top = np.asarray(pmi["u_hat"])[:, :m]
        streams = [quantize_alloc(u_top[layer], sub.bits_vecs[k])
                   for layer in range(u_top.shape[0])]
        u_hat = np.stack([dequantize_alloc(s, sub.bits_vecs[k]) for s in streams])
        return {"u_hat": u_hat, "streams": streams, "basis": k, "m": m,
                "bits": self.bits, "rank": int(pmi["rank"])}
