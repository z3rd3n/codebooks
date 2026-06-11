"""Demonstration: comparing 3GPP codebooks against a custom (ML-style) scheme.

The ``OracleSVDScheme`` stands in for a learned CSI feedback model: it
implements the same ``CodebookScheme`` interface (select/precoder/
overhead_bits), here simply quantizing the dominant eigenvector with a fixed
bit budget.  Replace it with your autoencoder and the comparison pipeline
stays identical.

Note on R17: it is a *port-selection* codebook, designed for beam-domain
channels (the gNB applies a full-connect PEB to the CSI-RS first).  Running
it directly on an antenna-domain channel, as here, understates it -- shown
anyway to illustrate the applicability boundary the paper describes.

Run: .venv/bin/python scripts/compare_schemes.py
"""

import numpy as np

from nr_csi.baselines import eigen_precoder
from nr_csi.channel import Ray, SyntheticRayChannel
from nr_csi.codebooks import (
    CodebookScheme,
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    Type1Codebook,
)
from nr_csi.config import AntennaConfig
from nr_csi.eval import evaluate


class OracleSVDScheme(CodebookScheme):
    """Eigenvector feedback with naive uniform quantization (ML stand-in)."""

    name = "oracle-SVD 8-bit"

    def __init__(self, bits_per_coeff: int = 8):
        self.bits = bits_per_coeff

    def select(self, H, rank=1):
        W = eigen_precoder(H[-1], rank=rank)  # (N3, P, v)
        step = 2.0 / (2**(self.bits // 2))
        return np.round(W / step) * step  # "PMI" = quantized precoder

    def precoder(self, pmi):
        from nr_csi.codebooks.base import normalize_columns

        return normalize_columns(pmi)[None] / np.sqrt(pmi.shape[-1])

    def overhead_bits(self, pmi):
        return {"latent": pmi.size * self.bits}


class RandomRayChannel(SyntheticRayChannel):
    """Fresh random sparse rays per drop (the deterministic test channel
    re-randomized so Monte-Carlo evaluation is meaningful)."""

    def __init__(self, antenna, N3, n_rx, n_paths=4):
        super().__init__(antenna, [], N3=N3, n_rx=n_rx)
        self.n_paths = n_paths

    def generate(self, n_slots=1, rng=None):
        rng = rng or np.random.default_rng()
        G1, G2 = self.antenna.n_beams
        self.rays = [
            Ray(
                gain=(rng.standard_normal() + 1j * rng.standard_normal())
                / np.sqrt(2 * self.n_paths),
                m1=rng.uniform(0, G1),
                m2=rng.uniform(0, G2),
                delay=rng.uniform(0, 3),
                pol_phase=rng.uniform(0, 2 * np.pi),
                a_rx=(rng.standard_normal(self.n_rx) + 1j * rng.standard_normal(self.n_rx))
                / np.sqrt(2),
            )
            for _ in range(self.n_paths)
        ]
        return super().generate(n_slots, rng)


def main() -> None:
    ant = AntennaConfig.standard(4, 2)
    N3 = 8
    chan = RandomRayChannel(ant, N3=N3, n_rx=2)
    schemes = [
        Type1Codebook(ant, N3=N3),
        R15Type2Codebook(ant, N3=N3, L=4),
        R16Type2Codebook(ant, N3=N3, param_combination=4),
        R17Type2Codebook(ant, N3=N3, param_combination=5),
        OracleSVDScheme(),
    ]
    print(f"{'scheme':<22} {'SE@0dB':>7} {'SE@10dB':>8} {'SE@20dB':>8} {'SGCS':>6} {'bits':>7}")
    for scheme in schemes:
        res = evaluate(scheme, chan, snr_db=[0, 10, 20], rank=1, n_drops=50,
                       rng=np.random.default_rng(7))
        print(f"{res.scheme:<22} {res.se[0]:>7.2f} {res.se[1]:>8.2f} "
              f"{res.se[2]:>8.2f} {res.sgcs:>6.3f} {res.overhead_bits:>7.0f}")


if __name__ == "__main__":
    main()
