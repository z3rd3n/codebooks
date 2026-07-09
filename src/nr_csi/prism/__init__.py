"""PRISM: one-sided CSI feedback via a published mixture of KLT sketches.

The successor to :mod:`nr_csi.ml` (GLIMPSE).  GLIMPSE's single KLT basis is
statistically matched to ONE channel regime and collapses out-of-distribution
(near-LoS CDL-D/E).  PRISM publishes a small dictionary of K KLT bases fitted
by a Lloyd-style K-subspaces alternation on a training mix that covers every
deployment regime; the UE picks, per report, the basis whose m-row sketch
captures the most energy and signals the choice with ceil(log2 K) bits.

Everything stays one-sided and parameter-free at the UE (K small matrix
multiplies + an argmax), the gNB decodes by least squares (fully linear, no
neural network anywhere), and K = 1 on pooled data degenerates exactly to a
broad-trained GLIMPSE -- the ablation that exposes the averaging penalty the
mixture removes.
"""

from .mixture import PrismCodec, fit_mixture
from .scheme import PrismScheme

__all__ = ["PrismCodec", "PrismScheme", "fit_mixture"]
