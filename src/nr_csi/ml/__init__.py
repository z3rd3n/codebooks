"""GLIMPSE: one-sided learned CSI feedback (gNB-Learned Inversion of
Measurement-Projected Subband Eigenvectors).

The UE side is a *fixed* (spec-frozen) pipeline with zero learned parameters:
eigen targets -> fixed KLT (or random) projection -> fixed scalar quantizer.
The gNB side is a learned unrolled reconstruction network.  See
``docs/ml/glimpse.md`` for the method, design rationale, and results.

Only NumPy is required for the UE side and the classical decoders; the
learned decoder needs the optional TensorFlow extra (``pip install -e
".[sionna]"``) and is imported lazily.
"""

from .projection import (
    GlimpseCodec,
    encoder_flops,
    fit_klt,
    measurement_matrix,
    type2_select_flops,
)
from .quantizer import (
    dequantize,
    lloyd_max,
    pack_indices,
    quantize,
    unpack_indices,
)
from .scheme import GlimpseScheme, LeastSquaresDecoder, OMPDecoder

__all__ = [
    "GlimpseCodec",
    "GlimpseScheme",
    "OMPDecoder",
    "LeastSquaresDecoder",
    "dequantize",
    "encoder_flops",
    "fit_klt",
    "lloyd_max",
    "measurement_matrix",
    "pack_indices",
    "quantize",
    "type2_select_flops",
    "unpack_indices",
]
