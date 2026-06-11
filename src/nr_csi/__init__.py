"""3GPP NR PMI codebook benchmark framework (Releases 15-18).

Implements the beamforming codebooks described in:
    "Precoding Matrix Indicator in the 5G NR Protocol: A Tutorial on 3GPP
    Beamforming Codebooks" (paper/main.tex), which follows TS 38.214 5.2.2.2.

Subpackages
-----------
utils       DFT bases, combinatorial index codecs, quantization tables
codebooks   Type I / Type II / eType II / FeType II / Doppler codebooks
channel     Synthetic and Sionna 38.901 channel sources
baselines   Full-CSI beamforming baselines (upper bounds)
metrics     Spectral efficiency, SGCS/NMSE, feedback overhead
eval        Evaluation harness for codebook vs. ML-scheme comparison
"""

__version__ = "0.1.0"
