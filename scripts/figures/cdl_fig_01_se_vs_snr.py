"""CDL (Sionna 38.901) fig_01 -- spectral efficiency vs SNR, all families.

Runs scripts/fig_01_se_vs_snr.py on a 3GPP CDL channel (default CDL-C, 3 km/h)
into results/sionna_cdl_gallery/.  See the matching .md for the analysis.

Run: .venv/bin/python scripts/cdl_fig_01_se_vs_snr.py [--fast --drops N --seed S --model C]
"""

import cdllib
import fig_01_se_vs_snr as orig

if __name__ == "__main__":
    cdllib.run_original(orig)
