"""CDL (Sionna 38.901) fig_02 -- rate-distortion plane (fidelity vs bits).

Runs scripts/figures/fig_02_rate_distortion.py on a 3GPP CDL channel (default CDL-C)
into results/sionna_cdl_gallery/.  See the matching .md for the analysis.

Run: .venv/bin/python scripts/figures/cdl_fig_02_rate_distortion.py [--fast --drops N --model C]
"""

import fig_02_rate_distortion as orig

from nr_csi.figtools import cdllib

if __name__ == "__main__":
    cdllib.run_original(orig)
