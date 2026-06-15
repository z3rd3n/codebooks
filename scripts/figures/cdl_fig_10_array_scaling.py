"""CDL (Sionna 38.901) fig_10 -- scaling with the antenna array (P = 8..32).

Runs scripts/figures/fig_10_array_scaling.py on 3GPP CDL channels (default CDL-C), one
CDL panel array per (N1,N2) geometry, into results/sionna_cdl_gallery/.
See the matching .md for the analysis.

Run: .venv/bin/python scripts/figures/cdl_fig_10_array_scaling.py [--fast --drops N --model C]
"""

import fig_10_array_scaling as orig

from nr_csi.figtools import cdllib

if __name__ == "__main__":
    cdllib.run_original(orig)
