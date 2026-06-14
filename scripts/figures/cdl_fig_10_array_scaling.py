"""CDL (Sionna 38.901) fig_10 -- scaling with the antenna array (P = 8..32).

Runs scripts/fig_10_array_scaling.py on 3GPP CDL channels (default CDL-C), one
CDL panel array per (N1,N2) geometry, into results/sionna_cdl_gallery/.
See the matching .md for the analysis.

Run: .venv/bin/python scripts/cdl_fig_10_array_scaling.py [--fast --drops N --model C]
"""

import cdllib
import fig_10_array_scaling as orig

if __name__ == "__main__":
    cdllib.run_original(orig)
