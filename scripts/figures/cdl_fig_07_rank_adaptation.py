"""CDL (Sionna 38.901) fig_07 -- rank (RI) adaptation: layers vs SNR.

Runs scripts/figures/fig_07_rank_adaptation.py on a 3GPP CDL channel (default CDL-C,
4 rx antennas) into results/sionna_cdl_gallery/.  See the matching .md.

Run: .venv/bin/python scripts/figures/cdl_fig_07_rank_adaptation.py [--fast --drops N --model C]
"""

import fig_07_rank_adaptation as orig

from nr_csi.figtools import cdllib

if __name__ == "__main__":
    cdllib.run_original(orig)
