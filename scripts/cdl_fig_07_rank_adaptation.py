"""CDL (Sionna 38.901) fig_07 -- rank (RI) adaptation: layers vs SNR.

Runs scripts/fig_07_rank_adaptation.py on a 3GPP CDL channel (default CDL-C,
4 rx antennas) into results/sionna_cdl_gallery/.  See the matching .md.

Run: .venv/bin/python scripts/cdl_fig_07_rank_adaptation.py [--fast --drops N --model C]
"""

import cdllib
import fig_07_rank_adaptation as orig

if __name__ == "__main__":
    cdllib.run_original(orig)
