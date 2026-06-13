"""CDL (Sionna 38.901) fig_06 -- MU-MIMO ZF sum rate from reported PMIs.

Runs scripts/fig_06_mu_mimo.py on a 3GPP CDL channel (default CDL-C) into
results/sionna_cdl_gallery/.  See the matching .md for the analysis.

Run: .venv/bin/python scripts/cdl_fig_06_mu_mimo.py [--fast --drops N --model C]
"""

import cdllib
import fig_06_mu_mimo as orig

if __name__ == "__main__":
    cdllib.run_original(orig)
