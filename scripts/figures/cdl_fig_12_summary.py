"""CDL (Sionna 38.901) fig_12 -- scorecard: every family on every axis.

Runs scripts/figures/fig_12_summary.py on a 3GPP CDL channel (default CDL-C) into
results/sionna_cdl_gallery/ (radar PNG + raw-numbers table).  See the matching .md.

Run: .venv/bin/python scripts/figures/cdl_fig_12_summary.py [--fast --drops N --model C]
"""

import fig_12_summary as orig

from nr_csi.figtools import cdllib

if __name__ == "__main__":
    cdllib.run_original(orig)
