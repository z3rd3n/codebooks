"""CDL (Sionna 38.901) fig_12 -- scorecard: every family on every axis.

Runs scripts/fig_12_summary.py on a 3GPP CDL channel (default CDL-C) into
results/sionna_cdl_gallery/ (radar PNG + raw-numbers table).  See the matching .md.

Run: .venv/bin/python scripts/cdl_fig_12_summary.py [--fast --drops N --model C]
"""

import cdllib
import fig_12_summary as orig

if __name__ == "__main__":
    cdllib.run_original(orig)
