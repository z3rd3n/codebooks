"""CDL (Sionna 38.901) fig_09 -- regular vs port-selection codebooks.

Runs scripts/fig_09_port_selection.py on a 3GPP CDL channel (default CDL-C)
into results/sionna_cdl_gallery/.  The three channel-domain views (antenna,
DFT-PEB beam, tuned PEB) are scored on the same CDL drops.  See the matching .md.

Run: .venv/bin/python scripts/cdl_fig_09_port_selection.py [--fast --drops N --model C]
"""

import cdllib
import fig_09_port_selection as orig

if __name__ == "__main__":
    cdllib.run_original(orig)
