"""CDL (Sionna 38.901) fig_04 -- overhead scaling laws.

fig_04 is computed entirely from the spec bit-count formulas and uses NO
channel, so the CDL figure is identical to the synthetic one.  It is included
for a complete gallery; the .md explains the channel-independence.
Into results/sionna_cdl_gallery/.

Run: .venv/bin/python scripts/figures/cdl_fig_04_overhead_scaling.py
"""

import fig_04_overhead_scaling as orig

from nr_csi.figtools import cdllib

if __name__ == "__main__":
    cdllib.run_original(orig)
