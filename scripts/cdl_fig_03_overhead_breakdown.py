"""CDL (Sionna 38.901) fig_03 -- per-element overhead breakdown.

The bit counts are structural (set by codebook configuration, not channel
values), so the CDL figure matches the synthetic one; it is regenerated here
on CDL for a consistent gallery.  Into results/sionna_cdl_gallery/.

Run: .venv/bin/python scripts/cdl_fig_03_overhead_breakdown.py [--model C]
"""

import cdllib
import fig_03_overhead_breakdown as orig

if __name__ == "__main__":
    cdllib.run_original(orig)
