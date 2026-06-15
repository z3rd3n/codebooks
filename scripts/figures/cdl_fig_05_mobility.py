"""CDL (Sionna 38.901) fig_05 -- mobility: CSI aging vs the R18 predicted PMI.

On CDL, Doppler comes from the UE speed (default 30 km/h, longer slot interval)
instead of the synthetic Doppler knob -- a realistic time-evolving channel.
Into results/sionna_cdl_gallery/.  See the matching .md for the analysis.

Run: .venv/bin/python scripts/figures/cdl_fig_05_mobility.py [--fast --drops N --model C]
"""

import os

os.environ.setdefault("NRCSI_CDL_SLOTS", "12")  # aging window needs up to ~10 slots

import fig_05_mobility as orig  # noqa: E402

from nr_csi.figtools import cdllib  # noqa: E402

if __name__ == "__main__":
    cdllib.run_original(orig)
