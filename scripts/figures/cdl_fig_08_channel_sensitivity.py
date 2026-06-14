"""CDL (Sionna 38.901) fig_08 -- robustness to the channel.

CDL has a fixed cluster structure with no "number of rays" knob, so the left
panel (sparsity sweep) is flat on CDL -- the .md explains this and points to
results/sionna_cdl/cdl_models.png for the real channel-richness story.  The
right panel (estimation-noise robustness) maps directly to CDL.
Into results/sionna_cdl_gallery/.

Run: .venv/bin/python scripts/cdl_fig_08_channel_sensitivity.py [--fast --drops N --model C]
"""

import cdllib
import fig_08_channel_sensitivity as orig

if __name__ == "__main__":
    cdllib.run_original(orig)
