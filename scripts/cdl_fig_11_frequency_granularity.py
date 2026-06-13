"""CDL (Sionna 38.901) fig_11 -- frequency granularity: fidelity and cost vs N3.

On CDL the physical frequency selectivity is set by the model's delay spread
(not the synthetic delay-fraction knob); sweeping N3 changes only the reporting
granularity.  Into results/sionna_cdl_gallery/.  See the matching .md.

Run: .venv/bin/python scripts/cdl_fig_11_frequency_granularity.py [--fast --drops N --model C]
"""

import cdllib
import fig_11_frequency_granularity as orig

if __name__ == "__main__":
    cdllib.run_original(orig)
