"""Figure/plotting helpers shared by the comparison scripts under ``scripts/``.

* :mod:`nr_csi.figtools.figlib` -- synthetic-channel benchmark helpers
  (paired seeds, the standard scheme set, the beam-domain PEB view, the
  common ``--drops/--seed/--out/--fast`` CLI, and the figure save helper).
* :mod:`nr_csi.figtools.cdllib` -- the Sionna 3GPP CDL companion to
  ``figlib`` (``CDLReplay`` plus ``run_original`` for the ``cdl_fig_*`` wrappers).

These live in the installed package (rather than next to the scripts) so the
figure/reproduction scripts can be organised into subfolders and still import
them with a plain ``from nr_csi.figtools.figlib import ...``.
"""
