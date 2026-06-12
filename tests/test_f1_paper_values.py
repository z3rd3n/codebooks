"""Full digitized-f1 regression (plan A1): every curve of the paper's Fig. f1
against the frozen reproduction setting.

The reference table in ``nr_csi.eval.f1.F1_DIGITIZED`` was digitized from
paper/pictures/f1.pdf (reading error ~0.1 b/s/Hz).  The channel realization is
not specified by the paper, so the test asserts a documented tolerance band --
max(+-0.6 b/s/Hz absolute, +-8% relative) -- around each of the 9 curves x 7
SNR points, using the calibrated ``F1_REPRODUCTION`` configuration.
"""

import numpy as np
import pytest

from nr_csi.eval.f1 import (
    F1_DIGITIZED,
    F1_REPRODUCTION,
    F1_SNR_DB,
    case_errors,
    run_f1_case,
)

pytestmark = pytest.mark.slow

ABS_TOL = 0.6
REL_TOL = 0.08


@pytest.fixture(scope="module")
def curves():
    out = {}
    for (N, ns) in F1_DIGITIZED:
        out[(N, ns)] = run_f1_case(
            N, ns, F1_SNR_DB,
            n_drops=F1_REPRODUCTION["n_drops"],
            n_paths=F1_REPRODUCTION["n_paths"],
            n_psk=F1_REPRODUCTION["n_psk"],
            seed=F1_REPRODUCTION["seed"],
            t1_second_beam=F1_REPRODUCTION["t1_second_beam"],
        )
    return out


@pytest.mark.parametrize("N,ns", sorted(F1_DIGITIZED), ids=lambda x: str(x))
def test_digitized_band(curves, N, ns):
    errs = case_errors(curves[(N, ns)], N, ns)
    ref = F1_DIGITIZED[(N, ns)]
    for key in ("ub", "t2", "t1"):
        band = np.maximum(ABS_TOL, REL_TOL * np.array(ref[key]))
        assert np.all(np.abs(errs[key]) <= band), (
            f"N={N}, Ns={ns}, curve {key}: errors {np.round(errs[key], 2)} "
            f"exceed band {np.round(band, 2)}"
        )


@pytest.mark.parametrize("N,ns", sorted(F1_DIGITIZED), ids=lambda x: str(x))
def test_curve_ordering(curves, N, ns):
    """UB >= Type II >= Type I at every SNR point (paper's qualitative shape)."""
    c = curves[(N, ns)]
    assert np.all(c.upper_bound >= c.type2 - 1e-9)
    assert np.all(c.type2 >= c.type1 - 1e-9)


def test_gap_grows_with_array_size(curves):
    """Type I-to-UB gap is larger at N=16 than N=4 (paper's main message:
    beam combination matters more for larger arrays)."""
    gap4 = np.mean(curves[(4, 1)].upper_bound - curves[(4, 1)].type1)
    gap16 = np.mean(curves[(16, 1)].upper_bound - curves[(16, 1)].type1)
    assert gap16 > gap4
