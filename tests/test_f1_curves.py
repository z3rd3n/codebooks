"""Statistical reproduction of the paper's Fig. f1 (slow).

Digitized anchor values from paper/pictures/f1.pdf (approximate):
  N=4,  1 stream: all three curves nearly overlap (~2.3 b/s/Hz @ 0 dB).
  N=16, 1 stream @ 0 dB: Type I ~3.4, Type II ~4.0, UB ~4.2.
  N=16, 2 streams: the Type I gap widens markedly vs 1 stream.

The paper's channel realization is unspecified, so values are matched with
generous tolerances; orderings and gap relations are asserted strictly.
"""

import numpy as np
import pytest

from nr_csi.eval.f1 import run_f1_case

pytestmark = pytest.mark.slow

SNR = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])


@pytest.fixture(scope="module")
def curves():
    return {
        (4, 1): run_f1_case(4, 1, SNR, n_drops=300, seed=1),
        (16, 1): run_f1_case(16, 1, SNR, n_drops=300, seed=2),
        (16, 2): run_f1_case(16, 2, SNR, n_drops=300, seed=3),
    }


def test_ordering_everywhere(curves):
    for cv in curves.values():
        assert (cv.upper_bound >= cv.type2 - 1e-9).all()
        assert (cv.type2 >= cv.type1 - 0.05).all()  # small statistical slack


def test_small_array_curves_nearly_overlap(curves):
    cv = curves[(4, 1)]
    assert (cv.upper_bound - cv.type2 < 0.15).all()
    assert (cv.upper_bound - cv.type1 < 0.8).all()


def test_gap_grows_with_antennas(curves):
    """Paper: 'As the number of antennas increases, the gap becomes more
    pronounced.'"""
    g4 = np.mean(curves[(4, 1)].upper_bound - curves[(4, 1)].type1)
    g16 = np.mean(curves[(16, 1)].upper_bound - curves[(16, 1)].type1)
    assert g16 > g4


def test_gap_grows_with_streams(curves):
    """Paper: 'when the number of data streams increases, the gap is further
    widened.'"""
    g1 = np.mean(curves[(16, 1)].type2 - curves[(16, 1)].type1)
    g2 = np.mean(curves[(16, 2)].type2 - curves[(16, 2)].type1)
    assert g2 > g1


def test_spot_values_n16_single_stream(curves):
    """Digitized f1 anchors at 0 dB, +-0.5 b/s/Hz tolerance."""
    cv = curves[(16, 1)]
    i0 = list(SNR).index(0.0)
    assert abs(cv.upper_bound[i0] - 4.2) < 0.5
    assert abs(cv.type2[i0] - 4.0) < 0.5
    assert abs(cv.type1[i0] - 3.4) < 0.5


def test_upper_bound_matches_full_array_gain(curves):
    """Single-stream UB at high SNR approaches log2(rho * N) + O(1)."""
    for N in (4, 16):
        cv = curves[(N, 1)]
        i30 = list(SNR).index(30.0)
        assert abs(cv.upper_bound[i30] - np.log2(1 + 1000 * N)) < 1.0
