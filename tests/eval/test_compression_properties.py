"""Compression-fidelity properties (plan B3): forced-Mv losslessness,
rate-distortion monotonicity, the R18 N4 sweep, and quantizer error bounds."""

import numpy as np
import pytest

from nr_csi.baselines.ideal import eigen_precoder
from nr_csi.channel import RandomRayChannel, Ray, SyntheticRayChannel
from nr_csi.codebooks import R15Type2Codebook, R16Type2Codebook, R18Type2Codebook
from nr_csi.codebooks.etype2_r16 import R16Type2Codebook as _R16
from nr_csi.config import AntennaConfig
from nr_csi.metrics.similarity import sgcs
from nr_csi.utils import dft
from nr_csi.utils import quantization as qt

ANT = AntennaConfig.standard(4, 2)  # P = 16


class FullTapR16(_R16):
    """Test-only: keep all N3 delay taps and never prune coefficients, so the
    frequency-domain DFT compression is exactly invertible."""

    def Mv(self, rank: int) -> int:
        return self.N3

    @property
    def K0(self) -> int:
        return 2 * self.L * self.N3


def tap_exact_channel(N3=4, L=4):
    """Rank-1 channel whose per-subband targets have *table-exact* tap-domain
    coefficients: one delay tap per beam row, so per-subband norms are
    constant and quantization is exact end to end.

    A (4,4) array with beams on 2-of-4 indices in *both* dimensions makes
    group (0,0) the unique exact basis: with k beams on k distinct indices of
    a length-k dimension, any oversampled group spans that whole dimension and
    the group-selection heuristic may legitimately pick a basis in which the
    tap matrix is dense."""
    ant = AntennaConfig.standard(4, 4)
    taps_per_row = [0, 1, 2, 3, 1, 0, 2, 3]
    k1_pol = [15, 14]  # p1 = 1, 2^(-1/4)
    k2_per_row = [7, 5, 4, 3, 7, 6, 4, 2]
    c_per_row = [0, 3, 9, 14, 5, 1, 12, 7]
    X = np.zeros((2 * L, N3), dtype=complex)
    for i in range(2 * L):
        p1 = qt.R16_REF_AMP[k1_pol[i // L]]
        amp = p1 * qt.R16_DIFF_AMP[k2_per_row[i]]
        X[i, taps_per_row[i]] = amp * np.exp(2j * np.pi * c_per_row[i] / 16)
    Y = dft.freq_basis(N3, np.arange(N3))  # (N3 taps, N3 subbands)
    C = X @ Y  # (2L, N3) per-subband coefficients
    B = dft.orthogonal_group(ant, 0, 0)[[0, 1, 4, 5]]  # (n1, n2) in {0,1}x{0,1}
    w = np.concatenate([B.T @ C[:L], B.T @ C[L:]], axis=0)  # (P, N3)
    H = w.conj().T[None, :, None, :]  # (1, N3, 1, P)
    return ant, H, (w / np.linalg.norm(w, axis=0)).T  # targets (N3, P)


class TestForcedFullTapLossless:
    def test_fd_compression_is_lossless_with_all_taps(self):
        ant, H, targets = tap_exact_channel()
        cbk = FullTapR16(ant, N3=4, param_combination=4)
        pmi = cbk.select(H, rank=1)
        W = cbk.precoder(pmi)
        for t in range(4):
            assert sgcs(targets[t][:, None], W[0, t]) > 1 - 1e-12

    def test_standard_mv_is_lossy_on_the_same_channel(self):
        """Contrast: the spec Mv = ceil(p_v*N3/R) = 1 cannot carry 4 taps."""
        ant, H, targets = tap_exact_channel()
        cbk = R16Type2Codebook(ant, N3=4, param_combination=4)
        assert cbk.Mv(1) == 1
        W = cbk.precoder(cbk.select(H, rank=1))
        s = np.mean([sgcs(targets[t][:, None], W[0, t]) for t in range(4)])
        assert s < 0.9


def mean_sgcs_and_bits(cbk, n_drops=12, rank=1, seed=0, n_paths=4):
    chan = RandomRayChannel(cbk.antenna, N3=cbk.N3, n_rx=2, n_paths=n_paths)
    rng = np.random.default_rng(seed)
    n_slots = getattr(cbk, "N4", 1)
    ss, bb = [], []
    for _ in range(n_drops):
        H = chan.generate(n_slots=n_slots, rng=rng)
        pmi = cbk.select(H, rank=rank)
        W = cbk.precoder(pmi)
        ss.append(sgcs(eigen_precoder(H, rank=rank), W))
        bb.append(cbk.total_overhead_bits(pmi))
    return float(np.mean(ss)), float(np.mean(bb))


class TestMonotonicity:
    """All rate-distortion knobs must point the right way: more beams/taps/
    budget/phase resolution => SGCS non-decreasing AND overhead increasing."""

    def assert_knob(self, cbks, tol=0.005):
        results = [mean_sgcs_and_bits(c, seed=3) for c in cbks]
        for (s0, b0), (s1, b1) in zip(results, results[1:]):
            assert s1 >= s0 - tol, f"SGCS dropped: {s0:.4f} -> {s1:.4f}"
            assert b1 > b0, f"overhead did not grow: {b0} -> {b1}"

    def test_r15_more_beams(self):
        self.assert_knob([R15Type2Codebook(ANT, N3=8, L=L) for L in (2, 3, 4)])

    def test_r15_finer_phase(self):
        s4, b4 = mean_sgcs_and_bits(R15Type2Codebook(ANT, N3=8, L=4, n_psk=4), seed=3)
        s8, b8 = mean_sgcs_and_bits(R15Type2Codebook(ANT, N3=8, L=4, n_psk=8), seed=3)
        assert s8 >= s4 - 0.005 and b8 > b4

    def test_r16_more_beams(self):
        # combos 2 -> 4 -> 7: L = 2, 4, 6 at fixed p_v = 1/4, beta = 1/2.
        # combo 7 is only configurable at P = 32 with ranks 3-4 disallowed
        # (5.2.2.2.5), so this knob runs on the 32-port array.
        ant32 = AntennaConfig.standard(4, 4)
        self.assert_knob(
            [
                R16Type2Codebook(
                    ant32, N3=8, param_combination=c,
                    ri_restriction=[1, 1, 0, 0] if c in (7, 8) else None,
                )
                for c in (2, 4, 7)
            ]
        )

    def test_r16_more_taps(self):
        # combos 4 -> 6: p_v 1/4 -> 1/2 (Mv 2 -> 4 at N3 = 8), L = 4, beta = 1/2
        self.assert_knob(
            [R16Type2Codebook(ANT, N3=8, param_combination=c) for c in (4, 6)]
        )

    def test_r16_larger_coefficient_budget(self):
        # combos 3 -> 4 -> 5: beta 1/4 -> 1/2 -> 3/4 at L = 4, p_v = 1/4
        cbks = [R16Type2Codebook(ANT, N3=8, param_combination=c) for c in (3, 4, 5)]
        results = [mean_sgcs_and_bits(c, seed=3) for c in cbks]
        for (s0, b0), (s1, b1) in zip(results, results[1:]):
            assert s1 >= s0 - 0.005
            assert b1 >= b0  # K_NZ caps at what the channel needs

    def test_r18_more_beams(self):
        # combos 2 -> 7: L = 2 -> 4 at p_v = 1/4 -> 1/2... use 3 -> 8 (L 4 -> 6)
        self.assert_knob(
            [R18Type2Codebook(ANT, N3=8, N4=2, param_combination=c) for c in (2, 7)]
        )


class TestR18StaticBudget:
    """S2 lock-in: the spec budget K0 = ceil(2*beta*L*M1*Q) doubles with the
    Q = 2 Doppler components, and on a *static* channel the idle Doppler bin
    donates its half of the budget to the DC bin.  So R18 at matched
    (L, p_v, beta) is at least as accurate as R16 even with nothing to
    predict -- at strictly larger overhead.  Spec-faithful behavior, not a
    bug (see README notes; the K0 formula itself is locked in
    test_etype2_r18.py)."""

    def test_static_r18_at_least_matched_r16(self):
        s16, b16 = mean_sgcs_and_bits(
            R16Type2Codebook(ANT, N3=8, param_combination=6), n_drops=20, seed=30)
        s18, b18 = mean_sgcs_and_bits(
            R18Type2Codebook(ANT, N3=8, N4=4, param_combination=7), n_drops=20, seed=30)
        assert s18 >= s16 - 0.005
        assert b18 > b16


class TestR18N4Sweep:
    def _horizon_sgcs(self, doppler, N4, horizon=8):
        """Mean SGCS over a fixed 8-slot horizon: one R18 report covering the
        first N4 slots, last predicted precoder held for the rest.

        The two rays differ in Doppler: a common rotation is absorbed by the
        strongest-coefficient phase reference, so only the *relative* shift
        exercises the temporal basis."""
        rays = [
            Ray(gain=1.0, m1=4, m2=2, pol_phase=0.7),
            Ray(gain=0.5, m1=8, m2=6, delay=1, doppler=doppler, pol_phase=2.1),
        ]
        chan = SyntheticRayChannel(ANT, rays, N3=4, n_rx=1, doppler_period=horizon)
        H = chan.generate(n_slots=horizon)
        cbk = R18Type2Codebook(ANT, N3=4, N4=N4, param_combination=7)  # Mv = 2
        W = cbk.precoder(cbk.select(H[:N4], rank=1))  # (N4, N3, P, 1)
        targets = eigen_precoder(H, rank=1)  # (horizon, N3, P, 1)
        vals = [
            sgcs(targets[s], W[min(s, N4 - 1)]) for s in range(horizon)
        ]
        return float(np.mean(vals))

    def test_prediction_improves_with_n4_on_grid(self):
        scores = [self._horizon_sgcs(doppler=1.0, N4=n) for n in (2, 4, 8)]
        assert scores[0] < scores[1] < scores[2]
        assert scores[2] > 0.99  # on-grid Doppler at N4 = 8: exact prediction

    def test_off_grid_degrades_gracefully(self):
        on = self._horizon_sgcs(doppler=1.0, N4=8)
        off = self._horizon_sgcs(doppler=1.37, N4=8)
        assert off < on
        assert off > 0.7  # graceful, not catastrophic


class TestQuantizerDistortionBounds:
    def test_r16_reference_amplitude_bound(self):
        rng = np.random.default_rng(0)
        table = qt.R16_REF_AMP
        vals = rng.uniform(np.nanmin(table), 1.0, 200)
        idx = qt.quantize_amplitude(vals, table)
        clean = np.sort(table[1:])
        for x, i in zip(vals, idx):
            pos = np.clip(np.searchsorted(clean, x), 1, len(clean) - 1)
            half = (clean[pos] - clean[pos - 1]) / 2
            assert abs(table[i] - x) <= half + 1e-12

    def test_r16_differential_amplitude_bound(self):
        rng = np.random.default_rng(1)
        table = qt.R16_DIFF_AMP
        vals = rng.uniform(table[0], 1.0, 200)
        idx = qt.quantize_amplitude(vals, table)
        for x, i in zip(vals, idx):
            pos = np.clip(np.searchsorted(table, x), 1, len(table) - 1)
            half = (table[pos] - table[pos - 1]) / 2
            assert abs(table[i] - x) <= half + 1e-12

    def test_r15_wideband_amplitude_bound(self):
        rng = np.random.default_rng(2)
        table = qt.R15_WB_AMP
        vals = rng.uniform(0.0, 1.0, 200)
        idx = qt.quantize_amplitude(vals, table)
        for x, i in zip(vals, idx):
            pos = np.clip(np.searchsorted(table, x), 1, len(table) - 1)
            half = (table[pos] - table[pos - 1]) / 2
            assert abs(table[i] - x) <= half + 1e-12

    @pytest.mark.parametrize("n_psk", [4, 8, 16])
    def test_phase_bound(self, n_psk):
        rng = np.random.default_rng(3)
        angles = rng.uniform(-np.pi, np.pi, 500)
        c = qt.quantize_phase(angles, n_psk)
        err = np.angle(np.exp(1j * (angles - 2 * np.pi * c / n_psk)))
        assert np.all(np.abs(err) <= np.pi / n_psk + 1e-12)
