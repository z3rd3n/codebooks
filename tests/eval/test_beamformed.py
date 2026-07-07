"""Beamformed CSI-RS adapter (``BeamformedPortsScheme``).

Validates the gNB-side eigenbeam construction (orthonormal, per-pol block
structure), the power convention through the port mapping, and the physics
the adapter exists for: port selection over beamformed ports recovers the
channel a raw-element port selection fundamentally cannot.
"""

import numpy as np
import pytest

from nr_csi.channel import RandomRayChannel, Ray, SyntheticRayChannel
from nr_csi.codebooks import R17Type2Codebook
from nr_csi.config import AntennaConfig
from nr_csi.eval import evaluate, evaluate_mu
from nr_csi.eval.beamformed import BeamformedPMI, BeamformedPortsScheme

RAW = AntennaConfig.standard(4, 2)  # P = 16 physical ports
EFF = AntennaConfig.standard(4, 1)  # P_eff = 8 = 2 * Kb


def _adapter(pc=4, n3=4, kb=4, raw_ports=16):
    eff_ant = AntennaConfig.standard(kb, 1)
    inner = R17Type2Codebook(eff_ant, N3=n3, param_combination=pc)
    return BeamformedPortsScheme(inner, n_raw_ports=raw_ports, n_beams_per_pol=kb)


class TestBeamMatrix:
    def _H(self, seed=0, n3=4):
        chan = RandomRayChannel(RAW, N3=n3, n_rx=2, n_paths=4)
        return chan.generate(n_slots=1, rng=np.random.default_rng(seed))

    def test_orthonormal_columns(self):
        B = _adapter().beams(self._H())
        assert B.shape == (16, 8)
        assert np.allclose(B.conj().T @ B, np.eye(8), atol=1e-10)

    def test_per_pol_block_structure(self):
        """Same beams applied to each polarization, no cross-pol mixing."""
        B = _adapter().beams(self._H(1))
        assert np.allclose(B[8:, :4], 0) and np.allclose(B[:8, 4:], 0)
        assert np.allclose(B[:8, :4], B[8:, 4:])

    def test_beams_capture_dominant_subspace(self):
        """A single on-grid ray has a rank-1 per-pol covariance: the first
        eigenbeam must be (a phase rotation of) the ray's steering vector."""
        from nr_csi.utils import dft

        ray = Ray(gain=1.0, m1=5, m2=2, pol_phase=0.3)
        H = SyntheticRayChannel(RAW, [ray], N3=4).generate()
        B = _adapter(kb=4).beams(H)
        v = dft.spatial_beam(RAW, 5, 2)  # per-pol steering vector, |v_i| = 1/sqrt(P/2)
        v = v / np.linalg.norm(v)
        corr = np.abs(v.conj() @ B[:8, 0])
        assert corr > 1 - 1e-9

    def test_port_count_mismatch_rejected(self):
        with pytest.raises(ValueError, match="ports"):
            _adapter().beams(self._H()[..., :12])

    def test_inner_port_consistency_enforced(self):
        inner = R17Type2Codebook(EFF, N3=4, param_combination=4)  # 8 ports
        with pytest.raises(ValueError, match="CSI-RS ports"):
            BeamformedPortsScheme(inner, n_raw_ports=16, n_beams_per_pol=2)

    def test_bad_beam_counts_rejected(self):
        inner = R17Type2Codebook(EFF, N3=4, param_combination=4)
        with pytest.raises(ValueError, match="n_beams_per_pol"):
            BeamformedPortsScheme(inner, n_raw_ports=16, n_beams_per_pol=9)
        with pytest.raises(ValueError, match="n_raw_ports"):
            BeamformedPortsScheme(inner, n_raw_ports=15, n_beams_per_pol=4)


class TestPrecoderMapping:
    def test_power_invariant_preserved(self):
        """Orthonormal B keeps every layer column at norm 1/sqrt(rank)."""
        chan = RandomRayChannel(RAW, N3=4, n_rx=2, n_paths=4)
        bf = _adapter()
        rng = np.random.default_rng(2)
        for rank in (1, 2):
            H = chan.generate(n_slots=1, rng=rng)
            W = bf.precoder(bf.select(H, rank=rank))
            assert W.shape == (1, 4, 16, rank)
            col = np.linalg.norm(W, axis=-2)
            assert np.allclose(col, 1 / np.sqrt(rank), atol=1e-9)

    def test_overhead_delegates_to_inner(self):
        chan = RandomRayChannel(RAW, N3=4, n_rx=2)
        bf = _adapter()
        pmi = bf.select(chan.generate(rng=np.random.default_rng(3)), rank=1)
        assert isinstance(pmi, BeamformedPMI)
        assert bf.overhead_bits(pmi) == bf.inner.overhead_bits(pmi.inner)
        assert bf.total_overhead_bits(pmi) == bf.inner.total_overhead_bits(pmi.inner)

    def test_attribute_passthrough(self):
        bf = _adapter()
        assert bf.L == bf.inner.L
        assert getattr(bf, "N4", 1) == 1  # inner has no N4 -> default applies
        with pytest.raises(AttributeError):
            _ = bf.nonexistent_attr


class TestPhysics:
    def test_single_ray_channel_recovered_almost_exactly(self):
        """One propagation path: a single eigenbeam per pol carries all the
        energy, so PS-over-beamformed-ports must reach SGCS ~ 1 while
        PS-over-raw-elements cannot (energy spread over all 8 elements)."""
        ray = Ray(gain=1.0, m1=5, m2=2, pol_phase=1.1, delay=1)
        chan = SyntheticRayChannel(RAW, [ray], N3=4, n_rx=2)
        bf = _adapter(pc=2, kb=2)  # P_eff = 4
        raw = R17Type2Codebook(RAW, N3=4, param_combination=2)
        r_bf = evaluate(bf, chan, snr_db=[10.0], rank=1, n_drops=2,
                        rng=np.random.default_rng(0))
        r_raw = evaluate(raw, chan, snr_db=[10.0], rank=1, n_drops=2,
                         rng=np.random.default_rng(0))
        assert r_bf.sgcs > 0.98
        assert r_bf.sgcs > r_raw.sgcs + 0.1

    def test_multipath_same_sgcs_at_fraction_of_bits(self):
        """4-path channel, same codebook both ways: the beamformed variant
        must match or beat raw-element SGCS while spending far fewer bits
        (8 effective vs 16 raw ports to quantize)."""
        chan = RandomRayChannel(RAW, N3=4, n_rx=2, n_paths=4)
        raw = R17Type2Codebook(RAW, N3=4, param_combination=4)
        bf = _adapter(pc=4, kb=4)
        r_raw = evaluate(raw, chan, snr_db=[10.0], rank=1, n_drops=8,
                         rng=np.random.default_rng(0))
        r_bf = evaluate(bf, chan, snr_db=[10.0], rank=1, n_drops=8,
                        rng=np.random.default_rng(0))
        assert r_bf.sgcs >= r_raw.sgcs - 1e-9
        assert r_bf.overhead_bits < 0.5 * r_raw.overhead_bits

    def test_evaluate_bounds_hold(self):
        chan = RandomRayChannel(RAW, N3=4, n_rx=2, n_paths=4)
        res = evaluate(_adapter(), chan, snr_db=[0.0, 10.0], rank=2, n_drops=4,
                       rng=np.random.default_rng(4))
        assert all(c <= u + 1e-9 for c, u in zip(res.se, res.capacity_upper_bound))
        assert all(m <= s + 1e-9 for m, s in zip(res.se_mmse, res.se))
        assert 0 < res.sgcs <= 1

    def test_composes_with_evaluate_mu(self):
        chan = RandomRayChannel(RAW, N3=4, n_rx=1, n_paths=4)
        res = evaluate_mu(_adapter(), chan, n_users=2, snr_db=[10.0], n_drops=3,
                          rng=np.random.default_rng(5))
        assert res.sum_rate[0] > 0
        assert np.asarray(res.per_drop_user_rates).shape == (3, 1, 2)
