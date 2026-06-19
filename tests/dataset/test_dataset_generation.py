"""Tests for the CDL dataset package.

Pure-NumPy tests (preprocess / splits / config) always run; HDF5 I/O tests
``importorskip('h5py')``; the end-to-end generation test additionally needs the
``sionna`` extra and is marked ``sionna`` (loads TensorFlow)."""

from __future__ import annotations

import numpy as np
import pytest

from nr_csi.config import AntennaConfig
from nr_csi.dataset import (
    DatasetConfig,
    config_tag,
    parse_antenna,
    samples_per_config,
    split_indices,
)
from nr_csi.dataset import (
    preprocess as pp,
)


# --------------------------------------------------------------------------- #
# Preprocess (pure NumPy)
# --------------------------------------------------------------------------- #
@pytest.fixture
def H():
    ant = AntennaConfig.standard(4, 2)  # P = 16
    rng = np.random.default_rng(0)
    H = rng.standard_normal((5, 2, ant.P, 32)) + 1j * rng.standard_normal((5, 2, ant.P, 32))
    return ant, H


def test_angular_delay_roundtrip_exact(H):
    ant, H = H
    ad = pp.to_angular_delay(H, ant, n_delay=None)  # no truncation -> lossless
    back = pp.from_angular_delay(ad, ant, n_freq=H.shape[-1])
    assert np.allclose(back, H, atol=1e-10)


def test_spatial_dft_unitary(H):
    ant, H = H
    sd = pp.spatial_dft(H, ant)
    assert np.isclose(np.sum(np.abs(sd) ** 2), np.sum(np.abs(H) ** 2))
    assert np.allclose(pp.spatial_dft(sd, ant, inverse=True), H, atol=1e-10)


def test_delay_truncation_shape(H):
    ant, H = H
    ad = pp.to_angular_delay(H, ant, n_delay=8)
    assert ad.shape == (5, 2, ant.P, 8)


def test_power_normalize(H):
    ant, H = H
    Hn, norm = pp.power_normalize(H)
    assert norm.shape == (5, 1, 1, 1)
    e = np.sum(np.abs(Hn) ** 2, axis=(-3, -2, -1))
    assert np.allclose(e, 1.0)


def test_real_imag_roundtrip(H):
    _, H = H
    x = pp.stack_real_imag(H)
    assert x.shape == (2, *H.shape) and x.dtype == np.float32
    assert np.allclose(pp.complex_from_real_imag(x), H, atol=1e-6)


def test_apply_awgn_changes_but_keeps_shape(H):
    _, H = H
    noisy = pp.apply_awgn(H, snr_db=10.0, rng=np.random.default_rng(1))
    assert noisy.shape == H.shape and not np.allclose(noisy, H)


# --------------------------------------------------------------------------- #
# Splits + config helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["random", "block"])
def test_split_disjoint_and_covers(method):
    idx = split_indices(100, (0.8, 0.1, 0.1), seed=3, method=method)
    allidx = np.concatenate([idx["train"], idx["val"], idx["test"]])
    assert sorted(allidx.tolist()) == list(range(100))
    assert len(idx["train"]) == 80 and len(idx["val"]) == 10 and len(idx["test"]) == 10


def test_split_deterministic():
    a = split_indices(50, seed=7)
    b = split_indices(50, seed=7)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_config_helpers():
    assert parse_antenna("16x1").P == 32
    assert config_tag(AntennaConfig.standard(4, 2)) == "4x2_P16"
    assert samples_per_config(10, 3) == [4, 3, 3]


def test_default_generalization_set():
    cfg = DatasetConfig.default_nlos_generalization(n_samples=12)
    assert [a.P for a in cfg.antennas] == [16, 32, 64]
    assert cfg.profiles == ("A", "B", "C")


def test_config_validation():
    with pytest.raises(ValueError):
        DatasetConfig(antennas=[AntennaConfig.standard(4, 2)], n_freq=1000, fft_size=512)
    with pytest.raises(ValueError):
        DatasetConfig(antennas=[AntennaConfig.standard(4, 2)], splits=(0.5, 0.4, 0.4))


# --------------------------------------------------------------------------- #
# HDF5 I/O (needs h5py)
# --------------------------------------------------------------------------- #
def test_io_roundtrip_and_split(tmp_path):
    pytest.importorskip("h5py")
    from nr_csi.dataset import io

    ant = AntennaConfig.standard(4, 2)
    tag = config_tag(ant)
    cfg = DatasetConfig(antennas=[ant], n_samples=10, shard_size=6, n_freq=16,
                        fft_size=32, out_dir=str(tmp_path))
    rng = np.random.default_rng(0)
    attrs = {"config_id": 0, "tag": tag, "N1": 4, "N2": 2, "O1": 4, "O2": 4,
             "Ng": 1, "P": ant.P, "n_rx": 2, "n_freq": 16}
    shards = []
    for si, c in enumerate([6, 4]):
        Hs = (rng.standard_normal((c, 2, ant.P, 16))
              + 1j * rng.standard_normal((c, 2, ant.P, 16))).astype(np.complex64)
        meta = {"norm": np.ones(c), "cdl_model": np.array(["A"] * c, dtype="S1"),
                "delay_spread_ns": np.full(c, 100.0), "ue_speed_kmh": np.full(c, 3.0)}
        path = tmp_path / tag / f"shard_{si:05d}.h5"
        io.write_shard(path, Hs, meta, attrs)
        shards.append(str(path.relative_to(tmp_path)))

    entry = {"config_id": 0, "tag": tag, "N1": 4, "N2": 2, "O1": 4, "O2": 4,
             "Ng": 1, "P": ant.P, "n_samples": 10, "shards": shards}
    io.write_manifest(tmp_path, cfg, [entry])

    m = io.read_manifest(tmp_path)
    assert m["total_samples"] == 10

    data = io.load_config_array(tmp_path, tag)
    assert data["H"].shape == (10, 2, ant.P, 16) and data["H"].dtype == np.complex64

    sizes = {s: io.load_config_array(tmp_path, tag, split=s)["H"].shape[0]
             for s in ("train", "val", "test")}
    assert sum(sizes.values()) == 10

    assert sum(1 for _ in io.iter_samples(tmp_path)) == 10
    one = next(io.iter_samples(tmp_path))
    assert one[0].shape == (2, ant.P, 16) and one[1]["cdl_model"] == "A"


# --------------------------------------------------------------------------- #
# End-to-end generation (needs sionna + tensorflow + h5py)
# --------------------------------------------------------------------------- #
@pytest.mark.sionna
def test_generate_tiny_end_to_end(tmp_path):
    pytest.importorskip("sionna")
    pytest.importorskip("tensorflow")
    pytest.importorskip("h5py")
    from nr_csi.dataset import generate_dataset, io

    ant = AntennaConfig.standard(4, 2)
    cfg = DatasetConfig(antennas=[ant], profiles=("A",), n_samples=6, batch_size=3,
                        shard_size=4, n_freq=16, fft_size=64, n_rx=2,
                        out_dir=str(tmp_path / "ds"))
    manifest = generate_dataset(cfg, progress=False)

    assert manifest["total_samples"] == 6
    data = io.load_config_array(tmp_path / "ds", config_tag(ant))
    assert data["H"].shape == (6, 2, ant.P, 16) and data["H"].dtype == np.complex64
    # stored channels are energy-normalized
    e = np.sum(np.abs(data["H"]) ** 2, axis=(1, 2, 3))
    assert np.allclose(e, 1.0, atol=1e-3)
    # metadata respects the configured profile + DS range
    assert set(data["cdl_model"]) == {"A"}
    assert np.all((data["delay_spread_ns"] >= 30.0) & (data["delay_spread_ns"] <= 300.0))
