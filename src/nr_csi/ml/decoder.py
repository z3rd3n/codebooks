"""gNB-side learned GLIMPSE decoder (TensorFlow, optional dependency).

An unrolled alternating-projection network.  Iteration ``k``:

    x <- x + alpha_k A_m^H (u - A_m x)          (exact data consistency:
                                                 rows of A are orthonormal)
    x <- x + CNN_k(grid(x), conditioning)       (learned prior step)

The prior CNN is a small residual 3D convolution over the physical
``(delay, angle_1, angle_2)`` grid with the two polarizations (and Re/Im) as
channels, using *circular* padding on all three axes -- the angle-delay
domain is a product of DFTs, so its topology genuinely is a 3-torus.
Conditioning channels carry the report's measurement count and quantizer
depth so ONE network serves every payload size (trained with measurement
dropout; see ``scripts/ml/train_glimpse.py``).

Everything trainable lives here, at the gNB: the UE-side report
(:mod:`.projection`/:mod:`.quantizer`) never changes.  Weights are stored as
a plain ``.npz`` plus a JSON manifest, loadable via :class:`KerasDecoder`
without any framework-specific model files.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass

import numpy as np

from ..config import AntennaConfig
from .projection import GlimpseCodec
from .quantizer import lloyd_max


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "TensorFlow is required for the learned GLIMPSE decoder; "
            'install the optional extra: pip install -e ".[sionna]"'
        ) from exc
    return tf


@dataclass(frozen=True)
class DecoderConfig:
    """Architecture + codec hyperparameters (JSON-serializable manifest)."""

    N1: int
    N2: int
    N3: int
    m_max: int = 64
    seed: int = 0  # random-basis seed (used only when no codec is provided)
    basis: str = "klt"  # provenance tag; the actual A/sigma are bundled in the .npz
    iterations: int = 6
    hidden: int = 32
    bits_max: int = 6  # normalizer for the quantizer-depth conditioning channel

    @property
    def P(self) -> int:
        return 2 * self.N1 * self.N2

    @property
    def D(self) -> int:
        return self.P * self.N3


def _circular_pad3d(tf, x):
    """Wrap-pad 1 voxel on each side of the (d1, d2, d3) axes of NDHWC."""
    x = tf.concat([x[:, -1:], x, x[:, :1]], axis=1)
    x = tf.concat([x[:, :, -1:], x, x[:, :, :1]], axis=2)
    x = tf.concat([x[:, :, :, -1:], x, x[:, :, :, :1]], axis=3)
    return x


class UnrolledDecoder:
    """The trainable network.  See the module docstring for the iteration."""

    def __init__(self, cfg: DecoderConfig, codec: GlimpseCodec | None = None,
                 rng_seed: int = 7) -> None:
        tf = _tf()
        self.cfg = cfg
        if codec is None:
            codec = GlimpseCodec(
                AntennaConfig.standard(cfg.N1, cfg.N2), cfg.N3,
                m_max=cfg.m_max, seed=cfg.seed,
            )
        self.codec = codec
        A = codec.A  # (m_max, D) complex, rows orthonormal
        self.Ar = tf.constant(A.real, tf.float32)
        self.Ai = tf.constant(A.imag, tf.float32)
        self.sigma = tf.constant(codec._sigma, tf.float32)  # (m_max,) coord std
        init = tf.random.Generator.from_seed(rng_seed)
        self.variables: list = []
        self._conv: list[list] = []
        c_in, c_h = 6, cfg.hidden  # 4 signal channels + 2 conditioning
        for k in range(cfg.iterations):
            shapes = [(3, 3, 3, c_in, c_h), (3, 3, 3, c_h, c_h), (3, 3, 3, c_h, 4)]
            layer = []
            for s in shapes:
                fan_in = np.prod(s[:4])
                w = tf.Variable(
                    init.normal(s, stddev=np.sqrt(2.0 / fan_in)), name=f"w{k}_{len(layer)}"
                )
                b = tf.Variable(tf.zeros(s[-1]), name=f"b{k}_{len(layer)}")
                layer.append((w, b))
                self.variables += [w, b]
            self._conv.append(layer)
        self.alpha = tf.Variable(tf.ones(cfg.iterations), name="alpha")
        self.gamma = tf.Variable(tf.zeros(cfg.iterations), name="gamma")  # prior step scale
        self.variables += [self.alpha, self.gamma]

    # ------------------------------------------------------- complex helpers
    def _measure(self, tf, xr, xi):
        """(batch, D) x2 -> (batch, m_max) x2:  y = A x."""
        yr = tf.matmul(xr, self.Ar, transpose_b=True) - tf.matmul(xi, self.Ai, transpose_b=True)
        yi = tf.matmul(xr, self.Ai, transpose_b=True) + tf.matmul(xi, self.Ar, transpose_b=True)
        return yr, yi

    def _adjoint(self, tf, ur, ui):
        """(batch, m_max) x2 -> (batch, D) x2:  x = A^H u."""
        xr = tf.matmul(ur, self.Ar) + tf.matmul(ui, self.Ai)
        xi = tf.matmul(ui, self.Ar) - tf.matmul(ur, self.Ai)
        return xr, xi

    def _to_grid(self, tf, xr, xi):
        """vec (batch, D) x2 -> grid (batch, N3, N1, N2, 4).

        vec index = (pol*N1*N2 + n1*N2 + n2) * N3 + delay; channels are
        (Re pol0, Im pol0, Re pol1, Im pol1).
        """
        c = self.cfg
        g = tf.stack([xr, xi], axis=-1)  # (batch, D, 2)
        g = tf.reshape(g, (-1, 2, c.N1, c.N2, c.N3, 2))
        g = tf.transpose(g, (0, 4, 2, 3, 1, 5))  # (batch, N3, N1, N2, pol, ri)
        return tf.reshape(g, (-1, c.N3, c.N1, c.N2, 4))

    def _from_grid(self, tf, g):
        c = self.cfg
        g = tf.reshape(g, (-1, c.N3, c.N1, c.N2, 2, 2))
        g = tf.transpose(g, (0, 4, 2, 3, 1, 5))  # (batch, pol, N1, N2, N3, ri)
        g = tf.reshape(g, (-1, c.D, 2))
        return g[..., 0], g[..., 1]

    def _prior(self, tf, k, g, cond):
        """Residual CNN step on the grid; ``cond`` (batch, 2) is broadcast."""
        c = self.cfg
        ones = tf.ones((tf.shape(g)[0], c.N3, c.N1, c.N2, 1))
        c0 = tf.reshape(cond[:, 0], (-1, 1, 1, 1, 1))
        c1 = tf.reshape(cond[:, 1], (-1, 1, 1, 1, 1))
        h = tf.concat([g, ones * c0, ones * c1], axis=-1)
        for j, (w, b) in enumerate(self._conv[k]):
            h = tf.nn.conv3d(_circular_pad3d(tf, h), w, [1] * 5, "VALID") + b
            if j < 2:
                h = tf.nn.relu(h)
        return g + self.gamma[k] * h

    # --------------------------------------------------------------- forward
    def __call__(self, ur, ui, mask, cond, training: bool = False):
        """Reconstruct angle-delay vectors.

        ``ur, ui``: (batch, m_max) *physical* (destandardized) measurements,
        zero beyond the report; ``mask``: (batch, m_max) 1.0 for received
        entries; ``cond``: (batch, 2) = (m / m_max, bits / bits_max).  Returns
        the list of iterates ``[(xr, xi), ...]`` (last = final), each
        (batch, D).  ``A`` has orthonormal rows, so ``A^H(y - A x)`` is the
        exact data-consistency projection.
        """
        tf = _tf()
        ur, ui = ur * mask, ui * mask
        xr, xi = self._adjoint(tf, ur, ui)
        out = []
        for k in range(self.cfg.iterations):
            yr, yi = self._measure(tf, xr, xi)
            rr, ri = (ur - yr) * mask, (ui - yi) * mask
            dr, di = self._adjoint(tf, rr, ri)
            xr, xi = xr + self.alpha[k] * dr, xi + self.alpha[k] * di
            g = self._prior(tf, k, self._to_grid(tf, xr, xi), cond)
            xr, xi = self._from_grid(tf, g)
            out.append((xr, xi))
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path: str | pathlib.Path) -> None:
        """Write ``<path>.npz`` (weights + bundled KLT basis) + ``<path>.json``.

        The published constant (``A``, ``sigma``) is bundled with the weights so
        a gNB loads the whole scheme from one file; the UE-side codec is saved
        separately by the trainer for publication.
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {v.name.rstrip(":0"): v.numpy() for v in self.variables}
        arrays["_klt_A"] = self.codec.A
        arrays["_klt_sigma"] = self.codec._sigma
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        path.with_suffix(".json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "UnrolledDecoder":
        path = pathlib.Path(path)
        cfg = DecoderConfig(**json.loads(path.with_suffix(".json").read_text()))
        with np.load(path.with_suffix(".npz")) as data:
            codec = GlimpseCodec(
                AntennaConfig.standard(cfg.N1, cfg.N2), cfg.N3, m_max=cfg.m_max,
                basis_matrix=data["_klt_A"], sigma=data["_klt_sigma"],
            )
            dec = cls(cfg, codec=codec)
            for v in dec.variables:
                v.assign(data[v.name.rstrip(":0")])
        return dec


class KerasDecoder:
    """Adapter: trained :class:`UnrolledDecoder` as a ``GlimpseDecoder``.

    Satisfies the ``scheme.GlimpseDecoder`` protocol (NumPy in / NumPy out),
    so a :class:`~nr_csi.ml.scheme.GlimpseScheme` built on it drops straight
    into ``nr_csi.eval.evaluate`` next to the 3GPP codebooks.
    """

    name = "learned"

    def __init__(self, model: UnrolledDecoder) -> None:
        tf = _tf()
        self.model = model
        self._fn = tf.function(
            lambda ur, ui, mask, cond: model(ur, ui, mask, cond)[-1],
            reduce_retracing=True,
        )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "KerasDecoder":
        return cls(UnrolledDecoder.load(path))

    def __call__(self, u: np.ndarray, m: int, bits: int = 3) -> np.ndarray:
        tf = _tf()
        cfg = self.model.cfg
        u = np.atleast_2d(u)
        pad = np.zeros((u.shape[0], cfg.m_max), dtype=complex)
        pad[:, :m] = u[:, :m]
        mask = np.zeros((u.shape[0], cfg.m_max), np.float32)
        mask[:, :m] = 1.0
        cond = np.tile([m / cfg.m_max, bits / cfg.bits_max], (u.shape[0], 1))
        xr, xi = self._fn(
            tf.constant(pad.real, tf.float32), tf.constant(pad.imag, tf.float32),
            tf.constant(mask), tf.constant(cond, tf.float32),
        )
        return (xr.numpy() + 1j * xi.numpy()).astype(complex)


def quantize_tf(u, bits_choice, table_bits=(2, 3, 4, 5)):
    """In-graph Lloyd-Max quantize/dequantize: ``u`` (batch, n) float32,
    ``bits_choice`` (batch,) int32 drawn from ``table_bits``.

    Used only in training (the deployed UE quantizes with
    :mod:`.quantizer`; the tables are identical constants).
    """
    tf = _tf()
    outs = []
    flat = tf.reshape(u, (1, -1))
    for b in table_bits:
        levels, bounds = lloyd_max(b)
        idx = tf.searchsorted(tf.constant(bounds[None, :], tf.float32), flat)
        vals = tf.gather(tf.constant(levels, tf.float32), idx[0])
        outs.append(tf.reshape(vals, tf.shape(u)))
    sel = tf.stack(outs, axis=0)  # (n_tables, batch, n)
    table_idx = tf.argmax(
        tf.cast(tf.equal(bits_choice[None, :], tf.constant(table_bits, bits_choice.dtype)[:, None]),
                tf.int32), axis=0)  # (batch,)
    return tf.gather(tf.transpose(sel, (1, 0, 2)), table_idx, axis=1, batch_dims=1)
