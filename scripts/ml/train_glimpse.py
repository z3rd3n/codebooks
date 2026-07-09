"""Train the gNB-side GLIMPSE decoder (measurement-dropout, one model per array).

The UE side is fixed (seeded projection + Lloyd-Max quantizer), so training is
plain supervised inversion -- no encoder gradients, no straight-through
quantization tricks.  Every batch randomizes, per sample:

* the measurement count ``m ~ U{m_min..m_max}`` (measurement dropout): ONE
  decoder then serves every payload size / any truncated (rateless) report;
* the quantizer depth ``B ~ U{2,3,4,5}``;
* a global phase (the one symmetry the report does not pin down);
* the input variant: clean, or eigen targets computed from a noisy channel
  (the stored 20 / 10 dB variants), with the *clean* vector as the target --
  the decoder learns measurement-noise robustness;
* layer-1 or layer-2 eigen targets (rank-2 support).

Loss: 1 - SGCS per PMI frequency unit (exactly the harness metric), deep
supervision over the unrolled iterates.

    .venv/bin/python scripts/ml/train_glimpse.py \
        --data data/ml/targets_p32.npz --out models/glimpse_p32
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nr_csi.ml.decoder import DecoderConfig, UnrolledDecoder, quantize_tf  # noqa: E402

TABLE_BITS = (2, 3, 4, 5)


def load_rows(path: pathlib.Path, split: str, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Assemble (input, target) rows: clean, noisy-input, and layer-2 variants."""
    z = np.load(path)
    clean = z[f"{split}/g_clean_l1"]
    rows_in = [clean]
    rows_tgt = [clean]
    for s in meta["noise_snrs"]:
        rows_in.append(z[f"{split}/g_n{int(s)}_l1"])
        rows_tgt.append(clean)  # denoise toward the true-channel target
    l2 = z[f"{split}/g_clean_l2"]
    rows_in.append(l2)
    rows_tgt.append(l2)
    g_in = np.concatenate(rows_in)
    g_tgt = np.concatenate(rows_tgt)
    return g_in, g_tgt


def to_ri(g: np.ndarray) -> np.ndarray:
    return np.stack([g.real, g.imag], axis=-1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data/ml/targets_p32.npz"))
    ap.add_argument("--meta", type=pathlib.Path, default=None,
                    help="dataset meta JSON (default: infer N1/N2/N3 from --geometry)")
    ap.add_argument("--geometry", default="4x4x8", help="N1xN2xN3")
    ap.add_argument("--noise-snrs", default="20,10")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("models/glimpse_p32"))
    ap.add_argument("--basis", default="klt", choices=("klt", "random"),
                    help="fixed UE projection: KLT (headline) or random (ablation)")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--m-min", type=int, default=2)
    ap.add_argument("--m-max", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-steps", type=int, default=0, help="debug: steps per epoch cap")
    args = ap.parse_args()

    import tensorflow as tf

    tf.random.set_seed(args.seed)
    n1, n2, n3 = (int(x) for x in args.geometry.split("x"))
    meta = {"noise_snrs": [float(s) for s in args.noise_snrs.split(",")]}
    cfg = DecoderConfig(N1=n1, N2=n2, N3=n3, m_max=args.m_max, basis=args.basis,
                        iterations=args.iterations, hidden=args.hidden)

    g_in, g_tgt = load_rows(args.data, "train", meta)
    gv_in, gv_tgt = load_rows(args.data, "val", meta)
    print(f"train rows {g_in.shape[0]:,}  val rows {gv_in.shape[0]:,}  D={g_in.shape[1]}")

    # Fit / build the fixed UE codec (published constant), then bind the decoder
    # to it.  KLT is fitted on the *clean* layer-1 training eigenvectors only.
    # Targets are unit-norm, so E||g||^2 = 1 = sum of all KLT eigenvalues, and
    # sum(sigma[:m_max]^2) is directly the captured-energy fraction.
    from nr_csi.config import AntennaConfig
    from nr_csi.ml.projection import GlimpseCodec, fit_klt
    ant = AntennaConfig.standard(n1, n2)
    if args.basis == "klt":
        with np.load(args.data) as z:
            gk = z["train/g_clean_l1"]
            A, sigma = fit_klt(gk, args.m_max)
            total_energy = float(np.mean(np.sum(np.abs(gk) ** 2, axis=1)))
        codec = GlimpseCodec(ant, n3, m_max=args.m_max, basis_matrix=A, sigma=sigma)
        print(f"KLT codec fitted: top-{args.m_max} captured energy "
              f"{float(np.sum(sigma**2)) / total_energy:.4f}")
    else:
        codec = GlimpseCodec(ant, n3, m_max=args.m_max, seed=args.seed)
        print(f"random codec (seed {args.seed})")
    codec.save(f"{args.out}_codec")
    model = UnrolledDecoder(cfg, codec=codec)
    n_params = int(sum(np.prod(v.shape) for v in model.variables))
    print(f"decoder: K={cfg.iterations} hidden={cfg.hidden} params={n_params:,}")

    ds = (tf.data.Dataset.from_tensor_slices((to_ri(g_in), to_ri(g_tgt)))
          .shuffle(g_in.shape[0], seed=args.seed)
          .batch(args.batch, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE))

    steps_per_epoch = g_in.shape[0] // args.batch
    total = args.epochs * steps_per_epoch
    sched = tf.keras.optimizers.schedules.CosineDecay(args.lr, total, alpha=1e-2)
    opt = tf.keras.optimizers.Adam(sched, global_clipnorm=1.0)

    # ---- constants for the in-graph pipeline
    m_max = cfg.m_max
    F = np.fft.fft(np.eye(n3), norm="ortho")  # delay -> frequency (from_delay)
    Fc = tf.constant(F, tf.complex64)
    w = 0.5 ** np.arange(cfg.iterations - 1, -1, -1)
    sup_w = tf.constant(w / w.sum(), tf.float32)

    def to_complex(x):  # (b, D, 2) -> (b, P, N3) complex
        c = tf.complex(x[..., 0], x[..., 1])
        return tf.reshape(c, (-1, cfg.P, n3))

    def sgcs_loss(x, v_tgt):
        v = tf.matmul(to_complex(x), Fc)  # (b, P, N3) beam-frequency
        num = tf.abs(tf.reduce_sum(tf.math.conj(v) * v_tgt, axis=1)) ** 2
        den = (tf.reduce_sum(tf.abs(v) ** 2, axis=1)
               * tf.reduce_sum(tf.abs(v_tgt) ** 2, axis=1))
        return 1.0 - tf.reduce_mean(num / tf.maximum(den, 1e-12))

    sigma = model.sigma  # (m_max,) per-coordinate std (published constant)

    def encode(g_ri, m, bits):
        """Fixed UE pipeline, in-graph: project, standardize, quantize, then
        destandardize to the physical measurements the decoder inverts."""
        yr, yi = model._measure(tf, g_ri[..., 0], g_ri[..., 1])  # physical y
        mask = tf.sequence_mask(m, m_max, tf.float32)
        ur = quantize_tf(yr / sigma, bits, TABLE_BITS) * sigma * mask  # y_hat
        ui = quantize_tf(yi / sigma, bits, TABLE_BITS) * sigma * mask
        cond = tf.stack([tf.cast(m, tf.float32) / m_max,
                         tf.cast(bits, tf.float32) / cfg.bits_max], axis=1)
        return ur, ui, mask, cond

    @tf.function(reduce_retracing=True)
    def train_step(g_in_ri, g_tgt_ri):
        b = tf.shape(g_in_ri)[0]
        theta = tf.random.uniform((b, 1), 0, 2 * np.pi)
        cos, sin = tf.cos(theta), tf.sin(theta)

        def rot(x):  # global phase augmentation (applied to input & target alike)
            return tf.stack([x[..., 0] * cos - x[..., 1] * sin,
                             x[..., 0] * sin + x[..., 1] * cos], axis=-1)

        g_in_r, g_tgt_r = rot(g_in_ri), rot(g_tgt_ri)
        m = tf.random.uniform((b,), args.m_min, m_max + 1, tf.int32)
        bits = tf.gather(tf.constant(TABLE_BITS, tf.int32),
                         tf.random.uniform((b,), 0, len(TABLE_BITS), tf.int32))
        ur, ui, mask, cond = encode(g_in_r, m, bits)
        v_tgt = tf.matmul(to_complex(g_tgt_r), Fc)
        with tf.GradientTape() as tape:
            iterates = model(ur, ui, mask, cond, training=True)
            losses = tf.stack([sgcs_loss(tf.stack(x, -1), v_tgt) for x in iterates])
            loss = tf.reduce_sum(sup_w * losses)
        grads = tape.gradient(loss, model.variables)
        opt.apply_gradients(zip(grads, model.variables))
        return losses[-1]

    @tf.function(reduce_retracing=True)
    def val_step(g_in_ri, g_tgt_ri, m_fix, bits_fix):
        b = tf.shape(g_in_ri)[0]
        m = tf.fill((b,), m_fix)
        bits = tf.fill((b,), bits_fix)
        ur, ui, mask, cond = encode(g_in_ri, m, bits)
        xr, xi = model(ur, ui, mask, cond)[-1]
        v_tgt = tf.matmul(to_complex(g_tgt_ri), Fc)
        return 1.0 - sgcs_loss(tf.stack([xr, xi], -1), v_tgt)

    val_grid = [(8, 3), (16, 3), (28, 3), (40, 2)]
    gv_in_ri, gv_tgt_ri = to_ri(gv_in), to_ri(gv_tgt)
    n_val = min(2048, gv_in_ri.shape[0])

    def validate() -> dict[str, float]:
        out = {}
        for m_fix, b_fix in val_grid:
            vals = [val_step(gv_in_ri[i:i + 512], gv_tgt_ri[i:i + 512],
                             tf.constant(m_fix, tf.int32), tf.constant(b_fix, tf.int32))
                    for i in range(0, n_val, 512)]
            out[f"sgcs_m{m_fix}_b{b_fix}"] = float(np.mean([float(v) for v in vals]))
        return out

    history = []
    best = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        t0 = time.time()
        losses = []
        for step, (bi, bt) in enumerate(ds):
            losses.append(float(train_step(bi, bt)))
            if args.limit_steps and step + 1 >= args.limit_steps:
                break
        metrics = validate()
        mean_v = float(np.mean(list(metrics.values())))
        row = {"epoch": epoch, "train_1msgcs": float(np.mean(losses)),
               "val_mean_sgcs": mean_v, **metrics, "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(json.dumps(row))
        if mean_v > best:
            best = mean_v
            model.save(args.out)
    (args.out.parent / f"{args.out.name}_history.json").write_text(
        json.dumps({"config": vars(args) | {"out": str(args.out), "data": str(args.data),
                                            "meta": None},
                    "params": n_params, "history": history}, indent=2, default=str))
    print(f"best val mean SGCS {best:.4f} -> {args.out}.npz")


if __name__ == "__main__":
    main()
