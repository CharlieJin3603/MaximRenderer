#!/usr/bin/env python3
"""
Neural BRDF trainer — anisotropic GGX brushed metal.

Architecture: Fourier feature embedding → MLP → log-space RGB
Loss:         MSE in log space (handles the specular peak's wide dynamic range)
At inference: BRDF = exp(model(x))

Input parameterization: (theta_H, phi_H, theta_V, phi_V) in half-angle space.
H = normalize(L+V), so the GGX specular peak aligns with theta_H ≈ 0 — a single
axis that the Fourier embedding can represent directly, rather than forcing the
network to discover the half-vector relationship from raw L/V angles.

Usage:
    python3 train_brdf.py                       # default: 3-layer, dim=64, GELU
    python3 train_brdf.py --layers 4 --dim 64   # larger model
    python3 train_brdf.py --n-freqs 24          # more Fourier frequencies
    python3 train_brdf.py --ablate              # run all 8 ablation configs
"""

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# ---------------------------------------------------------------------------
# Paths & hyperparameters
# ---------------------------------------------------------------------------
DATASET_PATH    = Path(__file__).parent / "brdf_dataset.npy"
CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"

# Additive offset before log transform: log(brdf + LOG_EPS).
# 1e-3 keeps near-zero values finite without biasing the specular peak.
LOG_EPS = 1e-3

# Input normalization: scale each channel to [0, 1]
#   theta in [0, pi/2]  →  multiply by 2/pi
#   phi   in [0, 2*pi]  →  multiply by 1/(2*pi)
INPUT_SCALE = np.array([2.0 / np.pi, 1.0 / (2.0 * np.pi),
                         2.0 / np.pi, 1.0 / (2.0 * np.pi)], dtype=np.float32)

VAL_FRACTION  = 0.10    # 5k validation samples from 50k total
BATCH_SIZE    = 512
N_EPOCHS      = 500
WARMUP_STEPS  = 500     # linear LR warmup before cosine decay
LR_PEAK       = 3e-4
LR_MIN        = 1e-5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    """Positional encoding: x ∈ [0,1]^4 → [sin(2^k π x), cos(2^k π x)]_{k=0..L-1}.
    Output dim: 4 * 2 * n_freqs. For n_freqs=8 this is 64.
    """
    def __init__(self, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        # Store as Python list — avoids being registered as a trainable parameter.
        self._freq_vals = [2.0 ** k * np.pi for k in range(n_freqs)]

    def __call__(self, x: mx.array) -> mx.array:
        freqs = mx.array(self._freq_vals, dtype=mx.float32)    # (n_freqs,)
        xf    = x[..., None] * freqs                            # (B, 4, n_freqs)
        enc   = mx.concatenate([mx.sin(xf), mx.cos(xf)], axis=-1)  # (B, 4, 2*n_freqs)
        return enc.reshape(x.shape[0], -1)                     # (B, 4 * 2 * n_freqs)


class NeuralBRDF(nn.Module):
    """MLP BRDF approximator.
    Input:  normalized (theta_H, phi_H, theta_V, phi_V), shape (B, 4)
    Output: log-space (R, G, B),                         shape (B, 3)
    """
    def __init__(self, n_layers: int = 3, hidden_dim: int = 64,
                 n_freqs: int = 8, activation: str = "gelu"):
        super().__init__()
        self.embedding  = FourierEmbedding(n_freqs)
        input_dim       = 4 * 2 * n_freqs
        act_fn          = nn.GELU if activation == "gelu" else nn.ReLU

        # Build layer list: [Linear, Act, Linear, Act, ..., Linear]
        # n_layers controls the number of linear layers total.
        layers = [nn.Linear(input_dim, hidden_dim), act_fn()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fn()]
        layers.append(nn.Linear(hidden_dim, 3))

        self.net        = layers
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.n_freqs    = n_freqs

    def __call__(self, x: mx.array) -> mx.array:
        x = self.embedding(x)
        for layer in self.net:
            x = layer(x)
        return x   # log-space BRDF, shape (B, 3)


def count_params(model: nn.Module) -> int:
    def _count(obj) -> int:
        if isinstance(obj, mx.array):
            return obj.size
        if isinstance(obj, dict):
            return sum(_count(v) for v in obj.values())
        if isinstance(obj, list):
            return sum(_count(v) for v in obj)
        return 0
    return _count(model.trainable_parameters())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset():
    """Returns (train_x, train_y, val_x, val_y) as numpy float32 arrays."""
    data = np.load(DATASET_PATH).astype(np.float32)       # (50000, 7)

    inputs  = data[:, :4] * INPUT_SCALE                    # (50000, 4) in [0, 1]
    targets = np.log(data[:, 4:] + LOG_EPS)               # (50000, 3) log-space

    rng     = np.random.default_rng(0)
    idx     = rng.permutation(len(data))
    n_val   = int(len(data) * VAL_FRACTION)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    return (inputs[train_idx], targets[train_idx],
            inputs[val_idx],   targets[val_idx])


# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------

def loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    pred = model(x)
    diff = pred - y
    return mx.mean(diff * diff)


def mse_loss_val(pred: mx.array, target: mx.array) -> mx.array:
    diff = pred - target
    return mx.mean(diff * diff)


def psnr_db(log_space_mse: float) -> float:
    """
    PSNR from log-space MSE.
    30 dB ↔ MSE < 1e-3 (mean squared log error < 0.032 log units, ~3% relative).
    """
    if log_space_mse <= 0.0:
        return float("inf")
    return -10.0 * np.log10(log_space_mse)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup then cosine decay
# ---------------------------------------------------------------------------

def lr_at_step(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return LR_PEAK * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
    return LR_MIN + (LR_PEAK - LR_MIN) * cosine


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict, run_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"  layers={config['n_layers']}  dim={config['hidden_dim']}  "
          f"act={config['activation']}  n_freqs={config['n_freqs']}")
    print(f"{'='*60}")

    train_x_np, train_y_np, val_x_np, val_y_np = load_dataset()
    n_train = len(train_x_np)
    print(f"  Dataset: {n_train:,} train / {len(val_x_np):,} val")

    # Convert validation set to MLX once (it's small)
    val_x = mx.array(val_x_np)
    val_y = mx.array(val_y_np)

    model = NeuralBRDF(
        n_layers=config["n_layers"],
        hidden_dim=config["hidden_dim"],
        n_freqs=config["n_freqs"],
        activation=config["activation"],
    )
    print(f"  Parameters: {count_params(model):,}")

    optimizer        = optim.Adam(learning_rate=LR_PEAK)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    steps_per_epoch = n_train // BATCH_SIZE
    total_steps     = N_EPOCHS * steps_per_epoch

    rng          = np.random.default_rng(42)
    best_val     = float("inf")
    history      = {"train_loss": [], "val_loss": [], "val_psnr": []}
    step         = 0
    t0           = time.time()

    checkpoint_dir = CHECKPOINT_DIR / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(N_EPOCHS):
        perm       = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_train - BATCH_SIZE + 1, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            bx  = mx.array(train_x_np[idx])
            by  = mx.array(train_y_np[idx])

            optimizer.learning_rate = lr_at_step(step, total_steps)

            loss, grads = loss_and_grad_fn(model, bx, by)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += float(loss)
            n_batches  += 1
            step       += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # Validation pass
        val_pred = model(val_x)
        val_loss = float(mse_loss_val(val_pred, val_y))
        mx.eval(val_loss)
        val_psnr = psnr_db(val_loss)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)

        if val_loss < best_val:
            best_val  = val_loss
            model.save_weights(str(checkpoint_dir / "best.npz"))

        if (epoch + 1) % 20 == 0 or epoch == 0:
            lr      = optimizer.learning_rate
            elapsed = time.time() - t0
            print(f"  epoch {epoch+1:4d}/{N_EPOCHS}  "
                  f"train={avg_train:.5f}  val={val_loss:.5f}  "
                  f"PSNR={val_psnr:.1f}dB  lr={lr:.1e}  ({elapsed:.0f}s)")

    model.save_weights(str(checkpoint_dir / "final.npz"))

    best_epoch = int(np.argmin(history["val_loss"]))
    summary = {
        "run":       run_name,
        "config":    config,
        "best_epoch":     best_epoch + 1,
        "best_val_loss":  history["val_loss"][best_epoch],
        "best_val_psnr":  history["val_psnr"][best_epoch],
    }
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump({**history, **summary}, f, indent=2)

    print(f"\n  Best: epoch {summary['best_epoch']}  "
          f"val={summary['best_val_loss']:.5f}  "
          f"PSNR={summary['best_val_psnr']:.1f}dB")
    print(f"  Saved: {checkpoint_dir / 'best.npz'}")

    # Warn if below the 30 dB target from the project plan
    if summary["best_val_psnr"] < 30.0:
        print(f"\n  *** PSNR {summary['best_val_psnr']:.1f}dB < 30dB target ***")
        print(f"  Consider: --n-freqs 24  or adding a skip connection.")

    return summary


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    {"n_layers": nl, "hidden_dim": dim, "activation": act, "n_freqs": 16}
    for nl  in [3, 4]
    for dim in [32, 64]
    for act in ["relu", "gelu"]
]


def run_ablation():
    print(f"Running {len(ABLATION_CONFIGS)} ablation configs...")
    results = []
    for cfg in ABLATION_CONFIGS:
        name   = f"L{cfg['n_layers']}_D{cfg['hidden_dim']}_{cfg['activation']}"
        result = train(cfg, name)
        results.append(result)

    results.sort(key=lambda r: r["best_val_loss"])

    print("\n" + "="*60)
    print(f"{'Run':<25} {'Val loss':>10} {'PSNR':>8}  Best epoch")
    print("-"*60)
    for r in results:
        marker = " ← best" if r is results[0] else ""
        print(f"  {r['run']:<23} {r['best_val_loss']:>10.5f} "
              f"{r['best_val_psnr']:>7.1f}dB  {r['best_epoch']:>4}{marker}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_DIR / "ablation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: {CHECKPOINT_DIR / 'ablation_summary.json'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train neural BRDF approximator")
    p.add_argument("--layers",     type=int,   default=3,
                   help="Number of MLP layers (default: 3)")
    p.add_argument("--dim",        type=int,   default=64,
                   help="Hidden layer width (default: 64)")
    p.add_argument("--activation", type=str,   default="gelu",
                   choices=["gelu", "relu"])
    p.add_argument("--n-freqs",    type=int,   default=16,
                   help="Fourier frequency bands per input dim (default: 16)")
    p.add_argument("--ablate",     action="store_true",
                   help="Run all 8 ablation configs (3/4 layers × 32/64 dim × relu/gelu)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.ablate:
        run_ablation()
    else:
        config = {
            "n_layers":   args.layers,
            "hidden_dim": args.dim,
            "activation": args.activation,
            "n_freqs":    args.n_freqs,
        }
        train(config, f"L{args.layers}_D{args.dim}_{args.activation}")


if __name__ == "__main__":
    main()
