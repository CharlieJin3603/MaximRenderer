#!/usr/bin/env python3
"""
Anisotropic GGX BRDF dataset generator for brushed metal material.

Evaluates the exact BRDF from Shaders.metal analytically to produce
noiseless ground truth. No path tracer required.

Output: brdf_dataset.npy  — float32 array, shape (50000, 7)
        brdf_dataset.csv  — same data, human-readable
Columns: theta_H, phi_H, theta_V, phi_V, R, G, B

Half-angle parameterization: inputs are the angles of H=normalize(L+V) and V in
local TBN space. The specular peak aligns with theta_H ≈ 0, making it a single
axis the Fourier embedding can represent directly.
"""

import numpy as np
import os

# ---------------------------------------------------------------------------
# Material constants — must match Shaders.metal lines 462-464
# ---------------------------------------------------------------------------
F0 = np.array([0.95, 0.93, 0.88], dtype=np.float32)
AX = 0.05   # roughness along brush direction T (sharp specular streak)
AY = 0.40   # roughness around tube B          (broad specular falloff)

N_UNIFORM    = 20_000   # uniform hemisphere pairs — global coverage
N_IMPORTANCE = 30_000   # GGX-importance-sampled  — specular lobe coverage
SEED         = 42


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def sph_to_cart(theta, phi):
    """Spherical → Cartesian in local TBN space: T=(1,0,0), N=(0,1,0), B=(0,0,1)."""
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), np.cos(theta), st * np.sin(phi)], axis=-1)


def cart_to_sph(v):
    """Unit vector → (theta, phi) in local TBN space. phi in [0, 2π)."""
    theta = np.arccos(np.clip(v[..., 1], -1.0, 1.0))
    phi   = np.arctan2(v[..., 2], v[..., 0]) % (2.0 * np.pi)
    return theta, phi


# ---------------------------------------------------------------------------
# BRDF terms — direct translations of Shaders.metal
# ---------------------------------------------------------------------------

def F_Schlick(VdotH):
    """Schlick Fresnel. VdotH: (...,) → returns (..., 3)."""
    f = (1.0 - VdotH)[..., np.newaxis]
    return F0 + (1.0 - F0) * (f ** 5)


def D_GGX_aniso(H):
    """Anisotropic GGX NDF (Burley 2012). H in local TBN space, shape (..., 3)."""
    HdotT = H[..., 0]
    HdotB = H[..., 2]
    HdotN = H[..., 1]
    d = HdotT**2 / AX**2 + HdotB**2 / AY**2 + HdotN**2
    return 1.0 / (np.pi * AX * AY * d**2 + 1e-10)


def G2_GGX_aniso(V, L):
    """Height-correlated Smith masking-shadowing (Heitz 2014). V, L in local TBN."""
    NdotV = np.maximum(V[..., 1], 1e-4)
    NdotL = np.maximum(L[..., 1], 1e-4)
    a2V = (V[..., 0] * AX)**2 + (V[..., 2] * AY)**2
    a2L = (L[..., 0] * AX)**2 + (L[..., 2] * AY)**2
    LambdaV = 0.5 * (-1.0 + np.sqrt(1.0 + a2V / NdotV**2))
    LambdaL = 0.5 * (-1.0 + np.sqrt(1.0 + a2L / NdotL**2))
    return 1.0 / (1.0 + LambdaV + LambdaL)


def eval_brdf(L, V):
    """
    Evaluate the full anisotropic GGX BRDF for a batch of (L, V) pairs.
    L, V: (..., 3) in local TBN space.
    Returns (..., 3) float32. Backfacing pairs return 0.
    """
    NdotV = np.maximum(V[..., 1], 1e-4)
    NdotL = np.maximum(L[..., 1], 1e-4)

    H     = L + V
    H     = H / np.maximum(np.linalg.norm(H, axis=-1, keepdims=True), 1e-8)
    VdotH = np.maximum((V * H).sum(axis=-1), 0.0)

    F  = F_Schlick(VdotH)                                        # (..., 3)
    D  = D_GGX_aniso(H)                                          # (...,)
    G2 = G2_GGX_aniso(V, L)                                      # (...,)
    denom = np.maximum(4.0 * NdotV * NdotL, 1e-6)
    brdf  = F * (D * G2 / denom)[..., np.newaxis]

    valid = (L[..., 1] > 0.0) & (V[..., 1] > 0.0)
    return (brdf * valid[..., np.newaxis]).astype(np.float32)


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def sample_uniform_hemisphere_pair(rng, n):
    """Sample n (L, V) pairs uniformly on the upper hemisphere.
    Returns angles in half-angle parameterization: (theta_H, phi_H, theta_V, phi_V).
    """
    u = rng.random((n, 4), dtype=np.float32)
    theta_l = np.arccos(np.sqrt(1.0 - u[:, 0]))
    phi_l   = 2.0 * np.pi * u[:, 1]
    theta_v = np.arccos(np.sqrt(1.0 - u[:, 2]))
    phi_v   = 2.0 * np.pi * u[:, 3]
    L = sph_to_cart(theta_l, phi_l)
    V = sph_to_cart(theta_v, phi_v)
    H = (L + V) / np.maximum(np.linalg.norm(L + V, axis=-1, keepdims=True), 1e-8)
    theta_H, phi_H = cart_to_sph(H)
    theta_V, phi_V = cart_to_sph(V)
    return theta_H, phi_H, theta_V, phi_V, L, V


def sample_ggx_H(rng, n):
    """Importance-sample H from the anisotropic GGX NDF.
    Matches sampleGGXAniso() in Shaders.metal exactly.
    """
    u = rng.random((n, 2), dtype=np.float32)
    phi   = np.arctan2(AY * np.sin(2.0 * np.pi * u[:, 0]),
                       AX * np.cos(2.0 * np.pi * u[:, 0]))
    cp    = np.cos(phi);  sp = np.sin(phi)
    a2    = 1.0 / np.maximum(cp**2 / AX**2 + sp**2 / AY**2, 1e-8)
    tan2  = a2 * u[:, 1] / np.maximum(1.0 - u[:, 1], 1e-6)
    cosT  = 1.0 / np.sqrt(1.0 + tan2)
    sinT  = np.sqrt(np.maximum(0.0, 1.0 - cosT**2))
    # T=(1,0,0), B=(0,0,1), N=(0,1,0) in local space
    return np.stack([sinT * cp, cosT, sinT * sp], axis=-1)


def sample_importance_pairs(rng, n):
    """Generate n (L, V) pairs concentrated near the specular lobe.
    Strategy: sample V uniformly, sample H from GGX NDF, reflect V through H to get L.
    H is already available directly, so theta_H/phi_H come straight from it — no
    need to reconstruct H from L afterward.
    """
    collected = []
    total = 0
    batch = max(n * 4, 10_000)   # oversample: ~50% discard rate expected

    while total < n:
        u_v = rng.random((batch, 2), dtype=np.float32)
        theta_v = np.arccos(np.sqrt(1.0 - u_v[:, 0]))
        phi_v   = 2.0 * np.pi * u_v[:, 1]
        V = sph_to_cart(theta_v, phi_v)

        H     = sample_ggx_H(rng, batch)
        VdotH = (V * H).sum(axis=-1, keepdims=True)
        L     = 2.0 * VdotH * H - V

        valid = (L[:, 1] > 0.0) & (VdotH[:, 0] > 0.0)
        H = H[valid]
        L = L[valid]
        V = V[valid]

        theta_H, phi_H = cart_to_sph(H)
        theta_V, phi_V = cart_to_sph(V)

        collected.append((theta_H, phi_H, theta_V, phi_V, L, V))
        total += len(L)

    tH = np.concatenate([c[0] for c in collected])[:n]
    pH = np.concatenate([c[1] for c in collected])[:n]
    tV = np.concatenate([c[2] for c in collected])[:n]
    pV = np.concatenate([c[3] for c in collected])[:n]
    Lc = np.concatenate([c[4] for c in collected])[:n]
    Vc = np.concatenate([c[5] for c in collected])[:n]
    return tH, pH, tV, pV, Lc, Vc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Generating {N_UNIFORM:,} uniform samples...")
    tH_u, pH_u, tV_u, pV_u, L_u, V_u = sample_uniform_hemisphere_pair(rng, N_UNIFORM)
    brdf_u = eval_brdf(L_u, V_u)

    print(f"Generating {N_IMPORTANCE:,} GGX-importance-sampled samples...")
    tH_i, pH_i, tV_i, pV_i, L_i, V_i = sample_importance_pairs(rng, N_IMPORTANCE)
    brdf_i = eval_brdf(L_i, V_i)

    theta_H = np.concatenate([tH_u, tH_i]).astype(np.float32)
    phi_H   = np.concatenate([pH_u, pH_i]).astype(np.float32)
    theta_V = np.concatenate([tV_u, tV_i]).astype(np.float32)
    phi_V   = np.concatenate([pV_u, pV_i]).astype(np.float32)
    brdf    = np.concatenate([brdf_u, brdf_i])

    idx = rng.permutation(len(theta_H))
    theta_H, phi_H, theta_V, phi_V, brdf = (
        theta_H[idx], phi_H[idx], theta_V[idx], phi_V[idx], brdf[idx]
    )

    data = np.column_stack([theta_H, phi_H, theta_V, phi_V, brdf])

    npy_path = os.path.join(out_dir, "brdf_dataset.npy")
    csv_path = os.path.join(out_dir, "brdf_dataset.csv")
    np.save(npy_path, data)
    np.savetxt(csv_path, data, delimiter=",",
               header="theta_H,phi_H,theta_V,phi_V,R,G,B", comments="", fmt="%.6f")

    print(f"\nSaved:")
    print(f"  {npy_path}  ({data.nbytes / 1024:.0f} KB)")
    print(f"  {csv_path}")
    print(f"\nDataset stats ({len(data):,} samples):")
    for i, ch in enumerate("RGB"):
        col = brdf[:, i]
        print(f"  BRDF {ch}: min={col.min():.4f}  max={col.max():.2f}  "
              f"mean={col.mean():.4f}  p99={np.percentile(col, 99):.3f}")
    nz = (brdf[:, 0] > 1e-6).sum()
    print(f"  Non-zero: {nz:,} / {len(data):,}  ({100*nz/len(data):.1f}%)")


if __name__ == "__main__":
    main()
