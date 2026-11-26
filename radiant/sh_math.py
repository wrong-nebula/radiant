"""Exact Order-1 Spherical Harmonics conventions, Frisvad tangent frames, and evaluation metrics.

Convention (shared verbatim across Warp tracer, PyTorch model, and Three.js WGSL/TSL):
For surface patch i with orthonormal local frame (T_i, B_i, N_i):
  I0   = E_{w ~ cos(theta)/pi}[ L_ind(w) ]                 (RGB, >= 0, diffuse irradiance / pi)
  I1_T = E_{w ~ cos(theta)/pi}[ L_ind(w) * (w . T_i) ]     (RGB, in [-I0, I0])
  I1_B = E_{w ~ cos(theta)/pi}[ L_ind(w) * (w . B_i) ]     (RGB, in [-I0, I0])
  I1_N = E_{w ~ cos(theta)/pi}[ L_ind(w) * (w . N_i) ]     (RGB, in [0, I0])

Packed layout per patch (12 floats):
  [I0_R, I0_G, I0_B, IT_R, IT_G, IT_B, IB_R, IB_G, IB_B, IN_R, IN_G, IN_B]
"""

from __future__ import annotations

import numpy as np

# Fixed physical scale normalization constant (meters) for SE(3) Form-Factor Q*K
POS_SCALE_METERS = 4.0


def build_frisvad_basis_np(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duff/Frisvad branchless orthonormal tangent frame (T, B, N) from unit normals [..., 3]."""
    n = np.asarray(normals, dtype=np.float32)
    n_len = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(n_len, 1e-8)

    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    sign = np.where(nz >= 0.0, 1.0, -1.0).astype(np.float32)
    a = -1.0 / (sign + nz)
    b = nx * ny * a

    t = np.stack(
        [
            1.0 + sign * nx * nx * a,
            sign * b,
            -sign * nx,
        ],
        axis=-1,
    ).astype(np.float32)
    b_vec = np.stack(
        [
            b,
            sign + ny * ny * a,
            -ny,
        ],
        axis=-1,
    ).astype(np.float32)

    # Ensure exact unit length
    t = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-8)
    b_vec = np.cross(n, t).astype(np.float32)
    b_vec = b_vec / np.maximum(np.linalg.norm(b_vec, axis=-1, keepdims=True), 1e-8)
    return t, b_vec, n


def local_sh_to_world_sh(
    sh_local: np.ndarray,
    t: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local SH [..., 12] to (I0 [..., 3], I1_world [..., 3, 3] as [..., rgb, xyz])."""
    i0 = sh_local[..., 0:3]
    it = sh_local[..., 3:6]
    ib = sh_local[..., 6:9]
    in_ = sh_local[..., 9:12]
    # [..., 3_rgb, 1] * [..., 1, 3_xyz] -> [..., 3_rgb, 3_xyz]
    i1_world = (
        it[..., :, None] * t[..., None, :]
        + ib[..., :, None] * b[..., None, :]
        + in_[..., :, None] * n[..., None, :]
    )
    return i0, i1_world


def tonemap_reinhard_srgb(irradiance: np.ndarray) -> np.ndarray:
    """Smooth HDR->LDR tone mapping (Reinhard + gamma 1/2.2) in [0, 1]."""
    x = np.maximum(np.asarray(irradiance, dtype=np.float64), 0.0)
    tm = x / (1.0 + x)
    return np.power(np.clip(tm, 0.0, 1.0), 1.0 / 2.2)


def tonemapped_psnr_db(pred_i0: np.ndarray, gt_i0: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute tone-mapped indirect irradiance PSNR in dB."""
    p = tonemap_reinhard_srgb(pred_i0)
    g = tonemap_reinhard_srgb(gt_i0)
    diff2 = (p - g) ** 2
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        diff2 = diff2[m]
    mse = float(np.mean(diff2))
    if mse <= 1e-12:
        return 60.0
    return float(10.0 * np.log10(1.0 / mse))


def color_bleed_correlation(
    pred_i0: np.ndarray,
    gt_i0: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Pearson correlation of tone-mapped indirect RGB across patches (captures bounce & color bleed)."""
    p = tonemap_reinhard_srgb(pred_i0)
    g = tonemap_reinhard_srgb(gt_i0)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        p = p[m]
        g = g[m]
    pf = p.reshape(-1)
    gf = g.reshape(-1)
    pf = pf - pf.mean()
    gf = gf - gf.mean()
    denom = float(np.linalg.norm(pf) * np.linalg.norm(gf))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(pf, gf) / denom)


def sh_directional_cosine(
    pred_sh: np.ndarray,
    gt_sh: np.ndarray,
    mask: np.ndarray | None = None,
    min_luma: float = 0.02,
) -> float:
    """Mean cosine similarity of L1 directional vectors weighted on lit patches."""
    p_l1 = pred_sh[..., 3:12].reshape(*pred_sh.shape[:-1], 3, 3)  # [..., 3_dir, 3_rgb]
    g_l1 = gt_sh[..., 3:12].reshape(*gt_sh.shape[:-1], 3, 3)
    # Collapse to luminance directional vector [..., 3_dir]
    w_rgb = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    p_vec = np.tensordot(p_l1, w_rgb, axes=([-1], [0]))
    g_vec = np.tensordot(g_l1, w_rgb, axes=([-1], [0]))
    g_luma = np.tensordot(gt_sh[..., 0:3], w_rgb, axes=([-1], [0]))

    valid = g_luma > min_luma
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return 1.0
    pv = p_vec[valid]
    gv = g_vec[valid]
    dots = np.sum(pv * gv, axis=-1)
    norms = np.linalg.norm(pv, axis=-1) * np.linalg.norm(gv, axis=-1)
    return float(np.mean(dots / np.maximum(norms, 1e-8)))


def sh_rmse(pred_sh: np.ndarray, gt_sh: np.ndarray, mask: np.ndarray | None = None) -> float:
    """RMSE over full 12-float Order-1 SH vectors."""
    diff2 = (np.asarray(pred_sh, dtype=np.float64) - np.asarray(gt_sh, dtype=np.float64)) ** 2
    if mask is not None:
        diff2 = diff2[np.asarray(mask, dtype=bool)]
    return float(np.sqrt(np.mean(diff2)))
