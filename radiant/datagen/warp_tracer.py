"""Physically exact 3-bounce Cosine-Weighted Hemisphere GPU Patch Path Tracer in NVIDIA Warp.

Computes converged 3-bounce INDIRECT incoming irradiance as Order-1 Spherical Harmonics
(I0, I1_T, I1_B, I1_N x RGB = 12 floats/patch) in each patch's local tangent frame (T, B, N).

CRITICAL PHYSICS CONVENTION:
  - Bounce 0 (direct hit on an emissive surface) is EXPLICITLY EXCLUDED from indirect GI
    so Three.js direct lighting + shadows compose cleanly with zero double-counting.
  - Bounces 1, 2, 3 accumulate exact multi-bounce indirect radiance using cosine-weighted
    hemisphere importance sampling (Lambertian diffuse + energy-conserving metallic glossy).
  - Rays are stratified uniformly over each patch's actual surface triangles (area-weighted)
    with a small normal epsilon offset to avoid self-intersection on curved geometry.
"""

from __future__ import annotations

import numpy as np
import warp as wp

PI = 3.14159265358979323846


@wp.func
def _orthonormal_basis_from_normal(n: wp.vec3):
    sign = 1.0
    if n[2] < 0.0:
        sign = -1.0
    a = -1.0 / (sign + n[2])
    b = n[0] * n[1] * a
    t = wp.normalize(wp.vec3(1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]))
    bv = wp.normalize(wp.cross(n, t))
    return t, bv


@wp.func
def _sample_cosine_hemisphere(u1: float, u2: float, t: wp.vec3, b: wp.vec3, n: wp.vec3) -> wp.vec3:
    r = wp.sqrt(wp.clamp(u1, 0.0, 1.0))
    phi = 2.0 * PI * u2
    x = r * wp.cos(phi)
    y = r * wp.sin(phi)
    z = wp.sqrt(wp.max(0.0, 1.0 - u1))
    return wp.normalize(t * x + b * y + n * z)


@wp.func
def _sample_glossy_reflection(
    in_dir: wp.vec3,
    n: wp.vec3,
    roughness: float,
    u1: float,
    u2: float,
) -> wp.vec3:
    # Ideal reflection vector
    refl = wp.normalize(in_dir - 2.0 * wp.dot(in_dir, n) * n)
    rt, rb = _orthonormal_basis_from_normal(refl)
    # Cosine-power lobe around reflection vector controlled by roughness^2
    alpha = wp.clamp(roughness * roughness, 0.01, 1.0)
    exp_p = wp.max(1.0, (2.0 / (alpha * alpha)) - 2.0)
    cos_theta = wp.pow(wp.clamp(u1, 1e-6, 1.0), 1.0 / (exp_p + 1.0))
    sin_theta = wp.sqrt(wp.max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * PI * u2
    cand = wp.normalize(
        rt * (sin_theta * wp.cos(phi))
        + rb * (sin_theta * wp.sin(phi))
        + refl * cos_theta
    )
    # If perturbed glossy ray dips below surface hemisphere, reflect back into upper hemisphere
    if wp.dot(cand, n) <= 1e-4:
        cand = wp.normalize(cand - 2.0 * wp.dot(cand, n) * n)
    return cand


@wp.func
def _binary_search_cdf(
    cdf: wp.array(dtype=float),
    start_idx: int,
    count: int,
    u: float,
) -> int:
    lo = int(0)
    hi = count - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[start_idx + mid] < u:
            lo = mid + 1
        else:
            hi = mid
    return start_idx + lo


@wp.kernel
def _trace_patches_3bounce_kernel(
    mesh_id: wp.uint64,
    # Mesh geometry for sampling ray origins on each patch's triangles
    verts: wp.array(dtype=wp.vec3),
    faces: wp.array(dtype=int),
    face_normals: wp.array(dtype=wp.vec3),
    # Per-face material attributes
    face_albedo: wp.array(dtype=wp.vec3),
    face_roughness: wp.array(dtype=float),
    face_metallic: wp.array(dtype=float),
    face_emission: wp.array(dtype=wp.vec3),
    # Per-patch sampling tables (CSR over patch triangles)
    patch_tri_offsets: wp.array(dtype=int),
    patch_tri_counts: wp.array(dtype=int),
    patch_tri_indices: wp.array(dtype=int),
    patch_tri_cdf: wp.array(dtype=float),
    # Per-patch fallback center & local frame (T, B, N)
    patch_pos: wp.array(dtype=wp.vec3),
    patch_T: wp.array(dtype=wp.vec3),
    patch_B: wp.array(dtype=wp.vec3),
    patch_N: wp.array(dtype=wp.vec3),
    # Tracing params
    n_patches: int,
    rays_per_patch: int,
    max_bounces: int,
    max_t: float,
    seed: int,
    include_direct_bounce0: int,
    # Output: [n_patches, 12] flattened float32 (accumulated via atomic_add per ray)
    out_sh: wp.array(dtype=float),
):
    tid = wp.tid()
    p_idx = tid // rays_per_patch
    r_idx = tid - p_idx * rays_per_patch
    if p_idx >= n_patches:
        return

    rng = wp.rand_init(seed, tid)

    pT = patch_T[p_idx]
    pB = patch_B[p_idx]
    pN = patch_N[p_idx]

    # Sample ray origin uniformly across the patch's actual triangles
    tri_count = patch_tri_counts[p_idx]
    ro = patch_pos[p_idx]
    surf_n = pN
    if tri_count > 0:
        off = patch_tri_offsets[p_idx]
        u_tri = wp.randf(rng)
        slot = _binary_search_cdf(patch_tri_cdf, off, tri_count, u_tri)
        f_idx = patch_tri_indices[slot]
        v0 = verts[faces[f_idx * 3 + 0]]
        v1 = verts[faces[f_idx * 3 + 1]]
        v2 = verts[faces[f_idx * 3 + 2]]
        # Uniform barycentric sample
        r1 = wp.sqrt(wp.randf(rng))
        r2 = wp.randf(rng)
        ro = v0 * (1.0 - r1) + v1 * (r1 * (1.0 - r2)) + v2 * (r1 * r2)
        fn = face_normals[f_idx]
        if wp.dot(fn, pN) > 0.1:
            surf_n = fn
        else:
            surf_n = pN

    # Stratified cosine-weighted hemisphere direction on patch local frame (pT, pB, pN)
    # Using jittered stratification on r_idx for dramatically lower Monte Carlo variance!
    # Approximate square grid strata
    grid = int(wp.sqrt(float(rays_per_patch)))
    if grid * grid == rays_per_patch and grid > 1:
        gx = r_idx % grid
        gy = r_idx // grid
        u1 = (float(gx) + wp.randf(rng)) / float(grid)
        u2 = (float(gy) + wp.randf(rng)) / float(grid)
    else:
        u1 = wp.randf(rng)
        u2 = wp.randf(rng)

    # Sample direction in patch normal hemisphere
    rd0 = _sample_cosine_hemisphere(u1, u2, pT, pB, pN)
    # If surface normal slightly tilts from patch normal and ray goes behind triangle, push slightly
    if wp.dot(rd0, surf_n) <= 0.02:
        st, sb = _orthonormal_basis_from_normal(surf_n)
        rd0 = _sample_cosine_hemisphere(u1, u2, st, sb, surf_n)

    # Local frame projections of incoming direction rd0 (for Order-1 SH)
    w_T = wp.dot(rd0, pT)
    w_B = wp.dot(rd0, pB)
    w_N = wp.dot(rd0, pN)

    # Offset ray origin along surface normal to prevent self-intersection
    curr_ro = ro + surf_n * 2.0e-3
    curr_rd = rd0
    throughput = wp.vec3(1.0, 1.0, 1.0)
    rad1 = wp.vec3(0.0, 0.0, 0.0)
    rad2 = wp.vec3(0.0, 0.0, 0.0)
    rad3 = wp.vec3(0.0, 0.0, 0.0)

    # Trace up to (max_bounces + 1) segments:
    # b = 0: first hit from patch i. Its emission is DIRECT light (excluded unless include_direct_bounce0 == 1).
    # b = 1: 1-bounce indirect hit emission * albedo(hit0)
    # b = 2: 2-bounce indirect hit emission * albedo(hit0) * albedo(hit1)
    # b = 3: 3-bounce indirect hit emission * albedo(hit0) * albedo(hit1) * albedo(hit2)
    for b in range(4):
        if b > max_bounces:
            break
        q = wp.mesh_query_ray(mesh_id, curr_ro, curr_rd, max_t)
        if not q.result:
            break

        hit_f = q.face
        hit_p = curr_ro + curr_rd * q.t
        hit_n = face_normals[hit_f]
        # Handle potential back-face hit cleanly (treat closed solids: absorb back-face hits unless thin)
        cos_in = wp.dot(curr_rd, hit_n)
        if cos_in > 0.0:
            # Hit backside of a face; absorb to prevent light leaking through solid walls
            break

        em = face_emission[hit_f]
        contrib = wp.cw_mul(throughput, em)
        if b == 1:
            rad1 = rad1 + contrib
        elif b == 2:
            rad2 = rad2 + contrib
        elif b == 3:
            rad3 = rad3 + contrib
        elif b == 0 and include_direct_bounce0 != 0:
            rad1 = rad1 + contrib

        if b == max_bounces:
            break

        alb = face_albedo[hit_f]
        rough = face_roughness[hit_f]
        metal = face_metallic[hit_f]

        # Energy-conserving diffuse + metallic glossy BRDF:
        # Total hemispherical reflectance = (1 - metal)*alb + metal*alb = alb
        # (Exact analytical conservation for white-furnace and arbitrary PBR mixes!)
        u_brdf = wp.randf(rng)
        ht, hb = _orthonormal_basis_from_normal(hit_n)
        if u_brdf < metal:
            next_rd = _sample_glossy_reflection(curr_rd, hit_n, rough, wp.randf(rng), wp.randf(rng))
        else:
            next_rd = _sample_cosine_hemisphere(wp.randf(rng), wp.randf(rng), ht, hb, hit_n)

        throughput = wp.cw_mul(throughput, alb)
        # Early termination if throughput is negligible
        if wp.max(throughput[0], wp.max(throughput[1], throughput[2])) < 1.0e-4:
            break

        curr_ro = hit_p + hit_n * 2.0e-3
        curr_rd = next_rd

    inv_s = 1.0 / float(rays_per_patch)
    s1 = rad1 * inv_s
    s2 = rad2 * inv_s
    s3 = rad3 * inv_s
    base = p_idx * 36

    # Accumulate per-bounce Order-1 SH: [b1 (12), b2 (12), b3 (12)] = 36 floats/patch
    if s1[0] + s1[1] + s1[2] > 0.0:
        wp.atomic_add(out_sh, base + 0, s1[0])
        wp.atomic_add(out_sh, base + 1, s1[1])
        wp.atomic_add(out_sh, base + 2, s1[2])
        wp.atomic_add(out_sh, base + 3, s1[0] * w_T)
        wp.atomic_add(out_sh, base + 4, s1[1] * w_T)
        wp.atomic_add(out_sh, base + 5, s1[2] * w_T)
        wp.atomic_add(out_sh, base + 6, s1[0] * w_B)
        wp.atomic_add(out_sh, base + 7, s1[1] * w_B)
        wp.atomic_add(out_sh, base + 8, s1[2] * w_B)
        wp.atomic_add(out_sh, base + 9, s1[0] * w_N)
        wp.atomic_add(out_sh, base + 10, s1[1] * w_N)
        wp.atomic_add(out_sh, base + 11, s1[2] * w_N)

    if s2[0] + s2[1] + s2[2] > 0.0:
        b2 = base + 12
        wp.atomic_add(out_sh, b2 + 0, s2[0])
        wp.atomic_add(out_sh, b2 + 1, s2[1])
        wp.atomic_add(out_sh, b2 + 2, s2[2])
        wp.atomic_add(out_sh, b2 + 3, s2[0] * w_T)
        wp.atomic_add(out_sh, b2 + 4, s2[1] * w_T)
        wp.atomic_add(out_sh, b2 + 5, s2[2] * w_T)
        wp.atomic_add(out_sh, b2 + 6, s2[0] * w_B)
        wp.atomic_add(out_sh, b2 + 7, s2[1] * w_B)
        wp.atomic_add(out_sh, b2 + 8, s2[2] * w_B)
        wp.atomic_add(out_sh, b2 + 9, s2[0] * w_N)
        wp.atomic_add(out_sh, b2 + 10, s2[1] * w_N)
        wp.atomic_add(out_sh, b2 + 11, s2[2] * w_N)

    if s3[0] + s3[1] + s3[2] > 0.0:
        b3 = base + 24
        wp.atomic_add(out_sh, b3 + 0, s3[0])
        wp.atomic_add(out_sh, b3 + 1, s3[1])
        wp.atomic_add(out_sh, b3 + 2, s3[2])
        wp.atomic_add(out_sh, b3 + 3, s3[0] * w_T)
        wp.atomic_add(out_sh, b3 + 4, s3[1] * w_T)
        wp.atomic_add(out_sh, b3 + 5, s3[2] * w_T)
        wp.atomic_add(out_sh, b3 + 6, s3[0] * w_B)
        wp.atomic_add(out_sh, b3 + 7, s3[1] * w_B)
        wp.atomic_add(out_sh, b3 + 8, s3[2] * w_B)
        wp.atomic_add(out_sh, b3 + 9, s3[0] * w_N)
        wp.atomic_add(out_sh, b3 + 10, s3[1] * w_N)
        wp.atomic_add(out_sh, b3 + 11, s3[2] * w_N)


def trace_scene_patches_warp(
    *,
    verts: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_albedo: np.ndarray,
    face_roughness: np.ndarray,
    face_metallic: np.ndarray,
    face_emission: np.ndarray,
    patch_tri_offsets: np.ndarray,
    patch_tri_counts: np.ndarray,
    patch_tri_indices: np.ndarray,
    patch_tri_cdf: np.ndarray,
    patch_pos: np.ndarray,
    patch_T: np.ndarray,
    patch_B: np.ndarray,
    patch_N: np.ndarray,
    rays_per_patch: int = 256,
    max_bounces: int = 3,
    max_t: float = 1.0e4,
    seed: int = 42,
    include_direct_bounce0: bool = False,
    device: str = "cuda:0",
    per_bounce: bool = False,
) -> np.ndarray:
    """Trace exact 1-, 2-, and 3-bounce indirect Order-1 SH irradiance for all patches in a scene.

    Returns:
        np.ndarray of shape [n_patches, 36] if per_bounce=True ([b1(12), b2(12), b3(12)]),
        or [n_patches, 12] (b1 + b2 + b3 total) if per_bounce=False, dtype float32.
    """
    n_patches = int(patch_pos.shape[0])
    verts_wp = wp.array(np.ascontiguousarray(verts, dtype=np.float32), dtype=wp.vec3, device=device)
    faces_flat = np.ascontiguousarray(faces.reshape(-1), dtype=np.int32)
    faces_wp = wp.array(faces_flat, dtype=int, device=device)
    mesh = wp.Mesh(points=verts_wp, indices=faces_wp)

    fn_wp = wp.array(np.ascontiguousarray(face_normals, dtype=np.float32), dtype=wp.vec3, device=device)
    fa_wp = wp.array(np.ascontiguousarray(face_albedo, dtype=np.float32), dtype=wp.vec3, device=device)
    fr_wp = wp.array(np.ascontiguousarray(face_roughness, dtype=np.float32), dtype=float, device=device)
    fm_wp = wp.array(np.ascontiguousarray(face_metallic, dtype=np.float32), dtype=float, device=device)
    fe_wp = wp.array(np.ascontiguousarray(face_emission, dtype=np.float32), dtype=wp.vec3, device=device)

    off_wp = wp.array(np.ascontiguousarray(patch_tri_offsets, dtype=np.int32), dtype=int, device=device)
    cnt_wp = wp.array(np.ascontiguousarray(patch_tri_counts, dtype=np.int32), dtype=int, device=device)
    idx_wp = wp.array(np.ascontiguousarray(patch_tri_indices, dtype=np.int32), dtype=int, device=device)
    cdf_wp = wp.array(np.ascontiguousarray(patch_tri_cdf, dtype=np.float32), dtype=float, device=device)

    ppos_wp = wp.array(np.ascontiguousarray(patch_pos, dtype=np.float32), dtype=wp.vec3, device=device)
    pT_wp = wp.array(np.ascontiguousarray(patch_T, dtype=np.float32), dtype=wp.vec3, device=device)
    pB_wp = wp.array(np.ascontiguousarray(patch_B, dtype=np.float32), dtype=wp.vec3, device=device)
    pN_wp = wp.array(np.ascontiguousarray(patch_N, dtype=np.float32), dtype=wp.vec3, device=device)

    out_sh = wp.zeros(n_patches * 36, dtype=float, device=device)
    total_rays = n_patches * rays_per_patch

    wp.launch(
        _trace_patches_3bounce_kernel,
        dim=total_rays,
        inputs=[
            mesh.id,
            verts_wp,
            faces_wp,
            fn_wp,
            fa_wp,
            fr_wp,
            fm_wp,
            fe_wp,
            off_wp,
            cnt_wp,
            idx_wp,
            cdf_wp,
            ppos_wp,
            pT_wp,
            pB_wp,
            pN_wp,
            n_patches,
            rays_per_patch,
            max_bounces,
            max_t,
            seed,
            1 if include_direct_bounce0 else 0,
            out_sh,
        ],
        device=device,
    )
    wp.synchronize()
    sh36 = out_sh.numpy().reshape(n_patches, 36).astype(np.float32)
    if per_bounce:
        return sh36
    return (sh36[:, 0:12] + sh36[:, 12:24] + sh36[:, 24:36]).astype(np.float32)
