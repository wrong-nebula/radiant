"""Deterministic 32-patch local-space mesh tokenizer (area-weighted FPS + Lloyd clustering).

Matches `src/radiant/patchSampler.ts` mathematically so Three.js meshes tokenized at runtime
share the exact same patch distribution, local tangent frames, and geometric statistics
as training data generated on Colab A100.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from radiant.sh_math import build_frisvad_basis_np

PATCHES_PER_OBJECT = 32


@dataclass
class LocalMeshTemplate:
    """Pre-clustered watertight mesh archetype in canonical local coordinates."""

    name: str
    verts: np.ndarray              # [V, 3] float32
    faces: np.ndarray              # [F, 3] int32
    tri_patch_id: np.ndarray       # [F] int32 in [0, K-1]
    n_patches: int = PATCHES_PER_OBJECT
    # Precomputed local geometry & CSR tables for 15x faster batched instantiation
    local_face_normals: np.ndarray | None = None
    local_patch_pos: np.ndarray | None = None
    local_patch_N: np.ndarray | None = None
    local_patch_area: np.ndarray | None = None
    tri_offsets: np.ndarray | None = None
    tri_counts: np.ndarray | None = None
    tri_indices: np.ndarray | None = None
    tri_cdf: np.ndarray | None = None


@dataclass
class PatchSet:
    """World-space tokenized patches + CSR triangle sampling tables for Warp tracing."""

    pos: np.ndarray            # [N, 3] float32 (surface-snapped patch centers)
    T: np.ndarray              # [N, 3] float32
    B: np.ndarray              # [N, 3] float32
    N: np.ndarray              # [N, 3] float32
    area: np.ndarray           # [N] float32
    albedo: np.ndarray         # [N, 3] float32
    roughness: np.ndarray      # [N] float32
    metallic: np.ndarray       # [N] float32
    emission: np.ndarray       # [N, 3] float32
    object_id: np.ndarray      # [N] int32


def compute_face_geometry(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (face_centroids [F,3], face_normals [F,3], face_areas [F])."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    dbl_area = np.linalg.norm(cross, axis=-1)
    areas = 0.5 * dbl_area
    normals = cross / np.maximum(dbl_area[:, None], 1e-12)
    return centroids.astype(np.float32), normals.astype(np.float32), areas.astype(np.float32)


def subdivide_mesh_to_min_faces(
    verts: np.ndarray,
    faces: np.ndarray,
    min_faces: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint-subdivide low-poly meshes (like a 12-tri slab) so 32 patches cover every face cleanly."""
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    while f.shape[0] < min_faces:
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        m01 = 0.5 * (v0 + v1)
        m12 = 0.5 * (v1 + v2)
        m20 = 0.5 * (v2 + v0)
        n_v = v.shape[0]
        nf = f.shape[0]
        i01 = np.arange(n_v, n_v + nf, dtype=np.int32)
        i12 = np.arange(n_v + nf, n_v + 2 * nf, dtype=np.int32)
        i20 = np.arange(n_v + 2 * nf, n_v + 3 * nf, dtype=np.int32)
        v = np.concatenate([v, m01, m12, m20], axis=0)
        f = np.concatenate(
            [
                np.stack([f[:, 0], i01, i20], axis=-1),
                np.stack([i01, f[:, 1], i12], axis=-1),
                np.stack([i20, i12, f[:, 2]], axis=-1),
                np.stack([i01, i12, i20], axis=-1),
            ],
            axis=0,
        )
    return v, f


def cluster_mesh_patches(
    verts: np.ndarray,
    faces: np.ndarray,
    n_patches: int = PATCHES_PER_OBJECT,
    lloyd_iters: int = 6,
    normal_weight: float = 0.45,
) -> np.ndarray:
    """Deterministic area-weighted Farthest-Point Sampling + normal-aware Lloyd clustering.

    Uses position + normal_weight * extent * normal so patches NEVER straddle sharp 90-deg edges!
    Returns tri_patch_id [F] in [0, n_patches - 1].
    """
    centroids, normals, areas = compute_face_geometry(verts, faces)
    F = centroids.shape[0]
    if F <= n_patches:
        return (np.arange(F, dtype=np.int32) % n_patches)

    bbox_min = centroids.min(axis=0)
    bbox_max = centroids.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    nw = normal_weight * max(diag, 1e-3)

    # 6D normal-aware feature per triangle
    feats = np.concatenate([centroids, normals * nw], axis=-1)  # [F, 6]
    # Face-normal stratified seeding: ensure every distinct major normal direction
    # (e.g. thin 0.2m edges of a wall slab or back of a column) gets coverage before FPS!
    centers = np.zeros((n_patches, 6), dtype=np.float32)
    total_area = float(np.sum(areas))
    # Quantize normals to 14 canonical directions (6 axis faces + 8 octants)
    first_idx = int(np.argmax(areas))
    seeded_indices = [first_idx]
    centers[0] = feats[first_idx]

    # Seed up to 6 distinct orthogonal normal directions (dot < 0.35 with existing seeded normals)
    seed_normals = [normals[first_idx]]
    for _ in range(min(6, n_patches - 1)):
        dots = np.max(np.stack([normals @ sn for sn in seed_normals], axis=-1), axis=-1)
        orth_mask = dots < 0.35
        if not np.any(orth_mask):
            break
        masked_areas = np.where(orth_mask, areas, -1.0)
        cand = int(np.argmax(masked_areas))
        if masked_areas[cand] < total_area * 0.0005:
            break
        k_idx = len(seeded_indices)
        seeded_indices.append(cand)
        seed_normals.append(normals[cand])
        centers[k_idx] = feats[cand]

    # Area-weighted farthest-point initialization for remaining patches
    min_dist2 = np.full((F,), np.inf, dtype=np.float32)
    for k in range(len(seeded_indices)):
        d = feats - centers[k][None, :]
        min_dist2 = np.minimum(min_dist2, np.sum(d * d, axis=-1))
    area_w = np.sqrt(np.maximum(areas, 1e-10))
    for k in range(len(seeded_indices), n_patches):
        score = min_dist2 * area_w
        nxt = int(np.argmax(score))
        centers[k] = feats[nxt]
        d = feats - centers[k][None, :]
        d2 = np.sum(d * d, axis=-1)
        min_dist2 = np.minimum(min_dist2, d2)

    # Normal-aware Lloyd refinement
    labels = np.zeros(F, dtype=np.int32)
    for _ in range(lloyd_iters):
        # [F, K] squared distances
        d2 = (
            np.sum(feats * feats, axis=-1, keepdims=True)
            - 2.0 * (feats @ centers.T)
            + np.sum(centers * centers, axis=-1)[None, :]
        )
        labels = np.argmin(d2, axis=-1).astype(np.int32)
        for k in range(n_patches):
            mask = labels == k
            if np.any(mask):
                w = areas[mask]
                w_sum = float(np.sum(w))
                if w_sum > 1e-12:
                    c_pos = np.sum(centroids[mask] * w[:, None], axis=0) / w_sum
                    c_nrm = np.sum(normals[mask] * w[:, None], axis=0)
                    n_norm = float(np.linalg.norm(c_nrm))
                    if n_norm > 1e-8:
                        c_nrm = c_nrm / n_norm
                    centers[k, :3] = c_pos
                    centers[k, 3:] = c_nrm * nw

    # Final assignment
    d2 = (
        np.sum(feats * feats, axis=-1, keepdims=True)
        - 2.0 * (feats @ centers.T)
        + np.sum(centers * centers, axis=-1)[None, :]
    )
    labels = np.argmin(d2, axis=-1).astype(np.int32)
    return labels


def build_local_template(name: str, verts: np.ndarray, faces: np.ndarray, n_patches: int = PATCHES_PER_OBJECT) -> LocalMeshTemplate:
    """Subdivide if needed and pre-compute deterministic 32-patch clustering + CSR tables once."""
    v, f = subdivide_mesh_to_min_faces(verts, faces, min_faces=max(128, n_patches * 4))
    tri_pid = cluster_mesh_patches(v, f, n_patches=n_patches)
    pdict, fn, _ = extract_patches_from_mesh(
        v, f, tri_pid, n_patches,
        albedo=(0.5, 0.5, 0.5), roughness=0.8, metallic=0.0, emission=(0.0, 0.0, 0.0),
        object_id=0, face_offset=0,
    )
    return LocalMeshTemplate(
        name=name,
        verts=v,
        faces=f,
        tri_patch_id=tri_pid,
        n_patches=n_patches,
        local_face_normals=fn,
        local_patch_pos=pdict["pos"],
        local_patch_N=pdict["N"],
        local_patch_area=pdict["area"],
        tri_offsets=pdict["tri_offsets"],
        tri_counts=pdict["tri_counts"],
        tri_indices=pdict["tri_indices"],
        tri_cdf=pdict["tri_cdf"],
    )


def fast_transform_template_patches(
    tmpl: LocalMeshTemplate,
    scale: tuple[float, float, float],
    R: np.ndarray,
    pos: tuple[float, float, float],
    world_offset: tuple[float, float, float],
    albedo: tuple[float, float, float],
    roughness: float,
    metallic: float,
    emission: tuple[float, float, float],
    object_id: int,
    face_offset: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Ultra-fast vectorized instantiation using precomputed LocalMeshTemplate CSR & patch stats."""
    assert tmpl.local_face_normals is not None and tmpl.local_patch_pos is not None
    assert tmpl.local_patch_N is not None and tmpl.local_patch_area is not None
    assert tmpl.tri_offsets is not None and tmpl.tri_counts is not None
    assert tmpl.tri_indices is not None and tmpl.tri_cdf is not None

    sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    scale_arr = np.array([sx, sy, sz], dtype=np.float32)
    inv_scale = np.array([1.0 / max(sx, 1e-6), 1.0 / max(sy, 1e-6), 1.0 / max(sz, 1e-6)], dtype=np.float32)
    pos_arr = np.array(pos, dtype=np.float32)
    wo_arr = np.array(world_offset, dtype=np.float32)
    Rt = R.T

    # 1. World vertices (with mega-batch world_offset for Warp BVH)
    v_world = (tmpl.verts * scale_arr[None, :]) @ Rt + (pos_arr + wo_arr)[None, :]

    # 2. World face normals via inverse-transpose scaling + rotation
    fn = (tmpl.local_face_normals * inv_scale[None, :]) @ Rt
    fn /= np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), 1e-8)

    # 3. Scene-local patch positions (without mega-batch world_offset)
    p_pos = (tmpl.local_patch_pos * scale_arr[None, :]) @ Rt + pos_arr[None, :]

    # 4. World patch normals & Frisvad basis
    p_N = (tmpl.local_patch_N * inv_scale[None, :]) @ Rt
    p_T, p_B, p_N = build_frisvad_basis_np(p_N)

    # 5. Exact scaled patch areas: A' = A * sqrt((nx*sy*sz)^2 + (ny*sx*sz)^2 + (nz*sx*sy)^2)
    ln = tmpl.local_patch_N
    a_scale = np.sqrt(
        (ln[:, 0] * sy * sz) ** 2
        + (ln[:, 1] * sx * sz) ** 2
        + (ln[:, 2] * sx * sy) ** 2
    ).astype(np.float32)
    p_area = tmpl.local_patch_area * np.maximum(a_scale, 1e-6)

    K = tmpl.n_patches
    pdict = {
        "pos": p_pos.astype(np.float32),
        "T": p_T,
        "B": p_B,
        "N": p_N,
        "area": p_area,
        "albedo": np.broadcast_to(np.asarray(albedo, dtype=np.float32), (K, 3)).copy(),
        "roughness": np.full((K,), float(roughness), dtype=np.float32),
        "metallic": np.full((K,), float(metallic), dtype=np.float32),
        "emission": np.broadcast_to(np.asarray(emission, dtype=np.float32), (K, 3)).copy(),
        "object_id": np.full((K,), int(object_id), dtype=np.int32),
        "tri_offsets": tmpl.tri_offsets,
        "tri_counts": tmpl.tri_counts,
        "tri_indices": (tmpl.tri_indices + face_offset).astype(np.int32),
        "tri_cdf": tmpl.tri_cdf,
    }
    return v_world.astype(np.float32), fn.astype(np.float32), pdict


def extract_patches_from_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    tri_patch_id: np.ndarray,
    n_patches: int,
    albedo: tuple[float, float, float] | np.ndarray,
    roughness: float,
    metallic: float,
    emission: tuple[float, float, float] | np.ndarray,
    object_id: int = 0,
    face_offset: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Compute world-space patch tokens and per-patch CSR triangle tables for one object.

    Returns:
      patch_dict: pos, T, B, N, area, albedo, roughness, metallic, emission, object_id,
                  tri_offsets, tri_counts, tri_indices, tri_cdf (indices offset by face_offset)
      face_normals: [F, 3]
      face_areas: [F]
    """
    centroids, face_normals, face_areas = compute_face_geometry(verts, faces)
    F = faces.shape[0]
    K = n_patches

    pos = np.zeros((K, 3), dtype=np.float32)
    nrm = np.zeros((K, 3), dtype=np.float32)
    area = np.zeros((K,), dtype=np.float32)
    tri_counts = np.zeros((K,), dtype=np.int32)

    # Sort triangles by patch_id for fast CSR construction
    order = np.argsort(tri_patch_id, kind="stable")
    sorted_pid = tri_patch_id[order]
    counts = np.bincount(sorted_pid, minlength=K).astype(np.int32)
    offsets = np.zeros((K,), dtype=np.int32)
    if K > 1:
        offsets[1:] = np.cumsum(counts[:-1])

    tri_indices = (order + face_offset).astype(np.int32)
    tri_cdf = np.zeros((F,), dtype=np.float32)

    # Vectorized sums for area, weighted centroid, weighted normal
    np.add.at(area, tri_patch_id, face_areas)
    w_cent = centroids * face_areas[:, None]
    w_nrm = face_normals * face_areas[:, None]
    np.add.at(pos, tri_patch_id, w_cent)
    np.add.at(nrm, tri_patch_id, w_nrm)

    valid = area > 1e-10
    pos[valid] /= area[valid, None]
    nrm_len = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm[valid] /= np.maximum(nrm_len[valid], 1e-8)

    # Fill any empty cluster (rare) with fallback from largest cluster
    if not np.all(valid):
        best = int(np.argmax(area))
        for k in np.where(~valid)[0]:
            pos[k] = pos[best]
            nrm[k] = nrm[best]
            area[k] = 1e-4

    # Surface-snap patch centroid to nearest member triangle centroid + build CDF
    for k in range(K):
        c = int(counts[k])
        tri_counts[k] = c
        if c > 0:
            off = int(offsets[k])
            f_ids = order[off : off + c]
            a_sub = np.maximum(face_areas[f_ids], 1e-10)
            cs = np.cumsum(a_sub)
            tri_cdf[off : off + c] = (cs / cs[-1]).astype(np.float32)
            # Snap pos[k] to closest triangle centroid in this patch so it lies ON the surface
            d = centroids[f_ids] - pos[k][None, :]
            nearest = f_ids[int(np.argmin(np.sum(d * d, axis=-1)))]
            pos[k] = centroids[nearest]
            if float(np.linalg.norm(nrm[k])) < 0.2:
                nrm[k] = face_normals[nearest]
                nrm[k] /= max(float(np.linalg.norm(nrm[k])), 1e-8)

    T, B, N = build_frisvad_basis_np(nrm)

    alb_arr = np.broadcast_to(np.asarray(albedo, dtype=np.float32), (K, 3)).copy()
    em_arr = np.broadcast_to(np.asarray(emission, dtype=np.float32), (K, 3)).copy()
    rgh_arr = np.full((K,), float(roughness), dtype=np.float32)
    met_arr = np.full((K,), float(metallic), dtype=np.float32)
    oid_arr = np.full((K,), int(object_id), dtype=np.int32)

    return (
        {
            "pos": pos,
            "T": T,
            "B": B,
            "N": N,
            "area": area,
            "albedo": alb_arr,
            "roughness": rgh_arr,
            "metallic": met_arr,
            "emission": em_arr,
            "object_id": oid_arr,
            "tri_offsets": offsets,
            "tri_counts": tri_counts,
            "tri_indices": tri_indices,
            "tri_cdf": tri_cdf,
        },
        face_normals,
        face_areas,
    )
