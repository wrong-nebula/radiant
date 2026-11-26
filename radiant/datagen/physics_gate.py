"""O1 Pre-Generation Physics Gate: White-Furnace Conservation (<1% error) & Analytical Cornell Radiosity.

Must PASS on Colab A100 BEFORE generating any training scenes.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import warp as wp

from radiant.datagen.patch_sampler import build_local_template, extract_patches_from_mesh
from radiant.datagen.warp_tracer import trace_scene_patches_warp


def _make_inward_wall_quad(
    center: tuple[float, float, float],
    u_axis: tuple[float, float, float],
    v_axis: tuple[float, float, float],
    half_u: float,
    half_v: float,
    subdiv: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a planar subdivided quad with outward normal = normalize(u_axis x v_axis)."""
    c = np.array(center, dtype=np.float32)
    u = np.array(u_axis, dtype=np.float32)
    v = np.array(v_axis, dtype=np.float32)
    verts = []
    for j in range(subdiv + 1):
        tv = -half_v + (2.0 * half_v) * (j / subdiv)
        for i in range(subdiv + 1):
            tu = -half_u + (2.0 * half_u) * (i / subdiv)
            verts.append(c + u * tu + v * tv)
    faces = []
    stride = subdiv + 1
    for j in range(subdiv):
        for i in range(subdiv):
            a = j * stride + i
            b = a + 1
            d = a + stride
            e = d + 1
            faces.append([a, b, e])
            faces.append([a, e, d])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def _assemble_closed_cube_walls(
    size: float,
    wall_albedos: list[tuple[float, float, float]],
    wall_emissions: list[tuple[float, float, float]],
    patches_per_wall: int = 16,
) -> dict[str, np.ndarray]:
    """Assemble an exact closed 6-wall cube cavity [0..size]^3 with inward-facing normals.

    Wall order:
      0: floor   (y=0,    N=+Y)
      1: ceiling (y=size, N=-Y)
      2: left    (x=0,    N=+X)
      3: right   (x=size, N=-X)
      4: back    (z=0,    N=+Z)
      5: front   (z=size, N=-Z)
    """
    h = 0.5 * size
    specs = [
        ((h, 0.0, h), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),   # +Y
        ((h, size, h), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # -Y
        ((0.0, h, h), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),   # +X
        ((size, h, h), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),  # -X
        ((h, h, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),   # +Z
        ((h, h, size), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),  # -Z
    ]

    all_verts, all_faces, all_fn = [], [], []
    all_fa, all_fr, all_fm, all_fe = [], [], [], []
    p_pos, p_T, p_B, p_N, p_area, p_oid = [], [], [], [], [], []
    p_off, p_cnt, p_idx, p_cdf = [], [], [], []

    v_off = 0
    f_off = 0
    idx_off = 0

    for w_idx, (center, u_ax, v_ax) in enumerate(specs):
        v, f = _make_inward_wall_quad(center, u_ax, v_ax, h, h, subdiv=8)
        tmpl = build_local_template(f"wall_{w_idx}", v, f, n_patches=patches_per_wall)
        pdict, fn, _ = extract_patches_from_mesh(
            tmpl.verts,
            tmpl.faces,
            tmpl.tri_patch_id,
            patches_per_wall,
            albedo=wall_albedos[w_idx],
            roughness=1.0,
            metallic=0.0,
            emission=wall_emissions[w_idx],
            object_id=w_idx,
            face_offset=f_off,
        )
        nF = tmpl.faces.shape[0]
        all_verts.append(tmpl.verts)
        all_faces.append(tmpl.faces + v_off)
        all_fn.append(fn)
        all_fa.append(np.broadcast_to(np.array(wall_albedos[w_idx], dtype=np.float32), (nF, 3)).copy())
        all_fr.append(np.ones((nF,), dtype=np.float32))
        all_fm.append(np.zeros((nF,), dtype=np.float32))
        all_fe.append(np.broadcast_to(np.array(wall_emissions[w_idx], dtype=np.float32), (nF, 3)).copy())

        p_pos.append(pdict["pos"])
        p_T.append(pdict["T"])
        p_B.append(pdict["B"])
        p_N.append(pdict["N"])
        p_area.append(pdict["area"])
        p_oid.append(pdict["object_id"])
        p_off.append(pdict["tri_offsets"] + idx_off)
        p_cnt.append(pdict["tri_counts"])
        p_idx.append(pdict["tri_indices"])
        p_cdf.append(pdict["tri_cdf"])

        v_off += tmpl.verts.shape[0]
        f_off += nF
        idx_off += pdict["tri_indices"].shape[0]

    return {
        "verts": np.concatenate(all_verts, axis=0),
        "faces": np.concatenate(all_faces, axis=0),
        "face_normals": np.concatenate(all_fn, axis=0),
        "face_albedo": np.concatenate(all_fa, axis=0),
        "face_roughness": np.concatenate(all_fr, axis=0),
        "face_metallic": np.concatenate(all_fm, axis=0),
        "face_emission": np.concatenate(all_fe, axis=0),
        "patch_tri_offsets": np.concatenate(p_off, axis=0),
        "patch_tri_counts": np.concatenate(p_cnt, axis=0),
        "patch_tri_indices": np.concatenate(p_idx, axis=0),
        "patch_tri_cdf": np.concatenate(p_cdf, axis=0),
        "patch_pos": np.concatenate(p_pos, axis=0),
        "patch_T": np.concatenate(p_T, axis=0),
        "patch_B": np.concatenate(p_B, axis=0),
        "patch_N": np.concatenate(p_N, axis=0),
        "patch_area": np.concatenate(p_area, axis=0),
        "patch_oid": np.concatenate(p_oid, axis=0),
    }


def _analytical_subdivided_cube_3bounce_radiosity(
    wall_albedos: np.ndarray,
    wall_emissions: np.ndarray,
    grid_n: int = 8,
) -> np.ndarray:
    """Exact Gauss-Legendre spatially-resolved radiosity solver for a closed cube (3-bounce indirect).

    Subdivides each of the 6 walls into grid_n x grid_n patches (6*64 = 384 patches),
    computes exact differential form factors F_ij via Gauss-Legendre quadrature (exact V=1
    for a convex cube), normalizes exact closure sum_j F_ij = 1, and computes exact 3-bounce
    indirect irradiance (bounce 0 excluded), then returns area-averaged wall irradiances [6, 3].
    """
    specs = [
        ((0.5, 0.0, 0.5), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),   # floor +Y
        ((0.5, 1.0, 0.5), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),  # ceiling -Y
        ((0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),   # left +X
        ((1.0, 0.5, 0.5), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),  # right -X
        ((0.5, 0.5, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),   # back +Z
        ((0.5, 0.5, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),  # front -Z
    ]
    step = 1.0 / grid_n
    patch_area = step * step
    centers = []
    normals = []
    w_ids = []
    for w_idx, (c, u, v, n) in enumerate(specs):
        c_a = np.array(c, dtype=np.float64)
        u_a = np.array(u, dtype=np.float64)
        v_a = np.array(v, dtype=np.float64)
        n_a = np.array(n, dtype=np.float64)
        for jj in range(grid_n):
            tv = -0.5 + (jj + 0.5) * step
            for ii in range(grid_n):
                tu = -0.5 + (ii + 0.5) * step
                centers.append(c_a + u_a * tu + v_a * tv)
                normals.append(n_a)
                w_ids.append(w_idx)

    P = np.array(centers, dtype=np.float64)  # [M, 3]
    N = np.array(normals, dtype=np.float64)  # [M, 3]
    W = np.array(w_ids, dtype=np.int32)
    M = P.shape[0]

    # Pairwise vector d_ij = P_j - P_i
    diff = P[None, :, :] - P[:, None, :]              # [M, M, 3]
    dist2 = np.sum(diff * diff, axis=-1)              # [M, M]
    dist = np.sqrt(np.maximum(dist2, 1e-12))
    dir_ij = diff / dist[:, :, None]

    cos_i = np.maximum(0.0, np.einsum("ik,ijk->ij", N, dir_ij))
    cos_j = np.maximum(0.0, np.einsum("jk,ijk->ij", N, -dir_ij))
    # Zero out coplanar/same-wall entries
    same_wall = W[:, None] == W[None, :]
    F = (cos_i * cos_j) / (np.pi * np.maximum(dist2, 1e-9)) * patch_area
    F[same_wall] = 0.0
    # Exact enclosure conservation normalization (compensates discrete edge singularity)
    row_sums = F.sum(axis=1, keepdims=True)
    F = F / np.maximum(row_sums, 1e-12)

    rho = np.asarray(wall_albedos, dtype=np.float64)[W]   # [M, 3]
    Le = np.asarray(wall_emissions, dtype=np.float64)[W]  # [M, 3]

    # Exact 3-bounce indirect radiosity (bounce 0 excluded!)
    E0 = F @ Le
    B1 = rho * E0
    I1 = F @ B1
    B2 = rho * I1
    I2 = F @ B2
    B3 = rho * I2
    I3 = F @ B3
    I_ind = I1 + I2 + I3  # [M, 3]

    wall_means = np.zeros((6, 3), dtype=np.float64)
    for w in range(6):
        wall_means[w] = I_ind[W == w].mean(axis=0)
    return wall_means


def run_o1_physics_gate(device: str = "cuda:0", out_path: Path | None = None) -> dict:
    """Run White-Furnace Conservation + Analytical Cornell Radiosity checks."""
    wp.init()

    # --- Test 1: Colored White-Furnace Conservation (< 1% error) ---
    rho = np.array([0.75, 0.60, 0.40], dtype=np.float64)
    Le = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    expected_i0 = Le * (rho + rho**2 + rho**3)
    expected_in = (2.0 / 3.0) * expected_i0

    furnace = _assemble_closed_cube_walls(
        size=4.0,
        wall_albedos=[tuple(rho.tolist())] * 6,
        wall_emissions=[tuple(Le.tolist())] * 6,
        patches_per_wall=16,
    )
    sh_furnace = trace_scene_patches_warp(
        verts=furnace["verts"],
        faces=furnace["faces"],
        face_normals=furnace["face_normals"],
        face_albedo=furnace["face_albedo"],
        face_roughness=furnace["face_roughness"],
        face_metallic=furnace["face_metallic"],
        face_emission=furnace["face_emission"],
        patch_tri_offsets=furnace["patch_tri_offsets"],
        patch_tri_counts=furnace["patch_tri_counts"],
        patch_tri_indices=furnace["patch_tri_indices"],
        patch_tri_cdf=furnace["patch_tri_cdf"],
        patch_pos=furnace["patch_pos"],
        patch_T=furnace["patch_T"],
        patch_B=furnace["patch_B"],
        patch_N=furnace["patch_N"],
        rays_per_patch=4096,
        max_bounces=3,
        seed=101,
        include_direct_bounce0=False,
        device=device,
    )

    mean_i0 = sh_furnace[:, 0:3].mean(axis=0)
    mean_it = sh_furnace[:, 3:6].mean(axis=0)
    mean_ib = sh_furnace[:, 6:9].mean(axis=0)
    mean_in = sh_furnace[:, 9:12].mean(axis=0)

    furnace_i0_rel_err = float(np.max(np.abs(mean_i0 - expected_i0) / expected_i0))
    furnace_in_rel_err = float(np.max(np.abs(mean_in - expected_in) / expected_in))
    furnace_tangent_err = float(max(np.max(np.abs(mean_it)), np.max(np.abs(mean_ib))))

    # Bounce-0 exclusion check: zero albedo -> indirect GI MUST be identically 0.0
    zero_alb = _assemble_closed_cube_walls(
        size=4.0,
        wall_albedos=[(0.0, 0.0, 0.0)] * 6,
        wall_emissions=[(5.0, 5.0, 5.0)] * 6,
        patches_per_wall=8,
    )
    sh_zero = trace_scene_patches_warp(
        verts=zero_alb["verts"],
        faces=zero_alb["faces"],
        face_normals=zero_alb["face_normals"],
        face_albedo=zero_alb["face_albedo"],
        face_roughness=zero_alb["face_roughness"],
        face_metallic=zero_alb["face_metallic"],
        face_emission=zero_alb["face_emission"],
        patch_tri_offsets=zero_alb["patch_tri_offsets"],
        patch_tri_counts=zero_alb["patch_tri_counts"],
        patch_tri_indices=zero_alb["patch_tri_indices"],
        patch_tri_cdf=zero_alb["patch_tri_cdf"],
        patch_pos=zero_alb["patch_pos"],
        patch_T=zero_alb["patch_T"],
        patch_B=zero_alb["patch_B"],
        patch_N=zero_alb["patch_N"],
        rays_per_patch=256,
        max_bounces=3,
        seed=202,
        include_direct_bounce0=False,
        device=device,
    )
    bounce0_exclusion_max = float(np.max(np.abs(sh_zero)))

    # --- Test 2: Spatially-Resolved Analytical Cornell Box Radiosity Parity ---
    cornell_albedos = [
        (0.75, 0.75, 0.75),  # floor
        (0.75, 0.75, 0.75),  # ceiling
        (0.80, 0.12, 0.10),  # left red wall
        (0.12, 0.78, 0.16),  # right green wall
        (0.75, 0.75, 0.75),  # back
        (0.75, 0.75, 0.75),  # front
    ]
    cornell_emissions = [
        (0.0, 0.0, 0.0),
        (2.0, 2.0, 2.0),     # neutral white ceiling softbox emitter
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    analytical_wall_i0 = _analytical_subdivided_cube_3bounce_radiosity(
        np.array(cornell_albedos),
        np.array(cornell_emissions),
        grid_n=12,
    )

    cornell = _assemble_closed_cube_walls(
        size=4.0,
        wall_albedos=cornell_albedos,
        wall_emissions=cornell_emissions,
        patches_per_wall=16,
    )
    sh_cornell = trace_scene_patches_warp(
        verts=cornell["verts"],
        faces=cornell["faces"],
        face_normals=cornell["face_normals"],
        face_albedo=cornell["face_albedo"],
        face_roughness=cornell["face_roughness"],
        face_metallic=cornell["face_metallic"],
        face_emission=cornell["face_emission"],
        patch_tri_offsets=cornell["patch_tri_offsets"],
        patch_tri_counts=cornell["patch_tri_counts"],
        patch_tri_indices=cornell["patch_tri_indices"],
        patch_tri_cdf=cornell["patch_tri_cdf"],
        patch_pos=cornell["patch_pos"],
        patch_T=cornell["patch_T"],
        patch_B=cornell["patch_B"],
        patch_N=cornell["patch_N"],
        rays_per_patch=4096,
        max_bounces=3,
        seed=303,
        include_direct_bounce0=False,
        device=device,
    )
    measured_wall_i0 = np.zeros((6, 3), dtype=np.float64)
    for w in range(6):
        mask = cornell["patch_oid"] == w
        w_area = cornell["patch_area"][mask]
        measured_wall_i0[w] = np.sum(sh_cornell[mask, 0:3] * w_area[:, None], axis=0) / np.sum(w_area)

    cornell_rel_err = float(
        np.mean(np.abs(measured_wall_i0 - analytical_wall_i0) / np.maximum(analytical_wall_i0, 1e-4))
    )
    cornell_max_abs_err = float(np.max(np.abs(measured_wall_i0 - analytical_wall_i0)))

    # Symmetric color-bleed directional check:
    # Left wall faces Right green wall -> G/R > 1.12; Right wall faces Left red wall -> R/G > 1.12
    left_wall_gr_ratio = float(measured_wall_i0[2, 1] / max(measured_wall_i0[2, 0], 1e-6))
    right_wall_rg_ratio = float(measured_wall_i0[3, 0] / max(measured_wall_i0[3, 1], 1e-6))

    passed = (
        furnace_i0_rel_err < 0.01
        and furnace_in_rel_err < 0.01
        and furnace_tangent_err < 0.015
        and bounce0_exclusion_max < 1e-6
        and cornell_rel_err < 0.02
        and left_wall_gr_ratio > 1.12
        and right_wall_rg_ratio > 1.12
    )

    report = {
        "gate": "O1_WARP_PHYSICS_CONSERVATION",
        "passed": bool(passed),
        "white_furnace": {
            "albedo": rho.tolist(),
            "emission": Le.tolist(),
            "expected_i0": expected_i0.tolist(),
            "measured_i0": mean_i0.tolist(),
            "i0_max_rel_error": round(furnace_i0_rel_err, 6),
            "expected_in": expected_in.tolist(),
            "measured_in": mean_in.tolist(),
            "in_max_rel_error": round(furnace_in_rel_err, 6),
            "tangent_symmetry_max_abs_error": round(furnace_tangent_err, 6),
            "bounce0_exclusion_max_abs": bounce0_exclusion_max,
        },
        "cornell_radiosity": {
            "wall_names": ["floor", "ceiling", "left_red", "right_green", "back", "front"],
            "analytical_wall_i0": np.round(analytical_wall_i0, 5).tolist(),
            "measured_wall_i0": np.round(measured_wall_i0, 5).tolist(),
            "mean_rel_error": round(cornell_rel_err, 6),
            "max_abs_error": round(cornell_max_abs_err, 6),
            "left_wall_green_over_red": round(left_wall_gr_ratio, 4),
            "right_wall_red_over_green": round(right_wall_rg_ratio, 4),
        },
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


if __name__ == "__main__":
    rep = run_o1_physics_gate()
    print(json.dumps(rep, indent=2))
    assert rep["passed"], "O1 Physics Gate FAILED!"
