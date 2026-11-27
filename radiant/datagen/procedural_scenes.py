"""Combinatorial Procedural + Watertight Sculptural Scene Generator for THREE-RADIANT-GI.

Builds:
  - 150+ parametric architectural & interior generators (slabs, doorways, windows, arches,
    barrel vaults, columns, partitions, tables, chairs, shelves, curved reflectors, plinths,
    skylight wells, movable area lights, colored bounce walls).
  - 200 distinct watertight sculptural & prop meshes (~1.2k-2.2k tris each, pre-clustered
    into 32 local patches), strictly partitioned into 160 TRAIN archetypes and 40 strictly
    UNSEEN HELD-OUT TEST archetypes (zero overlap!).
  - Supports batched mega-scene assembly for ultra-fast NVIDIA Warp GPU BVH tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from radiant.datagen.patch_sampler import (
    LOD_PATCH_COUNTS,
    PATCHES_PER_OBJECT,
    LocalMeshTemplate,
    build_local_template,
    build_multi_lod_templates,
    fast_transform_template_patches,
)


# ---------------------------------------------------------------------------
# Low-level watertight mesh primitives (always closed, outward-facing normals)
# ---------------------------------------------------------------------------

def _box_mesh(sx: float = 1.0, sy: float = 1.0, sz: float = 1.0, subdiv: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Closed unit box [-sx/2..sx/2] x [-sy/2..sy/2] x [-sz/2..sz/2] with outward normals."""
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    face_specs = [
        ((0, 0, hz), (hx, 0, 0), (0, hy, 0)),     # +Z
        ((0, 0, -hz), (-hx, 0, 0), (0, hy, 0)),   # -Z
        ((hx, 0, 0), (0, 0, -hz), (0, hy, 0)),    # +X
        ((-hx, 0, 0), (0, 0, hz), (0, hy, 0)),    # -X
        ((0, hy, 0), (hx, 0, 0), (0, 0, -hz)),    # +Y
        ((0, -hy, 0), (hx, 0, 0), (0, 0, hz)),    # -Y
    ]
    verts_list, faces_list = [], []
    v_off = 0
    for c, u, v in face_specs:
        c_arr = np.array(c, dtype=np.float32)
        u_arr = np.array(u, dtype=np.float32)
        v_arr = np.array(v, dtype=np.float32)
        for j in range(subdiv + 1):
            tv = -1.0 + 2.0 * (j / subdiv)
            for i in range(subdiv + 1):
                tu = -1.0 + 2.0 * (i / subdiv)
                verts_list.append(c_arr + u_arr * tu + v_arr * tv)
        stride = subdiv + 1
        for j in range(subdiv):
            for i in range(subdiv):
                a = v_off + j * stride + i
                b = a + 1
                d = a + stride
                e = d + 1
                faces_list.append([a, b, e])
                faces_list.append([a, e, d])
        v_off += stride * stride
    return np.array(verts_list, dtype=np.float32), np.array(faces_list, dtype=np.int32)


def _merge_meshes(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    off = 0
    for v, f in parts:
        verts.append(v.astype(np.float32))
        faces.append((f + off).astype(np.int32))
        off += v.shape[0]
    return np.concatenate(verts, axis=0), np.concatenate(faces, axis=0)


def _translate_mesh(v: np.ndarray, f: np.ndarray, t: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    return (v + np.array(t, dtype=np.float32)[None, :]).astype(np.float32), f


def _cylinder_mesh(
    radius_bottom: float = 0.5,
    radius_top: float = 0.5,
    height: float = 1.0,
    radial_segs: int = 24,
    height_segs: int = 4,
    flute_amp: float = 0.0,
    flute_freq: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed watertight cylinder / fluted column centered at origin, Y in [-h/2, h/2]."""
    verts = []
    faces = []
    hy = 0.5 * height
    for j in range(height_segs + 1):
        ty = j / height_segs
        y = -hy + height * ty
        r_base = radius_bottom * (1.0 - ty) + radius_top * ty
        for i in range(radial_segs):
            ang = 2.0 * math.pi * i / radial_segs
            r = r_base * (1.0 + flute_amp * math.cos(flute_freq * ang))
            verts.append([r * math.cos(ang), y, r * math.sin(ang)])
    for j in range(height_segs):
        for i in range(radial_segs):
            ni = (i + 1) % radial_segs
            a = j * radial_segs + i
            b = j * radial_segs + ni
            c = (j + 1) * radial_segs + ni
            d = (j + 1) * radial_segs + i
            # Outward normal: (d - a) x (b - a) points radially outward
            faces.append([a, d, c])
            faces.append([a, c, b])
    # Bottom cap (-Y)
    bot_center = len(verts)
    verts.append([0.0, -hy, 0.0])
    for i in range(radial_segs):
        ni = (i + 1) % radial_segs
        faces.append([bot_center, i, ni])
    # Top cap (+Y)
    top_center = len(verts)
    verts.append([0.0, hy, 0.0])
    top_ring = height_segs * radial_segs
    for i in range(radial_segs):
        ni = (i + 1) % radial_segs
        faces.append([top_center, top_ring + ni, top_ring + i])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def _curved_shell_mesh(
    radius: float = 1.0,
    thickness: float = 0.12,
    arc_angle: float = math.pi,
    length: float = 2.0,
    arc_segs: int = 18,
    len_segs: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed watertight thick curved shell (used for barrel vaults, arches, curved reflectors)."""
    r_in = max(radius - 0.5 * thickness, 0.1)
    r_out = radius + 0.5 * thickness
    hl = 0.5 * length
    # Profile loop around cross section: outer arc forward, inner arc backward
    profile = []
    for k in range(arc_segs + 1):
        a = -0.5 * arc_angle + arc_angle * (k / arc_segs)
        profile.append((r_out * math.sin(a), r_out * math.cos(a)))
    for k in range(arc_segs, -1, -1):
        a = -0.5 * arc_angle + arc_angle * (k / arc_segs)
        profile.append((r_in * math.sin(a), r_in * math.cos(a)))
    P = len(profile)

    verts = []
    faces = []
    for j in range(len_segs + 1):
        z = -hl + length * (j / len_segs)
        for px, py in profile:
            verts.append([px, py, z])
    for j in range(len_segs):
        for i in range(P):
            ni = (i + 1) % P
            a = j * P + i
            b = j * P + ni
            c = (j + 1) * P + ni
            d = (j + 1) * P + i
            faces.append([a, b, c])
            faces.append([a, c, d])
    # End caps at z=-hl and z=+hl (fan between outer and inner rings)
    n_arc = arc_segs + 1
    for k in range(arc_segs):
        # z = -hl (normal -Z)
        o0, o1 = k, k + 1
        i0, i1 = P - 1 - k, P - 2 - k
        faces.append([o0, i0, i1])
        faces.append([o0, i1, o1])
        # z = +hl (normal +Z)
        off = len_segs * P
        faces.append([off + o0, off + o1, off + i1])
        faces.append([off + o0, off + i1, off + i0])
    v_arr = np.array(verts, dtype=np.float32)
    f_arr = np.array(faces, dtype=np.int32)
    # Center vertically around 0
    v_arr[:, 1] -= 0.5 * (v_arr[:, 1].min() + v_arr[:, 1].max())
    return _ensure_outward_winding(v_arr, f_arr)


def _ensure_outward_winding(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure positive signed volume (outward normals) for any closed watertight mesh."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    signed_vol6 = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))))
    if signed_vol6 < 0.0:
        faces = faces[:, [0, 2, 1]].copy()
    return verts, faces


def _normalize_mesh_unit(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize watertight mesh to fit in [-0.5, 0.5]^3 centered at origin, outward winding."""
    v = verts.astype(np.float32)
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    center = 0.5 * (vmin + vmax)
    extent = float(np.max(vmax - vmin))
    v = (v - center[None, :]) / max(extent, 1e-6)
    return _ensure_outward_winding(v, faces)


def _harmonic_sculpture(rng: np.random.Generator, nu: int = 36, nv: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Watertight organic spherical-harmonic sculpture (~1,700 tris)."""
    # Random harmonic modes
    f1 = int(rng.integers(2, 6))
    f2 = int(rng.integers(2, 6))
    f3 = int(rng.integers(1, 5))
    a1 = float(rng.uniform(0.08, 0.26))
    a2 = float(rng.uniform(0.06, 0.22))
    a3 = float(rng.uniform(0.05, 0.18))
    p1 = float(rng.uniform(0.0, 2.0 * math.pi))
    p2 = float(rng.uniform(0.0, 2.0 * math.pi))
    sx = float(rng.uniform(0.75, 1.3))
    sy = float(rng.uniform(0.85, 1.45))
    sz = float(rng.uniform(0.75, 1.3))

    verts = []
    # South pole & north pole for exact watertight topology
    def eval_pt(theta: float, phi: float) -> list[float]:
        r = (
            1.0
            + a1 * math.sin(f1 * theta) * math.cos(f2 * phi + p1)
            + a2 * math.cos(f3 * theta + p2) * math.sin(f1 * phi)
            + a3 * math.sin(2.0 * theta) * math.cos(phi)
        )
        r = max(0.35, r)
        st, ct = math.sin(theta), math.cos(theta)
        return [sx * r * st * math.cos(phi), sy * r * ct, sz * r * st * math.sin(phi)]

    north_idx = 0
    verts.append(eval_pt(0.0, 0.0))
    for j in range(1, nv):
        theta = math.pi * j / nv
        for i in range(nu):
            phi = 2.0 * math.pi * i / nu
            verts.append(eval_pt(theta, phi))
    south_idx = len(verts)
    verts.append(eval_pt(math.pi, 0.0))

    faces = []
    # Top cap
    for i in range(nu):
        ni = (i + 1) % nu
        faces.append([north_idx, 1 + ni, 1 + i])
    # Middle bands
    for j in range(nv - 2):
        r0 = 1 + j * nu
        r1 = 1 + (j + 1) * nu
        for i in range(nu):
            ni = (i + 1) % nu
            faces.append([r0 + i, r0 + ni, r1 + ni])
            faces.append([r0 + i, r1 + ni, r1 + i])
    # Bottom cap
    last_ring = 1 + (nv - 2) * nu
    for i in range(nu):
        ni = (i + 1) % nu
        faces.append([south_idx, last_ring + i, last_ring + ni])

    return _normalize_mesh_unit(np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32))


def _torus_knot_sculpture(rng: np.random.Generator, segs_u: int = 72, segs_v: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Watertight knotted torus / ring sculpture (~2,300 tris)."""
    p = int(rng.choice([1, 2, 2, 3, 3, 4]))
    q = int(rng.choice([1, 3, 5, 2, 4, 3]))
    if math.gcd(p, q) > 1 and p > 1:
        q = p + 1
    tube_r = float(rng.uniform(0.16, 0.28))
    major_r = float(rng.uniform(0.55, 0.78))
    bump = float(rng.uniform(0.0, 0.22))

    def curve(t: float) -> np.ndarray:
        if p == 1:
            return np.array([major_r * math.cos(t), 0.12 * math.sin(2 * t), major_r * math.sin(t)], dtype=np.float64)
        r = major_r + 0.28 * math.cos(q * t)
        return np.array([r * math.cos(p * t), 0.32 * math.sin(q * t), r * math.sin(p * t)], dtype=np.float64)

    # Parallel-transport Frenet frame along closed curve
    pts = [curve(2.0 * math.pi * i / segs_u) for i in range(segs_u)]
    verts = []
    for i in range(segs_u):
        p_prev = pts[(i - 1) % segs_u]
        p_curr = pts[i]
        p_next = pts[(i + 1) % segs_u]
        tang = p_next - p_prev
        tang /= max(np.linalg.norm(tang), 1e-8)
        up = np.array([0.0, 1.0, 0.0]) if abs(tang[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        binorm = np.cross(tang, up)
        binorm /= max(np.linalg.norm(binorm), 1e-8)
        norm = np.cross(binorm, tang)
        for j in range(segs_v):
            a = 2.0 * math.pi * j / segs_v
            rr = tube_r * (1.0 + bump * math.cos(3.0 * a))
            pt = p_curr + rr * (math.cos(a) * norm + math.sin(a) * binorm)
            verts.append(pt)
    faces = []
    for i in range(segs_u):
        ni = (i + 1) % segs_u
        for j in range(segs_v):
            nj = (j + 1) % segs_v
            a = i * segs_v + j
            b = ni * segs_v + j
            c = ni * segs_v + nj
            d = i * segs_v + nj
            faces.append([a, b, c])
            faces.append([a, c, d])
    return _normalize_mesh_unit(np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32))


def _lathe_vessel_mesh(rng: np.random.Generator, radial: int = 32, stacks: int = 26) -> tuple[np.ndarray, np.ndarray]:
    """Watertight ceramic vase / urn / pedestal / lampshade (~1,700 tris)."""
    c0 = float(rng.uniform(0.18, 0.42))
    c1 = float(rng.uniform(0.10, 0.35))
    c2 = float(rng.uniform(-0.18, 0.22))
    freq1 = float(rng.uniform(1.0, 3.0))
    freq2 = float(rng.uniform(2.0, 5.0))
    ph1 = float(rng.uniform(0.0, math.pi))
    twist = float(rng.uniform(0.0, 0.12))
    lobe_k = int(rng.integers(3, 8))

    verts = []
    # Bottom center
    bot_idx = 0
    verts.append([0.0, -0.5, 0.0])
    for j in range(stacks + 1):
        t = j / stacks
        y = -0.5 + t
        r = max(0.08, c0 + c1 * math.sin(math.pi * freq1 * t + ph1) + c2 * math.cos(math.pi * freq2 * t))
        if j == 0 or j == stacks:
            r *= 0.82
        for i in range(radial):
            a = 2.0 * math.pi * i / radial
            rr = r * (1.0 + twist * math.cos(lobe_k * a + 3.0 * t))
            verts.append([rr * math.cos(a), y, rr * math.sin(a)])
    top_idx = len(verts)
    verts.append([0.0, 0.5, 0.0])

    faces = []
    for i in range(radial):
        ni = (i + 1) % radial
        faces.append([bot_idx, 1 + i, 1 + ni])
    for j in range(stacks):
        r0 = 1 + j * radial
        r1 = 1 + (j + 1) * radial
        for i in range(radial):
            ni = (i + 1) % radial
            faces.append([r0 + i, r1 + i, r1 + ni])
            faces.append([r0 + i, r1 + ni, r0 + ni])
    top_ring = 1 + stacks * radial
    for i in range(radial):
        ni = (i + 1) % radial
        faces.append([top_idx, top_ring + ni, top_ring + i])
    return _normalize_mesh_unit(np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32))


def _superquadric_mesh(rng: np.random.Generator, nu: int = 32, nv: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Watertight superellipsoid / bevelled monolith / cushion (~1,300 tris)."""
    e1 = float(rng.uniform(0.25, 1.9))
    e2 = float(rng.uniform(0.25, 1.9))
    sx = float(rng.uniform(0.6, 1.4))
    sy = float(rng.uniform(0.6, 1.6))
    sz = float(rng.uniform(0.6, 1.4))

    def spow(x: float, e: float) -> float:
        return math.copysign(abs(x) ** e, x)

    verts = [[0.0, sy, 0.0]]
    for j in range(1, nv):
        eta = 0.5 * math.pi - math.pi * (j / nv)
        ce, se = spow(math.cos(eta), e1), spow(math.sin(eta), e1)
        for i in range(nu):
            om = -math.pi + 2.0 * math.pi * (i / nu)
            verts.append([sx * ce * spow(math.cos(om), e2), sy * se, sz * ce * spow(math.sin(om), e2)])
    south = len(verts)
    verts.append([0.0, -sy, 0.0])

    faces = []
    for i in range(nu):
        ni = (i + 1) % nu
        faces.append([0, 1 + ni, 1 + i])
    for j in range(nv - 2):
        r0 = 1 + j * nu
        r1 = 1 + (j + 1) * nu
        for i in range(nu):
            ni = (i + 1) % nu
            faces.append([r0 + i, r0 + ni, r1 + ni])
            faces.append([r0 + i, r1 + ni, r1 + i])
    last = 1 + (nv - 2) * nu
    for i in range(nu):
        ni = (i + 1) % nu
        faces.append([south, last + i, last + ni])
    return _normalize_mesh_unit(np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32))


def _compound_furniture_mesh(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Parametric watertight architectural/interior furniture assemblies."""
    parts = []
    if kind == "table":
        w, d, h = rng.uniform(0.8, 1.6), rng.uniform(0.6, 1.1), rng.uniform(0.65, 0.85)
        t = 0.07
        leg = 0.07
        parts.append(_translate_mesh(*_box_mesh(w, t, d, 2), (0, h - 0.5 * t, 0)))
        for sx in (-1, 1):
            for sz in (-1, 1):
                parts.append(
                    _translate_mesh(
                        *_box_mesh(leg, h - t, leg, 1),
                        (sx * (0.5 * w - leg), 0.5 * (h - t), sz * (0.5 * d - leg)),
                    )
                )
    elif kind == "chair":
        w, d, sh, bh = 0.48, 0.48, 0.46, 0.90
        parts.append(_translate_mesh(*_box_mesh(w, 0.06, d, 2), (0, sh, 0)))
        parts.append(_translate_mesh(*_box_mesh(w, bh - sh, 0.06, 2), (0, 0.5 * (bh + sh), -0.5 * d + 0.03)))
        for sx in (-1, 1):
            for sz in (-1, 1):
                parts.append(_translate_mesh(*_box_mesh(0.05, sh, 0.05, 1), (sx * 0.20, 0.5 * sh, sz * 0.20)))
    elif kind == "shelf":
        w, d, h = rng.uniform(0.9, 1.6), rng.uniform(0.32, 0.45), rng.uniform(1.4, 2.2)
        n_tiers = int(rng.integers(3, 6))
        parts.append(_translate_mesh(*_box_mesh(0.05, h, d, 2), (-0.5 * w + 0.025, 0.5 * h, 0)))
        parts.append(_translate_mesh(*_box_mesh(0.05, h, d, 2), (0.5 * w - 0.025, 0.5 * h, 0)))
        parts.append(_translate_mesh(*_box_mesh(w, h, 0.04, 2), (0, 0.5 * h, -0.5 * d + 0.02)))
        for k in range(n_tiers):
            yy = 0.05 + (h - 0.1) * (k / max(n_tiers - 1, 1))
            parts.append(_translate_mesh(*_box_mesh(w - 0.08, 0.04, d, 1), (0, yy, 0)))
    elif kind == "arch":
        span = rng.uniform(1.2, 2.2)
        h = rng.uniform(1.8, 2.6)
        thick = 0.22
        parts.append(_translate_mesh(*_box_mesh(0.28, h * 0.65, thick, 2), (-0.5 * span, h * 0.325, 0)))
        parts.append(_translate_mesh(*_box_mesh(0.28, h * 0.65, thick, 2), (0.5 * span, h * 0.325, 0)))
        cv, cf = _curved_shell_mesh(radius=0.5 * span, thickness=0.26, arc_angle=math.pi, length=thick, arc_segs=14, len_segs=2)
        parts.append(_translate_mesh(cv, cf, (0, h * 0.65 + 0.25 * span, 0)))
    elif kind == "plinth":
        w = rng.uniform(0.45, 0.95)
        h = rng.uniform(0.55, 1.25)
        parts.append(_translate_mesh(*_box_mesh(w, h, w, 2), (0, 0.5 * h, 0)))
        parts.append(_translate_mesh(*_box_mesh(w * 1.12, 0.08, w * 1.12, 1), (0, 0.04, 0)))
        parts.append(_translate_mesh(*_box_mesh(w * 1.08, 0.07, w * 1.08, 1), (0, h + 0.035, 0)))
    elif kind == "l_baffle":
        w1, w2, h = rng.uniform(0.9, 1.8), rng.uniform(0.6, 1.4), rng.uniform(1.4, 2.4)
        parts.append(_translate_mesh(*_box_mesh(w1, h, 0.12, 2), (0, 0.5 * h, 0)))
        parts.append(_translate_mesh(*_box_mesh(0.12, h, w2, 2), (0.5 * w1 - 0.06, 0.5 * h, 0.5 * w2)))
    else:  # stepped bench / credenza
        w, d, h = rng.uniform(1.0, 1.8), rng.uniform(0.4, 0.6), rng.uniform(0.45, 0.85)
        parts.append(_translate_mesh(*_box_mesh(w, h * 0.65, d, 2), (0, h * 0.675, 0)))
        parts.append(_translate_mesh(*_box_mesh(0.08, h * 0.35, d * 0.85, 1), (-0.42 * w, h * 0.175, 0)))
        parts.append(_translate_mesh(*_box_mesh(0.08, h * 0.35, d * 0.85, 1), (0.42 * w, h * 0.175, 0)))
    v, f = _merge_meshes(parts)
    return _normalize_mesh_unit(v, f)


# ---------------------------------------------------------------------------
# Archetype Library (strictly disjoint train vs. unseen held-out split)
# ---------------------------------------------------------------------------

def _try_fetch_cc0_watertight_meshes(cache_dir: str = "/content/cc0_meshes") -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Ephemerally fetch compact real CC0/public-domain watertight OBJ meshes on Colab (safe fallback)."""
    import urllib.request
    from pathlib import Path

    # Classic public-domain / CC0 geometry repository raw OBJ endpoints (~1k-5k tris, tiny downloads)
    sources = [
        ("cc0_bunny", "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj"),
        ("cc0_teapot", "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/teapot.obj"),
        ("cc0_spot", "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/spot.obj"),
        ("cc0_suzanne", "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/suzanne.obj"),
        ("cc0_cow", "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/cow.obj"),
        ("cc0_rocker_arm", "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/rocker-arm.obj"),
        ("cc0_beetle", "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/beetle.obj"),
        ("cc0_fandisk", "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/fandisk.obj"),
        ("cc0_bob", "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/bob.obj"),
        ("cc0_blub", "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/blub.obj"),
    ]
    cdir = Path(cache_dir)
    loaded: list[tuple[str, np.ndarray, np.ndarray]] = []
    try:
        cdir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return loaded

    for name, url in sources:
        local_f = cdir / f"{name}.obj"
        try:
            if not local_f.exists():
                req = urllib.request.Request(url, headers={"User-Agent": "THREE-RADIANT-GI/1.0"})
                with urllib.request.urlopen(req, timeout=6) as resp:
                    local_f.write_bytes(resp.read())
            verts, faces = [], []
            for line in local_f.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    toks = line.split()[1:]
                    idxs = [int(t.split("/")[0]) - 1 for t in toks]
                    for k in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[k], idxs[k + 1]])
            if len(verts) >= 50 and len(faces) >= 100:
                v_arr = np.array(verts, dtype=np.float32)
                f_arr = np.array(faces, dtype=np.int32)
                # Keep compact (< 6000 tris) to preserve fast Warp BVH speed
                if f_arr.shape[0] <= 12000 and int(f_arr.min()) >= 0 and int(f_arr.max()) < v_arr.shape[0]:
                    v_norm, f_norm = _normalize_mesh_unit(v_arr, f_arr)
                    loaded.append((name, v_norm, f_norm))
        except Exception:
            continue
    return loaded


class ArchetypeLibrary:
    """Pre-clustered Multi-LOD (4, 8, 16, 32, 64) local templates for architecture + 200+ watertight meshes."""

    def __init__(self, seed: int = 2026, fetch_cc0: bool = True, multi_lod: bool = True) -> None:
        from concurrent.futures import ThreadPoolExecutor

        rng = np.random.default_rng(seed)
        self.lod_counts: tuple[int, ...] = LOD_PATCH_COUNTS if multi_lod else (PATCHES_PER_OBJECT,)
        self.lod_bank: dict[str, dict[int, LocalMeshTemplate]] = {}

        # Gather all raw meshes first so multi-LOD clustering parallelizes cleanly across CPU cores
        arch_specs = [
            ("slab", *_box_mesh(1.0, 1.0, 1.0, subdiv=4)),
            ("col_round", *_cylinder_mesh(0.5, 0.5, 1.0, 20, 4)),
            ("col_fluted", *_cylinder_mesh(0.5, 0.46, 1.0, 24, 4, flute_amp=0.07, flute_freq=10)),
            ("vault", *_curved_shell_mesh(1.0, 0.14, math.pi * 0.88, 2.0, 16, 4)),
            ("reflector", *_curved_shell_mesh(1.0, 0.10, math.pi * 0.55, 1.5, 12, 3)),
        ]

        prop_specs: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
        furniture_kinds = ["table", "chair", "shelf", "arch", "plinth", "l_baffle", "bench"]
        for i in range(200):
            r = i % 5
            if r == 0:
                v, f = _harmonic_sculpture(rng)
                tag = f"harmonic_{i}"
            elif r == 1:
                v, f = _torus_knot_sculpture(rng)
                tag = f"knot_{i}"
            elif r == 2:
                v, f = _lathe_vessel_mesh(rng)
                tag = f"vessel_{i}"
            elif r == 3:
                v, f = _superquadric_mesh(rng)
                tag = f"superquad_{i}"
            else:
                kind = furniture_kinds[(i // 5) % len(furniture_kinds)]
                v, f = _compound_furniture_mesh(kind, rng)
                tag = f"{kind}_{i}"
            prop_specs.append((tag, v, f, i >= 160))

        # Ephemerally load real CC0/public-domain watertight meshes (80% train / 20% strictly held-out!)
        self.n_cc0_loaded = 0
        if fetch_cc0:
            cc0_meshes = _try_fetch_cc0_watertight_meshes()
            self.n_cc0_loaded = len(cc0_meshes)
            for idx, (cname, cv, cf) in enumerate(cc0_meshes):
                prop_specs.append((cname, cv, cf, (idx % 5 == 0)))

        all_jobs: list[tuple[str, np.ndarray, np.ndarray]] = list(arch_specs) + [
            (tag, v, f) for (tag, v, f, _is_ho) in prop_specs
        ]

        def _cluster_one(item: tuple[str, np.ndarray, np.ndarray]) -> tuple[str, dict[int, LocalMeshTemplate]]:
            nm, vv, ff = item
            return nm, build_multi_lod_templates(nm, vv, ff, lod_counts=self.lod_counts)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for nm, bank in pool.map(_cluster_one, all_jobs):
                self.lod_bank[nm] = bank

        self.slab = self.lod_bank["slab"][PATCHES_PER_OBJECT]
        self.column_round = self.lod_bank["col_round"][PATCHES_PER_OBJECT]
        self.column_fluted = self.lod_bank["col_fluted"][PATCHES_PER_OBJECT]
        self.vault_shell = self.lod_bank["vault"][PATCHES_PER_OBJECT]
        self.curved_reflector = self.lod_bank["reflector"][PATCHES_PER_OBJECT]

        self.train_props: list[LocalMeshTemplate] = []
        self.heldout_props: list[LocalMeshTemplate] = []
        for tag, _v, _f, is_ho in prop_specs:
            tmpl32 = self.lod_bank[tag][PATCHES_PER_OBJECT]
            if is_ho:
                self.heldout_props.append(tmpl32)
            else:
                self.train_props.append(tmpl32)

    def with_lod(self, tmpl: LocalMeshTemplate, k: int) -> LocalMeshTemplate:
        """Return the pre-clustered LocalMeshTemplate for the same archetype at patch count k."""
        bank = self.lod_bank.get(tmpl.name)
        if bank is not None and k in bank:
            return bank[k]
        return tmpl


# ---------------------------------------------------------------------------
# Color palettes & materials
# ---------------------------------------------------------------------------

SATURATED_BOUNCE_COLORS = [
    (0.84, 0.11, 0.09),  # Cornell crimson red
    (0.11, 0.78, 0.16),  # Cornell emerald green
    (0.10, 0.28, 0.86),  # Cobalt blue
    (0.88, 0.56, 0.08),  # Warm amber / ochre
    (0.78, 0.14, 0.58),  # Magenta / berry
    (0.08, 0.72, 0.76),  # Teal / cyan
    (0.86, 0.30, 0.10),  # Terracotta orange
    (0.46, 0.16, 0.80),  # Indigo / violet
    (0.76, 0.78, 0.10),  # Chartreuse / mustard
]

NEUTRAL_WALL_COLORS = [
    (0.78, 0.76, 0.73),  # Warm gallery plaster
    (0.74, 0.76, 0.79),  # Cool concrete white
    (0.62, 0.60, 0.58),  # Mid brutalist concrete
    (0.84, 0.82, 0.78),  # Bright chalk white
    (0.54, 0.44, 0.34),  # Warm oak / timber
    (0.42, 0.43, 0.45),  # Slate stone
    (0.70, 0.65, 0.58),  # Travertine limestone
]

EMITTER_TINTS = [
    (1.00, 0.94, 0.84),  # Warm tungsten / gallery 3500K
    (1.00, 0.98, 0.96),  # Neutral daylight 5500K
    (0.85, 0.93, 1.00),  # Cool skylight 7500K
    (1.00, 0.80, 0.58),  # Golden hour / sodium warm
    (0.95, 0.88, 1.00),  # Soft studio lavender-white
]


@dataclass
class ObjectInstance:
    template: LocalMeshTemplate
    scale: tuple[float, float, float]
    rot_y: float
    rot_x: float
    rot_z: float
    pos: tuple[float, float, float]
    albedo: tuple[float, float, float]
    roughness: float
    metallic: float
    emission: tuple[float, float, float]


def _euler_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (Ry @ Rx @ Rz).astype(np.float32)


def _sample_neutral(rng: np.random.Generator) -> tuple[float, float, float]:
    base = np.array(NEUTRAL_WALL_COLORS[int(rng.integers(0, len(NEUTRAL_WALL_COLORS)))], dtype=np.float32)
    jitter = rng.uniform(-0.07, 0.07, size=3).astype(np.float32)
    c = np.clip(base + jitter, 0.12, 0.88)
    return (float(c[0]), float(c[1]), float(c[2]))


def _sample_saturated(rng: np.random.Generator) -> tuple[float, float, float]:
    base = np.array(SATURATED_BOUNCE_COLORS[int(rng.integers(0, len(SATURATED_BOUNCE_COLORS)))], dtype=np.float32)
    jitter = rng.uniform(-0.06, 0.06, size=3).astype(np.float32)
    c = np.clip(base + jitter, 0.05, 0.90)
    return (float(c[0]), float(c[1]), float(c[2]))


def _sample_emitter_rgb(rng: np.random.Generator, strength_range: tuple[float, float] = (5.0, 18.0)) -> tuple[float, float, float]:
    tint = np.array(EMITTER_TINTS[int(rng.integers(0, len(EMITTER_TINTS)))], dtype=np.float32)
    strength = float(rng.uniform(*strength_range))
    c = tint * strength
    return (float(c[0]), float(c[1]), float(c[2]))


def _add_room_enclosure(
    objs: list[ObjectInstance],
    lib: ArchetypeLibrary,
    W: float,
    H: float,
    D: float,
    rng: np.random.Generator,
    *,
    vaulted: bool = False,
    open_front: bool = False,
    skylight_gap: bool = False,
    cornell_bias: bool = False,
) -> None:
    """Build watertight thick-slab room shell centered at (0, H/2, 0), floor top at y=0."""
    t = 0.18
    floor_c = _sample_neutral(rng)
    ceil_c = _sample_neutral(rng)
    back_c = _sample_neutral(rng)
    front_c = _sample_neutral(rng)

    if cornell_bias or rng.random() < 0.72:
        left_c = _sample_saturated(rng)
    else:
        left_c = _sample_neutral(rng)
    if cornell_bias or rng.random() < 0.72:
        right_c = _sample_saturated(rng)
    else:
        right_c = _sample_neutral(rng)
    if rng.random() < 0.30:
        back_c = _sample_saturated(rng)

    # Floor slab (top at y=0)
    objs.append(ObjectInstance(lib.slab, (W + 2 * t, t, D + 2 * t), 0, 0, 0, (0, -0.5 * t, 0), floor_c, float(rng.uniform(0.55, 0.95)), 0.0, (0, 0, 0)))

    # Ceiling: solid, vaulted, or skylight split
    if vaulted:
        objs.append(ObjectInstance(lib.vault_shell, (W * 0.55, H * 0.42, D * 0.54), 0, 0, 0, (0, H * 0.88, 0), ceil_c, 0.85, 0.0, (0, 0, 0)))
    elif skylight_gap:
        # Two ceiling wings with a luminous skylight well in the middle
        wing = W * 0.34
        objs.append(ObjectInstance(lib.slab, (wing, t, D + 2 * t), 0, 0, 0, (-0.5 * W + 0.5 * wing, H + 0.5 * t, 0), ceil_c, 0.88, 0.0, (0, 0, 0)))
        objs.append(ObjectInstance(lib.slab, (wing, t, D + 2 * t), 0, 0, 0, (0.5 * W - 0.5 * wing, H + 0.5 * t, 0), ceil_c, 0.88, 0.0, (0, 0, 0)))
        # Skylight emitter panel above the gap
        sky_em = _sample_emitter_rgb(rng, (6.0, 16.0))
        objs.append(ObjectInstance(lib.slab, (W * 0.36, 0.10, D * 0.75), 0, 0, 0, (0, H + 0.42, 0), (0.9, 0.9, 0.9), 0.9, 0.0, sky_em))
    else:
        objs.append(ObjectInstance(lib.slab, (W + 2 * t, t, D + 2 * t), 0, 0, 0, (0, H + 0.5 * t, 0), ceil_c, 0.90, 0.0, (0, 0, 0)))

    # Left & Right walls
    objs.append(ObjectInstance(lib.slab, (t, H, D + 2 * t), 0, 0, 0, (-0.5 * W - 0.5 * t, 0.5 * H, 0), left_c, float(rng.uniform(0.65, 0.98)), 0.0, (0, 0, 0)))
    objs.append(ObjectInstance(lib.slab, (t, H, D + 2 * t), 0, 0, 0, (0.5 * W + 0.5 * t, 0.5 * H, 0), right_c, float(rng.uniform(0.65, 0.98)), 0.0, (0, 0, 0)))
    # Back wall
    objs.append(ObjectInstance(lib.slab, (W, H, t), 0, 0, 0, (0, 0.5 * H, -0.5 * D - 0.5 * t), back_c, float(rng.uniform(0.65, 0.98)), 0.0, (0, 0, 0)))
    # Front wall (optional for open studio / camera wall)
    if not open_front:
        objs.append(ObjectInstance(lib.slab, (W, H, t), 0, 0, 0, (0, 0.5 * H, 0.5 * D + 0.5 * t), front_c, 0.88, 0.0, (0, 0, 0)))


def apply_scene_lod(
    objs: list[ObjectInstance],
    lib: ArchetypeLibrary,
    rng: np.random.Generator,
    mode: str = "distance",
    W: float = 6.0,
    H: float = 4.0,
    D: float = 6.0,
    max_tokens: int = 680,
) -> tuple[list[ObjectInstance], list[int]]:
    """Assign per-object Distance Patch LOD (4, 8, 16, 32, 64) preserving exact geometry & surface area."""
    if mode == "fixed32" or len(lib.lod_counts) <= 1:
        return objs, [inst.template.n_patches for inst in objs]

    lods: list[int] = []
    if mode == "distance":
        # Sample realistic virtual camera viewpoint (front-outside looking in, room corner, or interior orbit)
        u = float(rng.random())
        if u < 0.45:
            cam = np.array([rng.uniform(-0.35 * W, 0.35 * W), rng.uniform(0.35 * H, 0.65 * H), rng.uniform(0.45 * D, 1.35 * D)], dtype=np.float32)
        elif u < 0.80:
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            rad = float(rng.uniform(0.35, 0.95)) * max(W, D)
            cam = np.array([math.cos(ang) * rad, rng.uniform(0.25 * H, 0.75 * H), math.sin(ang) * rad], dtype=np.float32)
        else:
            cam = np.array([rng.uniform(-0.4 * W, 0.4 * W), rng.uniform(0.25 * H, 0.70 * H), rng.uniform(-0.4 * D, 0.4 * D)], dtype=np.float32)

        for inst in objs:
            pos = np.array(inst.pos, dtype=np.float32)
            rad = 0.5 * float(max(inst.scale))
            dist = max(float(np.linalg.norm(pos - cam)), 0.35)
            # Screen-space projected angular coverage with mild jitter so model generalizes across thresholds
            cov = (rad / dist) * float(rng.uniform(0.78, 1.32))
            is_em = max(inst.emission) > 1e-3
            if is_em:
                # Emitters stay sharp (16-32) when near, or 8 when distant
                k = 32 if cov >= 0.22 else (16 if cov >= 0.11 else 8)
            elif cov >= 0.58:
                k = 64 if (64 in lib.lod_counts and rng.random() < 0.42) else 32
            elif cov >= 0.28:
                k = 32
            elif cov >= 0.16:
                k = 16
            elif cov >= 0.095:
                k = 8
            else:
                k = 4
            lods.append(k)
    elif mode == "random_mixed":
        choices = np.array([4, 8, 16, 32, 64], dtype=np.int32)
        probs = np.array([0.16, 0.22, 0.26, 0.26, 0.10], dtype=np.float64)
        lods = [int(rng.choice(choices, p=probs)) for _ in objs]
    else:
        lods = [PATCHES_PER_OBJECT for _ in objs]

    # Cap total scene tokens gracefully by halving highest-LOD objects if over budget
    while sum(lods) > max_tokens:
        mx = max(lods)
        if mx <= 4:
            break
        idx = lods.index(mx)
        lods[idx] = mx // 2

    out_objs: list[ObjectInstance] = []
    for inst, k in zip(objs, lods):
        out_objs.append(
            ObjectInstance(
                template=lib.with_lod(inst.template, k),
                scale=inst.scale,
                rot_y=inst.rot_y,
                rot_x=inst.rot_x,
                rot_z=inst.rot_z,
                pos=inst.pos,
                albedo=inst.albedo,
                roughness=inst.roughness,
                metallic=inst.metallic,
                emission=inst.emission,
            )
        )
    return out_objs, lods


def generate_scene_spec(
    lib: ArchetypeLibrary,
    rng: np.random.Generator,
    *,
    split: str = "train",
    topology_override: str | None = None,
    lod_mode: str = "fixed32",
) -> tuple[list[ObjectInstance], dict]:
    """Generate a diverse multi-object 3D room (8-24 objects) with optional Mixed Distance Patch LOD."""
    is_heldout = split == "heldout"
    prop_pool = lib.heldout_props if is_heldout else lib.train_props

    train_topos = [
        "cornell_box",
        "rectangular_interior",
        "l_corridor",
        "vaulted_gallery",
        "doorway_chambers",
        "skylight_studio",
        "colonnade_hall",
        "baffle_occlusion",
        "sunlit_window_hall",
        "compact_alcove",
        "grand_hall",
    ]
    heldout_topos = [
        "unseen_cornell_sculptural",
        "unseen_vaulted_atrium",
        "unseen_split_doorway",
        "unseen_diagonal_gallery",
    ]
    if topology_override is not None:
        topo = topology_override
    else:
        pool = heldout_topos if is_heldout else train_topos
        topo = str(rng.choice(pool))

    objs: list[ObjectInstance] = []
    W = float(rng.uniform(4.0, 8.5))
    H = float(rng.uniform(2.8, 4.8))
    D = float(rng.uniform(4.0, 8.5))

    if topo == "compact_alcove":
        W = float(rng.uniform(2.2, 3.8))
        H = float(rng.uniform(2.2, 3.0))
        D = float(rng.uniform(2.2, 3.8))
        _add_room_enclosure(objs, lib, W, H, D, rng, cornell_bias=True)
    elif topo == "grand_hall":
        W = float(rng.uniform(8.5, 13.5))
        H = float(rng.uniform(4.8, 7.0))
        D = float(rng.uniform(8.5, 13.5))
        _add_room_enclosure(objs, lib, W, H, D, rng, vaulted=(rng.random() < 0.4), skylight_gap=(rng.random() < 0.4))
    elif topo == "sunlit_window_hall":
        W = float(rng.uniform(5.2, 9.0))
        H = float(rng.uniform(3.2, 4.8))
        D = float(rng.uniform(5.2, 9.0))
        _add_room_enclosure(objs, lib, W, H, D, rng, cornell_bias=(rng.random() < 0.5))
        # High-intensity angled sunlit window portal on one side wall + bright floor sun-bounce patch!
        side = float(rng.choice([-1.0, 1.0]))
        sun_col = (26.0, 23.0, 17.5)
        objs.append(ObjectInstance(lib.slab, (0.10, H * 0.52, D * 0.38), 0, 0, 0, (side * (0.5 * W - 0.06), H * 0.58, float(rng.uniform(-0.15 * D, 0.15 * D))), (0.98, 0.96, 0.92), 0.9, 0.0, sun_col))
    elif topo in ("cornell_box", "unseen_cornell_sculptural"):
        W = float(rng.uniform(4.2, 6.2))
        H = float(rng.uniform(3.2, 4.5))
        D = float(rng.uniform(4.2, 6.2))
        _add_room_enclosure(objs, lib, W, H, D, rng, cornell_bias=True, open_front=(rng.random() < 0.25))
    elif topo in ("vaulted_gallery", "unseen_vaulted_atrium"):
        W = float(rng.uniform(5.0, 7.6))
        H = float(rng.uniform(3.6, 4.8))
        D = float(rng.uniform(5.5, 8.5))
        _add_room_enclosure(objs, lib, W, H, D, rng, vaulted=(topo == "vaulted_gallery"), skylight_gap=(topo == "unseen_vaulted_atrium"))
    elif topo == "skylight_studio":
        _add_room_enclosure(objs, lib, W, H, D, rng, skylight_gap=True)
    else:
        _add_room_enclosure(objs, lib, W, H, D, rng)

    # Architectural interior partitions / doorways / columns / curved reflectors
    if topo in ("doorway_chambers", "unseen_split_doorway"):
        # Partition wall with doorway opening between left/right or front/back halves
        z_part = float(rng.uniform(-0.12 * D, 0.12 * D))
        door_w = float(rng.uniform(1.2, 1.9))
        side_w = max(0.4, 0.5 * (W - door_w))
        p_col = _sample_saturated(rng) if rng.random() < 0.4 else _sample_neutral(rng)
        objs.append(ObjectInstance(lib.slab, (side_w, H, 0.18), 0, 0, 0, (-0.5 * W + 0.5 * side_w, 0.5 * H, z_part), p_col, 0.85, 0.0, (0, 0, 0)))
        objs.append(ObjectInstance(lib.slab, (side_w, H, 0.18), 0, 0, 0, (0.5 * W - 0.5 * side_w, 0.5 * H, z_part), p_col, 0.85, 0.0, (0, 0, 0)))
        lintel_h = max(0.35, H - 2.2)
        objs.append(ObjectInstance(lib.slab, (door_w, lintel_h, 0.18), 0, 0, 0, (0, H - 0.5 * lintel_h, z_part), p_col, 0.85, 0.0, (0, 0, 0)))
    elif topo == "l_corridor":
        # Solid corner block carving out an L-shaped room
        bx = float(rng.uniform(0.35 * W, 0.52 * W))
        bz = float(rng.uniform(0.35 * D, 0.52 * D))
        corner_sign_x = float(rng.choice([-1.0, 1.0]))
        corner_sign_z = float(rng.choice([-1.0, 1.0]))
        c_col = _sample_saturated(rng) if rng.random() < 0.55 else _sample_neutral(rng)
        objs.append(
            ObjectInstance(
                lib.slab,
                (bx, H, bz),
                0,
                0,
                0,
                (corner_sign_x * (0.5 * W - 0.5 * bx), 0.5 * H, corner_sign_z * (0.5 * D - 0.5 * bz)),
                c_col,
                0.85,
                0.0,
                (0, 0, 0),
            )
        )
    elif topo == "colonnade_hall":
        col_tmpl = lib.column_fluted if rng.random() < 0.5 else lib.column_round
        col_c = _sample_neutral(rng)
        r_col = float(rng.uniform(0.22, 0.36))
        for cx in (-0.24 * W, 0.24 * W):
            for cz in (-0.22 * D, 0.22 * D):
                objs.append(ObjectInstance(col_tmpl, (r_col * 2, H, r_col * 2), 0, 0, 0, (cx, 0.5 * H, cz), col_c, 0.78, 0.0, (0, 0, 0)))
    elif topo in ("baffle_occlusion", "unseen_diagonal_gallery"):
        # Interior solid occluding baffle wall / reflector
        bw = float(rng.uniform(0.38 * W, 0.62 * W))
        bh = float(rng.uniform(0.65 * H, 0.92 * H))
        ang = float(rng.uniform(-0.6, 0.6)) if topo == "unseen_diagonal_gallery" else float(rng.choice([0.0, math.pi * 0.5]))
        b_col = _sample_saturated(rng) if rng.random() < 0.6 else _sample_neutral(rng)
        objs.append(
            ObjectInstance(
                lib.slab,
                (bw, bh, 0.16),
                ang,
                0,
                0,
                (float(rng.uniform(-0.15 * W, 0.15 * W)), 0.5 * bh, float(rng.uniform(-0.15 * D, 0.15 * D))),
                b_col,
                0.82,
                0.0,
                (0, 0, 0),
            )
        )
        if rng.random() < 0.5:
            objs.append(
                ObjectInstance(
                    lib.curved_reflector,
                    (1.2, 1.2, 1.4),
                    float(rng.uniform(0, 2 * math.pi)),
                    0,
                    0,
                    (float(rng.uniform(-0.28 * W, 0.28 * W)), 1.0, float(rng.uniform(-0.28 * D, 0.28 * D))),
                    _sample_saturated(rng),
                    0.55,
                    0.0,
                    (0, 0, 0),
                )
            )

    # 1 to 3 Movable Area Lights (ceiling softboxes, wall bounce panels, or hanging lamps)
    n_lights = int(rng.integers(1, 4))
    for li in range(n_lights):
        lw_hi = max(0.55, min(2.4, 0.42 * W))
        ld_hi = max(0.55, min(2.4, 0.42 * D))
        lw = float(rng.uniform(min(0.75, 0.5 * lw_hi), lw_hi))
        ld = float(rng.uniform(min(0.75, 0.5 * ld_hi), ld_hi))
        lx = float(rng.uniform(-0.32 * W, 0.32 * W))
        lz = float(rng.uniform(-0.32 * D, 0.32 * D))
        em_rgb = _sample_emitter_rgb(rng, (6.0, 22.0) if n_lights == 1 else (4.0, 14.0))
        if li == 0 or rng.random() < 0.7:
            # Ceiling / hanging area lamp
            ly = H - float(rng.uniform(0.08, min(0.45, 0.16 * H)))
            objs.append(ObjectInstance(lib.slab, (lw, 0.08, ld), float(rng.uniform(-0.2, 0.2)), 0, 0, (lx, ly, lz), (0.95, 0.95, 0.95), 0.9, 0.0, em_rgb))
        else:
            # Vertical standing studio light panel / wall sconce
            lh_hi = max(0.85, min(2.2, 0.6 * H))
            lh = float(rng.uniform(min(0.9, 0.6 * lh_hi), lh_hi))
            objs.append(
                ObjectInstance(
                    lib.slab,
                    (lw * 0.7, lh, 0.08),
                    float(rng.uniform(0, 2 * math.pi)),
                    0,
                    0,
                    (lx, min(0.5 * lh + 0.3, H - 0.5 * lh - 0.1), lz),
                    (0.95, 0.95, 0.95),
                    0.9,
                    0.0,
                    em_rgb,
                )
            )

    # Resolve effective LOD mode ("auto" = 25% fixed32 + 55% distance LOD + 20% random mixed LOD)
    eff_lod = lod_mode
    if lod_mode == "auto":
        u_lod = float(rng.random())
        if u_lod < 0.25:
            eff_lod = "fixed32"
        elif u_lod < 0.80:
            eff_lod = "distance"
        else:
            eff_lod = "random_mixed"

    # Interior sculptural props & furniture (more objects allowed under Distance LOD since far props cost 4-8 tokens!)
    obj_cap = 24 if eff_lod in ("distance", "random_mixed") else 19
    prop_hi = 12 if eff_lod in ("distance", "random_mixed") else 9
    max_props = max(3, min(prop_hi, obj_cap - len(objs)))
    n_props = int(rng.integers(3, max_props + 1))
    placed_xy: list[tuple[float, float, float]] = []
    room_min_xz = min(W, D)
    s_hi = min(1.85, max(0.50, 0.25 * room_min_xz))
    s_lo = min(0.55, 0.55 * s_hi)
    for _ in range(n_props):
        tmpl = prop_pool[int(rng.integers(0, len(prop_pool)))]
        s_base = float(rng.uniform(s_lo, s_hi))
        sx = s_base * float(rng.uniform(0.8, 1.25))
        sy = min(s_base * float(rng.uniform(0.8, 1.45)), max(0.45, H - 0.35))
        sz = s_base * float(rng.uniform(0.8, 1.25))
        rad = 0.5 * max(sx, sz)
        hx = max(0.05, 0.40 * W - rad)
        hz = max(0.05, 0.40 * D - rad)
        # Non-overlapping placement with light rejection
        px, pz = 0.0, 0.0
        for _try in range(12):
            cx = float(rng.uniform(-hx, hx))
            cz = float(rng.uniform(-hz, hz))
            if all((cx - ox) ** 2 + (cz - oz) ** 2 > (rad + orad) ** 2 * 0.65 for ox, oz, orad in placed_xy):
                px, pz = cx, cz
                break
        placed_xy.append((px, pz, rad))
        py = min(0.5 * sy, max(0.25, H - 0.5 * sy - 0.12))
        if rng.random() < 0.35:
            p_alb = _sample_saturated(rng)
        else:
            p_alb = _sample_neutral(rng)
        metal = float(rng.uniform(0.65, 0.95)) if rng.random() < 0.16 else 0.0
        rough = float(rng.uniform(0.18, 0.45)) if metal > 0.5 else float(rng.uniform(0.45, 0.95))
        objs.append(
            ObjectInstance(
                template=tmpl,
                scale=(sx, sy, sz),
                rot_y=float(rng.uniform(0.0, 2.0 * math.pi)),
                rot_x=0.0,
                rot_z=0.0,
                pos=(px, py, pz),
                albedo=p_alb,
                roughness=rough,
                metallic=metal,
                emission=(0.0, 0.0, 0.0),
            )
        )

    objs, obj_lods = apply_scene_lod(objs, lib, rng, mode=eff_lod, W=W, H=H, D=D)
    meta = {
        "topology": topo,
        "split": split,
        "lod_mode": eff_lod,
        "dims": [round(W, 3), round(H, 3), round(D, 3)],
        "n_objects": len(objs),
        "n_tokens": int(sum(obj_lods)),
        "object_lods": obj_lods,
    }
    return objs, meta


def generate_dynamic_occlusion_pair(
    lib: ArchetypeLibrary,
    rng: np.random.Generator,
) -> tuple[list[ObjectInstance], list[ObjectInstance], dict]:
    """Generate a matched pair of UNSEEN rooms: (open, occluded) with a solid wall slid between light & target.

    Used in O3 to verify that sliding an unseen wall between emitter and target drops predicted
    indirect irradiance by >= 40% matching ground truth.
    """
    W, H, D = float(rng.uniform(5.2, 6.6)), float(rng.uniform(3.2, 4.0)), float(rng.uniform(5.0, 6.2))
    t = 0.18
    left_c = _sample_saturated(rng)
    right_c = _sample_neutral(rng)
    neutral = _sample_neutral(rng)

    base_objs: list[ObjectInstance] = [
        # Floor, ceiling, 4 outer walls
        ObjectInstance(lib.slab, (W + 2 * t, t, D + 2 * t), 0, 0, 0, (0, -0.5 * t, 0), neutral, 0.85, 0.0, (0, 0, 0)),
        ObjectInstance(lib.slab, (W + 2 * t, t, D + 2 * t), 0, 0, 0, (0, H + 0.5 * t, 0), neutral, 0.90, 0.0, (0, 0, 0)),
        ObjectInstance(lib.slab, (t, H, D + 2 * t), 0, 0, 0, (-0.5 * W - 0.5 * t, 0.5 * H, 0), left_c, 0.85, 0.0, (0, 0, 0)),
        ObjectInstance(lib.slab, (t, H, D + 2 * t), 0, 0, 0, (0.5 * W + 0.5 * t, 0.5 * H, 0), right_c, 0.85, 0.0, (0, 0, 0)),
        ObjectInstance(lib.slab, (W, H, t), 0, 0, 0, (0, 0.5 * H, -0.5 * D - 0.5 * t), neutral, 0.85, 0.0, (0, 0, 0)),
        ObjectInstance(lib.slab, (W, H, t), 0, 0, 0, (0, 0.5 * H, 0.5 * D + 0.5 * t), neutral, 0.85, 0.0, (0, 0, 0)),
        # Bright area emitter on the LEFT side of the room (x = -0.32 * W)
        ObjectInstance(lib.slab, (1.2, 0.10, 1.6), 0, 0, 0, (-0.32 * W, H - 0.16, 0.0), (0.95, 0.95, 0.95), 0.9, 0.0, (16.0, 14.5, 12.5)),
    ]

    # Target unseen sculpture on the RIGHT side (x = +0.30 * W) -> object index 7!
    target_tmpl = lib.heldout_props[int(rng.integers(0, len(lib.heldout_props)))]
    base_objs.append(
        ObjectInstance(
            target_tmpl,
            (1.1, 1.3, 1.1),
            float(rng.uniform(0, 2 * math.pi)),
            0,
            0,
            (0.30 * W, 0.65, 0.0),
            (0.80, 0.80, 0.80),
            0.80,
            0.0,
            (0, 0, 0),
        )
    )
    # Extra unseen prop on left side for rich bounce
    prop2 = lib.heldout_props[int(rng.integers(0, len(lib.heldout_props)))]
    base_objs.append(
        ObjectInstance(
            prop2,
            (0.95, 1.0, 0.95),
            float(rng.uniform(0, 2 * math.pi)),
            0,
            0,
            (-0.28 * W, 0.50, 0.22 * D),
            _sample_saturated(rng),
            0.75,
            0.0,
            (0, 0, 0),
        )
    )

    # Open scene: baffle wall tucked flat against the back-left corner (not blocking light -> target)
    open_objs = list(base_objs) + [
        ObjectInstance(lib.slab, (0.18, H * 0.95, D * 0.86), 0, 0, 0, (-0.47 * W, 0.475 * H, 0.0), (0.25, 0.25, 0.25), 0.9, 0.0, (0, 0, 0))
    ]
    # Occluded scene: SAME dark baffle wall slid right into the center x=0 separating emitter & target!
    occ_objs = list(base_objs) + [
        ObjectInstance(lib.slab, (0.18, H * 0.95, D * 0.86), 0, 0, 0, (0.0, 0.475 * H, 0.0), (0.25, 0.25, 0.25), 0.9, 0.0, (0, 0, 0))
    ]
    meta = {"topology": "dynamic_occlusion_pair", "target_object_id": 7, "dims": [round(W, 3), round(H, 3), round(D, 3)]}
    return open_objs, occ_objs, meta


def instantiate_scene_geometry(
    objs: list[ObjectInstance],
    *,
    world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Convert a list of ObjectInstances into flat mesh + 32-patch-per-object arrays ready for Warp."""
    ox, oy, oz = world_offset
    all_verts, all_faces, all_fn = [], [], []
    all_fa, all_fr, all_fm, all_fe = [], [], [], []
    p_pos, p_T, p_B, p_N, p_area = [], [], [], [], []
    p_alb, p_rgh, p_met, p_em, p_oid = [], [], [], [], []
    p_off, p_cnt, p_idx, p_cdf = [], [], [], []

    v_off = 0
    f_off = 0
    idx_off = 0

    for oid, inst in enumerate(objs):
        tmpl = inst.template
        R = _euler_matrix(inst.rot_x, inst.rot_y, inst.rot_z)
        nF = tmpl.faces.shape[0]

        v_world, fn, pdict = fast_transform_template_patches(
            tmpl=tmpl,
            scale=inst.scale,
            R=R,
            pos=inst.pos,
            world_offset=world_offset,
            albedo=inst.albedo,
            roughness=inst.roughness,
            metallic=inst.metallic,
            emission=inst.emission,
            object_id=oid,
            face_offset=f_off,
        )

        all_verts.append(v_world)
        all_faces.append(tmpl.faces + v_off)
        all_fn.append(fn)
        all_fa.append(np.broadcast_to(np.array(inst.albedo, dtype=np.float32), (nF, 3)).copy())
        all_fr.append(np.full((nF,), float(inst.roughness), dtype=np.float32))
        all_fm.append(np.full((nF,), float(inst.metallic), dtype=np.float32))
        all_fe.append(np.broadcast_to(np.array(inst.emission, dtype=np.float32), (nF, 3)).copy())

        p_pos.append(pdict["pos"])
        p_T.append(pdict["T"])
        p_B.append(pdict["B"])
        p_N.append(pdict["N"])
        p_area.append(pdict["area"])
        p_alb.append(pdict["albedo"])
        p_rgh.append(pdict["roughness"])
        p_met.append(pdict["metallic"])
        p_em.append(pdict["emission"])
        p_oid.append(pdict["object_id"])

        p_off.append(pdict["tri_offsets"] + idx_off)
        p_cnt.append(pdict["tri_counts"])
        p_idx.append(pdict["tri_indices"])
        p_cdf.append(pdict["tri_cdf"])

        v_off += v_world.shape[0]
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
        "patch_pos_world_offset": np.array([ox, oy, oz], dtype=np.float32),
        "patch_pos": np.concatenate(p_pos, axis=0),
        "patch_T": np.concatenate(p_T, axis=0),
        "patch_B": np.concatenate(p_B, axis=0),
        "patch_N": np.concatenate(p_N, axis=0),
        "patch_area": np.concatenate(p_area, axis=0),
        "patch_albedo": np.concatenate(p_alb, axis=0),
        "patch_roughness": np.concatenate(p_rgh, axis=0),
        "patch_metallic": np.concatenate(p_met, axis=0),
        "patch_emission": np.concatenate(p_em, axis=0),
        "patch_oid": np.concatenate(p_oid, axis=0),
    }
