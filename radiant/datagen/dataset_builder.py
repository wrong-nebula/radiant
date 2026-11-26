"""Batched NVIDIA Warp Dataset Builder for Colab A100 RAM.

Generates:
  - 32,000+ diverse multi-object training scenes (256 rays/patch x 3 bounces) directly in
    Colab RAM via mega-batched Warp GPU BVH ray tracing (~3-6 minutes on A100).
  - 256 strictly held-out UNSEEN test scenes (1,024 rays/patch x 3 bounces) using strictly
    disjoint unseen topologies & unseen sculptural meshes + 16 dynamic occlusion pairs,
    saved as a compact <= 25 MB compressed `.npz` verification bundle.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import warp as wp

from radiant.datagen.procedural_scenes import (
    ArchetypeLibrary,
    ObjectInstance,
    generate_dynamic_occlusion_pair,
    generate_scene_spec,
    instantiate_scene_geometry,
)
from radiant.datagen.warp_tracer import trace_scene_patches_warp


def _merge_mega_batch(scene_geoms: list[dict[str, np.ndarray]]) -> tuple[dict[str, np.ndarray], list[int]]:
    """Pack multiple spatially-disjoint scenes into a single GPU Warp BVH launch."""
    verts_l, faces_l, fn_l = [], [], []
    fa_l, fr_l, fm_l, fe_l = [], [], [], []
    off_l, cnt_l, idx_l, cdf_l = [], [], [], []
    ppos_l, pT_l, pB_l, pN_l = [], [], [], []
    patch_counts: list[int] = []

    v_off = 0
    f_off = 0
    i_off = 0

    for g in scene_geoms:
        nV = g["verts"].shape[0]
        nF = g["faces"].shape[0]
        nP = g["patch_pos"].shape[0]
        nI = g["patch_tri_indices"].shape[0]
        patch_counts.append(nP)

        verts_l.append(g["verts"])
        faces_l.append(g["faces"] + v_off)
        fn_l.append(g["face_normals"])
        fa_l.append(g["face_albedo"])
        fr_l.append(g["face_roughness"])
        fm_l.append(g["face_metallic"])
        fe_l.append(g["face_emission"])

        off_l.append(g["patch_tri_offsets"] + i_off)
        cnt_l.append(g["patch_tri_counts"])
        idx_l.append(g["patch_tri_indices"] + f_off)
        cdf_l.append(g["patch_tri_cdf"])

        # Add world_offset back for Warp ray origins!
        ppos_l.append(g["patch_pos"] + g["patch_pos_world_offset"][None, :])
        pT_l.append(g["patch_T"])
        pB_l.append(g["patch_B"])
        pN_l.append(g["patch_N"])

        v_off += nV
        f_off += nF
        i_off += nI

    merged = {
        "verts": np.concatenate(verts_l, axis=0),
        "faces": np.concatenate(faces_l, axis=0),
        "face_normals": np.concatenate(fn_l, axis=0),
        "face_albedo": np.concatenate(fa_l, axis=0),
        "face_roughness": np.concatenate(fr_l, axis=0),
        "face_metallic": np.concatenate(fm_l, axis=0),
        "face_emission": np.concatenate(fe_l, axis=0),
        "patch_tri_offsets": np.concatenate(off_l, axis=0),
        "patch_tri_counts": np.concatenate(cnt_l, axis=0),
        "patch_tri_indices": np.concatenate(idx_l, axis=0),
        "patch_tri_cdf": np.concatenate(cdf_l, axis=0),
        "patch_pos": np.concatenate(ppos_l, axis=0),
        "patch_T": np.concatenate(pT_l, axis=0),
        "patch_B": np.concatenate(pB_l, axis=0),
        "patch_N": np.concatenate(pN_l, axis=0),
    }
    return merged, patch_counts


def _pack_token_features(g: dict[str, np.ndarray]) -> np.ndarray:
    """Pack per-patch token fields into [N, 21] float32:
      [0:3]   pos (m)
      [3:6]   T
      [6:9]   B
      [9:12]  N
      [12]    area (m^2)
      [13:16] albedo (linear RGB)
      [16]    roughness
      [17]    metallic
      [18:21] emission (linear HDR RGB)
    """
    return np.concatenate(
        [
            g["patch_pos"],
            g["patch_T"],
            g["patch_B"],
            g["patch_N"],
            g["patch_area"][:, None],
            g["patch_albedo"],
            g["patch_roughness"][:, None],
            g["patch_metallic"][:, None],
            g["patch_emission"],
        ],
        axis=-1,
    ).astype(np.float32)


def trace_scene_specs_batched(
    specs: list[list[ObjectInstance]],
    *,
    rays_per_patch: int = 256,
    max_bounces: int = 3,
    mega_batch_size: int = 48,
    base_seed: int = 1000,
    device: str = "cuda:0",
    log_prefix: str = "datagen",
) -> dict[str, np.ndarray]:
    """Trace a list of scene specs in mega-batches on GPU and return packed CSR dataset in RAM."""
    wp.init()
    n_scenes = len(specs)
    all_tokens: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_oids: list[np.ndarray] = []
    scene_offsets = np.zeros((n_scenes + 1,), dtype=np.int64)

    t_start = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=8)

    def build_one(idx_and_objs: tuple[int, list[ObjectInstance]]) -> dict[str, np.ndarray]:
        local_idx, objs = idx_and_objs
        gx = local_idx % 8
        gy = local_idx // 8
        # Space scenes 80m apart; max_t=35m guarantees zero cross-scene rays!
        return instantiate_scene_geometry(objs, world_offset=(gx * 80.0, gy * 80.0, 0.0))

    cursor = 0
    for start in range(0, n_scenes, mega_batch_size):
        end = min(start + mega_batch_size, n_scenes)
        batch_specs = specs[start:end]
        geoms = list(pool.map(build_one, enumerate(batch_specs)))
        merged, p_counts = _merge_mega_batch(geoms)

        sh_batch = trace_scene_patches_warp(
            verts=merged["verts"],
            faces=merged["faces"],
            face_normals=merged["face_normals"],
            face_albedo=merged["face_albedo"],
            face_roughness=merged["face_roughness"],
            face_metallic=merged["face_metallic"],
            face_emission=merged["face_emission"],
            patch_tri_offsets=merged["patch_tri_offsets"],
            patch_tri_counts=merged["patch_tri_counts"],
            patch_tri_indices=merged["patch_tri_indices"],
            patch_tri_cdf=merged["patch_tri_cdf"],
            patch_pos=merged["patch_pos"],
            patch_T=merged["patch_T"],
            patch_B=merged["patch_B"],
            patch_N=merged["patch_N"],
            rays_per_patch=rays_per_patch,
            max_bounces=max_bounces,
            max_t=35.0,
            seed=base_seed + start,
            include_direct_bounce0=False,
            device=device,
        )

        p_off = 0
        for local_i, (g, cnt) in enumerate(zip(geoms, p_counts)):
            all_tokens.append(_pack_token_features(g).astype(np.float16))
            all_targets.append(sh_batch[p_off : p_off + cnt].astype(np.float16))
            all_oids.append(g["patch_oid"].astype(np.int16))
            p_off += cnt
            cursor += cnt
            scene_offsets[start + local_i + 1] = cursor

        if (end % (mega_batch_size * 4) == 0) or end == n_scenes or start == 0:
            elapsed = time.perf_counter() - t_start
            rate = end / max(elapsed, 1e-6)
            print(
                f"[{log_prefix}] {end}/{n_scenes} scenes traced ({cursor:,} patches, "
                f"{rate:.1f} scenes/s, elapsed {elapsed:.1f}s)",
                flush=True,
            )

    pool.shutdown(wait=True)
    return {
        "tokens_fp16": np.concatenate(all_tokens, axis=0),     # [total_patches, 21] fp16
        "targets_fp16": np.concatenate(all_targets, axis=0),   # [total_patches, 12] fp16
        "object_ids": np.concatenate(all_oids, axis=0),        # [total_patches] int16
        "scene_offsets": scene_offsets,                        # [n_scenes + 1] int64
    }


def build_datasets_in_ram(
    *,
    n_train: int = 32000,
    n_heldout: int = 256,
    n_occlusion_pairs: int = 16,
    train_spp: int = 256,
    heldout_spp: int = 1024,
    seed: int = 2026,
    device: str = "cuda:0",
    heldout_bundle_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build train dataset (in RAM) + strictly held-out unseen bundle + occlusion pairs."""
    t0 = time.perf_counter()
    print("[datagen] Building pre-clustered 32-patch ArchetypeLibrary (160 train + 40 unseen props)...", flush=True)
    lib = ArchetypeLibrary(seed=seed)
    print(f"[datagen] Library built in {time.perf_counter() - t0:.1f}s.", flush=True)

    # 1. Generate training scene specs
    rng_train = np.random.default_rng(seed + 1)
    train_specs = [generate_scene_spec(lib, rng_train, split="train")[0] for _ in range(n_train)]
    train_data = trace_scene_specs_batched(
        train_specs,
        rays_per_patch=train_spp,
        max_bounces=3,
        mega_batch_size=64,
        base_seed=10000,
        device=device,
        log_prefix="train_datagen",
    )

    # 2. Generate 256 strictly held-out unseen scenes (1024 spp!)
    rng_val = np.random.default_rng(seed + 99999)
    heldout_specs = []
    heldout_topos = []
    for _ in range(n_heldout):
        s, meta = generate_scene_spec(lib, rng_val, split="heldout")
        heldout_specs.append(s)
        heldout_topos.append(meta["topology"])
    heldout_data = trace_scene_specs_batched(
        heldout_specs,
        rays_per_patch=heldout_spp,
        max_bounces=3,
        mega_batch_size=16,
        base_seed=500000,
        device=device,
        log_prefix="heldout_datagen",
    )
    heldout_data["topologies"] = np.array(heldout_topos)

    # 3. Generate dynamic occlusion pairs (16 pairs = 32 scenes traced at 1024 spp!)
    rng_occ = np.random.default_rng(seed + 777777)
    occ_specs = []
    occ_target_oids = []
    for _ in range(n_occlusion_pairs):
        open_objs, occ_objs, meta = generate_dynamic_occlusion_pair(lib, rng_occ)
        occ_specs.append(open_objs)
        occ_specs.append(occ_objs)
        occ_target_oids.append(meta["target_object_id"])
    occ_data = trace_scene_specs_batched(
        occ_specs,
        rays_per_patch=heldout_spp,
        max_bounces=3,
        mega_batch_size=16,
        base_seed=900000,
        device=device,
        log_prefix="occlusion_datagen",
    )
    occ_data["target_object_ids"] = np.array(occ_target_oids, dtype=np.int32)

    if heldout_bundle_path is not None:
        heldout_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            heldout_bundle_path,
            tokens_fp16=heldout_data["tokens_fp16"],
            targets_fp16=heldout_data["targets_fp16"],
            object_ids=heldout_data["object_ids"],
            scene_offsets=heldout_data["scene_offsets"],
            topologies=heldout_data["topologies"],
            occ_tokens_fp16=occ_data["tokens_fp16"],
            occ_targets_fp16=occ_data["targets_fp16"],
            occ_object_ids=occ_data["object_ids"],
            occ_scene_offsets=occ_data["scene_offsets"],
            occ_target_object_ids=occ_data["target_object_ids"],
        )
        size_mb = heldout_bundle_path.stat().st_size / (1024 * 1024)
        print(f"[datagen] Saved held-out verification bundle to {heldout_bundle_path} ({size_mb:.2f} MB)", flush=True)

    return train_data, heldout_data, occ_data
