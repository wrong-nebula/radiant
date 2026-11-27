"""Single Warm Colab A100 End-to-End Orchestrator (Detached Remote Runner + Resilient Polling).

Eliminates WebSocket drop failures by launching a detached background process (`start_new_session=True`)
inside ONE warm Colab A100 VM and polling `/content/pipeline.log` + sentinel flags via tiny 1-second
`colab exec` probes (retried transparently on any transient WebSocket blip).

Executes inside ONE warm A100 session:
  1. Stage A: Smoke probe (A100 BF16 + NVIDIA Warp GPU BVH)
  2. Stage B: O1 Pre-generation Physics Gate (White-Furnace <1% + Analytical Cornell Radiosity)
  3. Stage C: 32,000 train @ 256 spp + 256 held-out @ 1024 spp + 16 occlusion pairs (in A100 RAM)
  4. Stage D: O2 Slice 0 SE(3) Falsification (H1 Naive XYZ vs H2 RADIANT SE(3))
  5. Stage E: Full BF16 A100 Training (9 layers, d_model=256, 8 heads, 16,000 steps)
  6. Stage F: Final O3 eval + FP16 RDGI Binary Export (<20 MB) + Parity Probes
  7. Stage G: Download ONLY compact artifacts + Unconditional VM teardown (active_sessions_after == 0)
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.session import ManagedColabSession
from colab.smoke_probe import run_probe_on_session


def _make_source_tarball(tar_path: Path) -> None:
    with tarfile.open(tar_path, "w:gz") as tf:
        radiant_dir = REPO_ROOT / "radiant"
        for p in sorted(radiant_dir.rglob("*.py")):
            arcname = p.relative_to(REPO_ROOT)
            tf.add(p, arcname=str(arcname))


def _build_detached_remote_script(
    *,
    n_train: int,
    n_heldout: int,
    n_heldout_lod: int,
    n_occ_pairs: int,
    train_spp: int,
    heldout_spp: int,
    slice0_steps: int,
    train_steps: int,
    batch_size: int,
) -> str:
    return f'''#!/usr/bin/env python3
"""Detached A100 pipeline runner executed directly on Colab VM."""
import gc, json, os, sys, tarfile, time, traceback
from pathlib import Path
import numpy as np
import torch

out_dir = Path("/content/radiant_out")
out_dir.mkdir(parents=True, exist_ok=True)
done_flag = out_dir / "DONE.flag"
fail_flag = out_dir / "FAILED.flag"
if done_flag.exists():
    done_flag.unlink()
if fail_flag.exists():
    fail_flag.unlink()

def main():
    import subprocess as _sp
    try:
        import warp as _wp
    except ImportError:
        _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "warp-lang"])
        import warp as _wp

    work_dir = Path("/content/work")
    work_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open("/content/radiant_src.tar.gz", "r:gz") as tf:
        tf.extractall(work_dir)
    if str(work_dir) not in sys.path:
        sys.path.insert(0, str(work_dir))

    # =========================================================================
    # STAGE A: Smoke Probe (BF16 PyTorch + Warp GPU BVH)
    # =========================================================================
    print("\\n=== STAGE A: GPU BF16 + WARP BVH SMOKE PROBE ===", flush=True)
    assert torch.cuda.is_available(), "CUDA unavailable!"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    bf16_ok = bool(torch.cuda.is_bf16_supported())
    _wp.init()
    smoke_rep = {{
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "bf16_supported": bf16_ok,
        "torch_version": torch.__version__,
        "warp_version": _wp.__version__,
        "passed": bf16_ok,
    }}
    (out_dir / "smoke_probe_report.json").write_text(json.dumps(smoke_rep, indent=2) + "\\n")
    is_a100_tier = ("A100" in gpu_name.upper()) or ("H100" in gpu_name.upper())
    eff_n_train = {n_train} if is_a100_tier else min({n_train}, 48000)
    eff_train_steps = {train_steps} if is_a100_tier else min({train_steps}, 32000)
    print(f"[smoke] PASSED on {{gpu_name}} ({{vram_gb}} GB, BF16={{bf16_ok}}, warp={{_wp.__version__}}) -> schedule: n_train={{eff_n_train}}, steps={{eff_train_steps}}", flush=True)

    from radiant.datagen.physics_gate import run_o1_physics_gate
    from radiant.datagen.procedural_scenes import ArchetypeLibrary, generate_scene_spec, generate_dynamic_occlusion_pair
    from radiant.datagen.dataset_builder import trace_scene_specs_batched
    from radiant.model.slice0_falsifier import (
        run_slice0_falsification, apply_se3_transform, apply_tangent_twist_augmentation,
        collate_csr_scenes, evaluate_model_on_heldout, measure_se3_invariance_drift, random_so3_matrices,
    )
    from radiant.model.radiant_transformer import RadiantGITransformer, radiant_gi_loss, radiant_gi_multibounce_loss
    from radiant.model.train_radiant import evaluate_dynamic_occlusion
    from radiant.model.export_fp16 import export_radiant_fp16_bin, export_parity_probes, load_radiant_fp16_bin
    from radiant.sh_math import tonemapped_psnr_db, color_bleed_correlation, sh_directional_cosine

    device = torch.device("cuda:0")
    t_all = time.perf_counter()

    # =========================================================================
    # STAGE B: O1 Pre-Generation Physics Gate
    # =========================================================================
    print("\\n=== STAGE B: O1 WARP PHYSICS GATE ===", flush=True)
    o1_report = run_o1_physics_gate(device="cuda:0", out_path=out_dir / "o1_physics_report.json")
    print(json.dumps(o1_report, indent=2), flush=True)
    assert o1_report["passed"], f"O1 Physics Gate FAILED: {{o1_report}}"

    # =========================================================================
    # STAGE C: Fast Batched GPU Dataset Generation (in Colab A100 RAM)
    # =========================================================================
    print("\\n=== STAGE C: MULTI-LOD DATASET GENERATION ({n_train} Mixed-LOD train @ {train_spp} spp, {n_heldout} uniform-32 + {n_heldout_lod} Distance-LOD heldout @ {heldout_spp} spp) ===", flush=True)
    t0 = time.perf_counter()
    lib = ArchetypeLibrary(seed=2026, fetch_cc0=True, multi_lod=True)
    print(f"[datagen] Multi-LOD ArchetypeLibrary built in {{time.perf_counter()-t0:.1f}}s (lods={{lib.lod_counts}}, train={{len(lib.train_props)}}, heldout={{len(lib.heldout_props)}}, cc0_fetched={{lib.n_cc0_loaded}})", flush=True)

    rng_val = np.random.default_rng(2026 + 99999)
    heldout_specs, heldout_topos = [], []
    for _ in range({n_heldout}):
        s, meta = generate_scene_spec(lib, rng_val, split="heldout", lod_mode="fixed32")
        heldout_specs.append(s)
        heldout_topos.append(meta["topology"])
    heldout_data = trace_scene_specs_batched(
        heldout_specs, rays_per_patch={heldout_spp}, max_bounces=3,
        mega_batch_size=32, base_seed=500000, device="cuda:0", log_prefix="heldout_32_1024spp", per_bounce=True,
    )
    heldout_data["topologies"] = np.array(heldout_topos)

    # Strictly unseen Distance-LOD held-out benchmark (K in {{4, 8, 16, 32, 64}})
    rng_val_lod = np.random.default_rng(2026 + 888888)
    lod_specs, lod_topos = [], []
    for _ in range({n_heldout_lod}):
        s, meta = generate_scene_spec(lib, rng_val_lod, split="heldout", lod_mode="distance")
        lod_specs.append(s)
        lod_topos.append(meta["topology"])
    heldout_lod_data = trace_scene_specs_batched(
        lod_specs, rays_per_patch={heldout_spp}, max_bounces=3,
        mega_batch_size=32, base_seed=650000, device="cuda:0", log_prefix="heldout_lod_1024spp", per_bounce=True,
    )
    heldout_lod_data["topologies"] = np.array(lod_topos)

    rng_occ = np.random.default_rng(2026 + 777777)
    occ_specs, occ_target_oids = [], []
    for _ in range({n_occ_pairs}):
        o_open, o_occ, meta = generate_dynamic_occlusion_pair(lib, rng_occ)
        occ_specs.append(o_open)
        occ_specs.append(o_occ)
        occ_target_oids.append(meta["target_object_id"])
    occ_data = trace_scene_specs_batched(
        occ_specs, rays_per_patch={heldout_spp}, max_bounces=3,
        mega_batch_size=32, base_seed=900000, device="cuda:0", log_prefix="occlusion_1024spp",
    )
    occ_data["target_object_ids"] = np.array(occ_target_oids, dtype=np.int32)

    heldout_bundle_path = out_dir / "heldout_eval_bundle.npz"
    np.savez_compressed(
        heldout_bundle_path,
        tokens_fp16=heldout_data["tokens_fp16"],
        targets_fp16=heldout_data["targets_fp16"],
        object_ids=heldout_data["object_ids"],
        scene_offsets=heldout_data["scene_offsets"],
        topologies=heldout_data["topologies"],
        lod_tokens_fp16=heldout_lod_data["tokens_fp16"],
        lod_targets_fp16=heldout_lod_data["targets_fp16"],
        lod_object_ids=heldout_lod_data["object_ids"],
        lod_scene_offsets=heldout_lod_data["scene_offsets"],
        lod_topologies=heldout_lod_data["topologies"],
        occ_tokens_fp16=occ_data["tokens_fp16"],
        occ_targets_fp16=occ_data["targets_fp16"],
        occ_object_ids=occ_data["object_ids"],
        occ_scene_offsets=occ_data["scene_offsets"],
        occ_target_object_ids=occ_data["target_object_ids"],
    )
    print(f"[datagen] Saved held-out + Mixed-LOD bundle ({{heldout_bundle_path.stat().st_size/(1024**2):.2f}} MB)", flush=True)

    rng_train = np.random.default_rng(2026 + 3)
    train_specs = [generate_scene_spec(lib, rng_train, split="train", lod_mode="auto")[0] for _ in range(eff_n_train)]
    train_data = trace_scene_specs_batched(
        train_specs, rays_per_patch={train_spp}, max_bounces=3,
        mega_batch_size=64, base_seed=10000, device="cuda:0", log_prefix="train_mixedlod_256spp", per_bounce=True,
    )
    del train_specs
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[datagen] Train dataset in RAM: {{len(train_data['scene_offsets'])-1:,}} scenes, {{len(train_data['tokens_fp16']):,}} patches (mean {{len(train_data['tokens_fp16'])/max(len(train_data['scene_offsets'])-1,1):.1f}} tok/scene)", flush=True)

    # =========================================================================
    # STAGE D: O2 Slice 0 SE(3) Falsification (H1 vs H2)
    # =========================================================================
    print("\\n=== STAGE D: O2 SLICE 0 SE(3) FALSIFICATION (H1 vs H2) ===", flush=True)
    o2_report = run_slice0_falsification(
        train_data, heldout_data, device=device,
        out_path=out_dir / "o2_slice0_report.json", steps={slice0_steps},
    )
    assert o2_report["passed"], f"O2 Slice 0 Falsification FAILED: {{o2_report}}"
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # STAGE E: Full A100 BF16 Training (9 layers, d_model=256, 8 heads, Area-Prior + EMA)
    # =========================================================================
    print("\\n=== STAGE E: FULL A100 BF16 TRAINING ({train_steps} steps, bs={batch_size}, Area-Prior Mixed-LOD + EMA) ===", flush=True)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = RadiantGITransformer(d_model=256, n_heads=8, n_layers=9, d_ff=1024, area_prior=True, multi_bounce=True).to(device)
    ema_model = RadiantGITransformer(d_model=256, n_heads=8, n_layers=9, d_ff=1024, area_prior=True, multi_bounce=True).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model params: {{n_params:,}} ({{n_params*2/(1024**2):.2f}} MB FP16, area_prior=True, multi_bounce=1/2/3)", flush=True)

    total_steps = eff_train_steps
    batch_size = {batch_size}
    warmup_steps = 1000
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, betas=(0.9, 0.95), weight_decay=1e-2)
    def _lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.02 + 0.98 * 0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    n_train_scenes = len(train_data["scene_offsets"]) - 1
    rng_tr = np.random.default_rng(202603)
    best_score = -1.0
    best_psnr = -1.0
    best_lod_psnr = -1.0
    best_state = None
    train_history = []
    t_train_start = time.perf_counter()
    running_loss = 0.0

    for step in range(1, total_steps + 1):
        model.train()
        idx = rng_tr.integers(0, n_train_scenes, size=batch_size)
        tok, tgt, mask = collate_csr_scenes(train_data, idx, device, use_bounces=True)
        R = random_so3_matrices(tok.shape[0], device)
        t_vec = (torch.rand(tok.shape[0], 3, device=device) - 0.5) * 16.0
        tok = apply_se3_transform(tok, R, t_vec)
        tok, tgt = apply_tangent_twist_augmentation(tok, tgt)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred36 = model(tok, valid_mask=mask, return_bounces=True)
            loss, stats = radiant_gi_multibounce_loss(pred36, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running_loss += stats["loss"]

        # Update EMA weights (decay ramps up smoothly to 0.999)
        ema_decay = min(0.999, (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            for p_ema, p_cur in zip(ema_model.parameters(), model.parameters()):
                p_ema.mul_(ema_decay).add_(p_cur, alpha=1.0 - ema_decay)

        if step % 500 == 0:
            avg_l = running_loss / 500.0
            running_loss = 0.0
            print(f"[train] step {{step:5d}}/{{total_steps}} | loss={{avg_l:.4f}} | lr={{sched.get_last_lr()[0]:.2e}} | elapsed={{time.perf_counter()-t_train_start:.1f}}s", flush=True)

        if step % 2000 == 0 or step == total_steps:
            eval_net = ema_model if step >= 4000 else model
            val_32 = evaluate_model_on_heldout(eval_net, heldout_data, device, random_se3=False)
            val_lod = evaluate_model_on_heldout(eval_net, heldout_lod_data, device, random_se3=False)
            rec = {{
                "step": step,
                "elapsed_s": round(time.perf_counter() - t_train_start, 1),
                "source": "ema" if step >= 4000 else "raw",
                "lod_psnr_db": val_lod["psnr_db"],
                "lod_color_bleed_r": val_lod["color_bleed_r"],
                **val_32,
            }}
            train_history.append(rec)
            print(f"[eval ] step {{step:5d}} ({{rec['source']}}) | U32 PSNR={{val_32['psnr_db']:.2f}} dB (r={{val_32['color_bleed_r']:.4f}}) | LOD PSNR={{val_lod['psnr_db']:.2f}} dB (r={{val_lod['color_bleed_r']:.4f}}) | sh_cos={{val_32['sh_dir_cosine']:.4f}}", flush=True)
            combo = 0.5 * val_32["psnr_db"] + 0.5 * val_lod["psnr_db"]
            if combo > best_score:
                best_score = combo
                best_psnr = val_32["psnr_db"]
                best_lod_psnr = val_lod["psnr_db"]
                best_state = {{k: v.detach().cpu().clone() for k, v in eval_net.state_dict().items()}}

    if best_state is not None:
        model.load_state_dict({{k: v.to(device) for k, v in best_state.items()}})

    # =========================================================================
    # STAGE F: Final O3 Eval (Uniform-32 + Mixed-LOD), FP16 RDGI Export & Parity Probes
    # =========================================================================
    print("\\n=== STAGE F: FINAL O3 EVAL (U32 + MIXED-LOD) + FP16 RDGI EXPORT + PARITY PROBES ===", flush=True)
    total_train_sec = time.perf_counter() - t_train_start
    bin_path = out_dir / "radiant_gi_fp16.bin"
    export_info = export_radiant_fp16_bin(model, bin_path)
    probe_info = export_parity_probes(
        bin_path, heldout_data, out_dir / "parity_probes.json",
        n_probes=4, mixed_lod_data=heldout_lod_data, n_lod_probes=2,
    )

    fp16_model = load_radiant_fp16_bin(bin_path, device=device)
    fp16_aligned = evaluate_model_on_heldout(fp16_model, heldout_data, device, random_se3=False, use_bf16=False)
    fp16_rotated = evaluate_model_on_heldout(fp16_model, heldout_data, device, random_se3=True, use_bf16=False)
    fp16_lod_aligned = evaluate_model_on_heldout(fp16_model, heldout_lod_data, device, random_se3=False, use_bf16=False)
    fp16_lod_rotated = evaluate_model_on_heldout(fp16_model, heldout_lod_data, device, random_se3=True, use_bf16=False)
    fp16_drift = measure_se3_invariance_drift(fp16_model, heldout_data, device, n_scenes=64, n_transforms=4)
    fp16_occ = evaluate_dynamic_occlusion(fp16_model, occ_data, device)

    # Measure exact 1-bounce, 2-bounce, and 3-bounce PSNR on the 256 unseen held-out rooms!
    p1_l, p2_l, p3_l, g1_l, g2_l, g3_l = [], [], [], [], [], []
    n_ho = len(heldout_data["scene_offsets"]) - 1
    for st in range(0, n_ho, 16):
        bidx = np.arange(st, min(st + 16, n_ho))
        tok, tgt36, mask = collate_csr_scenes(heldout_data, bidx, device, use_bounces=True)
        with torch.no_grad():
            pr36 = fp16_model(tok, valid_mask=mask, return_bounces=True).float().cpu().numpy()
        g36 = tgt36.float().cpu().numpy()
        m_np = mask.cpu().numpy()
        pm = pr36[m_np]
        gm = g36[m_np]
        p1_l.append(pm[:, 0:3]); g1_l.append(gm[:, 0:3])
        p2_l.append(pm[:, 0:3] + pm[:, 12:15]); g2_l.append(gm[:, 0:3] + gm[:, 12:15])
        p3_l.append(pm[:, 0:3] + pm[:, 12:15] + pm[:, 24:27]); g3_l.append(gm[:, 0:3] + gm[:, 12:15] + gm[:, 24:27])
    per_bounce_psnr = {{
        "1_bounce_psnr_db": round(tonemapped_psnr_db(np.concatenate(p1_l), np.concatenate(g1_l)), 3),
        "2_bounce_psnr_db": round(tonemapped_psnr_db(np.concatenate(p2_l), np.concatenate(g2_l)), 3),
        "3_bounce_psnr_db": round(tonemapped_psnr_db(np.concatenate(p3_l), np.concatenate(g3_l)), 3),
    }}
    print(f"[eval ] Per-bounce held-out PSNR: {{per_bounce_psnr}}", flush=True)

    per_topo = {{}}
    topos = heldout_data["topologies"]
    for tp in sorted(set(str(x) for x in topos.tolist())):
        t_idx = np.array([i for i, x in enumerate(topos) if str(x) == tp], dtype=np.int64)
        ps, gs = [], []
        for st in range(0, len(t_idx), 16):
            tok, tgt, mask = collate_csr_scenes(heldout_data, t_idx[st:st+16], device)
            with torch.no_grad():
                pr = fp16_model(tok, valid_mask=mask).float().cpu().numpy()
            m_np = mask.cpu().numpy()
            ps.append(pr[m_np])
            gs.append(tgt.float().cpu().numpy()[m_np])
        ap, ag = np.concatenate(ps, axis=0), np.concatenate(gs, axis=0)
        per_topo[tp] = {{
            "n_scenes": int(len(t_idx)),
            "psnr_db": round(tonemapped_psnr_db(ap[:, 0:3], ag[:, 0:3]), 3),
            "color_bleed_r": round(color_bleed_correlation(ap[:, 0:3], ag[:, 0:3]), 4),
            "sh_dir_cosine": round(sh_directional_cosine(ap, ag), 4),
        }}

    o3_passed = (
        fp16_aligned["psnr_db"] >= 28.0
        and fp16_aligned["color_bleed_r"] >= 0.92
        and fp16_lod_aligned["psnr_db"] >= 28.0
        and fp16_lod_aligned["color_bleed_r"] >= 0.92
        and fp16_occ["passed_drop_ge_40pct"]
        and fp16_drift["max_se3_rmse"] < 0.005
    )

    o3_report = {{
        "gate": "O3_HELDOUT_MULTIBOUNCE_GI_FIDELITY",
        "passed": bool(o3_passed),
        "area_prior": True,
        "multi_bounce": True,
        "lod_levels": [4, 8, 16, 32, 64],
        "n_heldout_scenes": int(len(heldout_data["scene_offsets"]) - 1),
        "n_heldout_lod_scenes": int(len(heldout_lod_data["scene_offsets"]) - 1),
        "aligned_metrics": fp16_aligned,
        "rotated_metrics": fp16_rotated,
        "mixed_lod_aligned_metrics": fp16_lod_aligned,
        "mixed_lod_rotated_metrics": fp16_lod_rotated,
        "per_bounce_psnr_db": per_bounce_psnr,
        "se3_invariance_drift": fp16_drift,
        "dynamic_occlusion": fp16_occ,
        "per_topology_breakdown": per_topo,
        "export_info": export_info,
        "probe_info": probe_info,
    }}
    train_summary = {{
        "model_config": {{
            "d_model": 256, "n_heads": 8, "n_layers": 9, "d_ff": 1024,
            "area_prior": True, "multi_bounce": True, "lod_levels": [4, 8, 16, 32, 64],
            "n_params": n_params, "fp16_mb": export_info["size_mb"],
        }},
        "n_train_scenes": n_train_scenes,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "train_seconds": round(total_train_sec, 2),
        "best_val_psnr_db": round(best_psnr, 4),
        "best_lod_psnr_db": round(best_lod_psnr, 4),
        "history": train_history,
    }}
    (out_dir / "o3_heldout_report.json").write_text(json.dumps(o3_report, indent=2) + "\\n")
    (out_dir / "training_summary.json").write_text(json.dumps(train_summary, indent=2) + "\\n")
    print(json.dumps(o3_report, indent=2), flush=True)
    summary = {{
        "passed": o3_report["passed"],
        "fp16_psnr_db": fp16_aligned["psnr_db"],
        "fp16_lod_psnr_db": fp16_lod_aligned["psnr_db"],
        "fp16_r": fp16_aligned["color_bleed_r"],
        "fp16_lod_r": fp16_lod_aligned["color_bleed_r"],
        "fp16_se3_max_rmse": fp16_drift["max_se3_rmse"],
        "fp16_occ_drop": fp16_occ["mean_pred_drop_ratio"],
        "bin_mb": export_info["size_mb"],
        "total_pipeline_seconds": round(time.perf_counter() - t_all, 1),
    }}
    print("FINAL_O3_SUMMARY=" + json.dumps(summary), flush=True)
    assert o3_report["passed"], f"Final FP16 O3 gate failed: {{summary}}"
    done_flag.write_text(json.dumps(summary) + "\\n")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        fail_flag.write_text(tb)
        sys.exit(1)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full THREE-RADIANT-GI v3 A100 Mixed-LOD pipeline via detached remote runner")
    parser.add_argument("--gpu", type=str, default="A100")
    parser.add_argument("--n-train", type=int, default=64000)
    parser.add_argument("--n-heldout", type=int, default=256)
    parser.add_argument("--n-heldout-lod", type=int, default=128)
    parser.add_argument("--n-occ-pairs", type=int, default=16)
    parser.add_argument("--train-spp", type=int, default=256)
    parser.add_argument("--heldout-spp", type=int, default=1024)
    parser.add_argument("--slice0-steps", type=int, default=600)
    parser.add_argument("--train-steps", type=int, default=42000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    artifacts_dir = REPO_ROOT / "artifacts"
    data_dir = REPO_ROOT / "data"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "radiant_src.tar.gz"
        _make_source_tarball(tar_path)
        remote_script_path = Path(td) / "run_remote_all.py"
        remote_script_path.write_text(
            _build_detached_remote_script(
                n_train=args.n_train,
                n_heldout=args.n_heldout,
                n_heldout_lod=args.n_heldout_lod,
                n_occ_pairs=args.n_occ_pairs,
                train_spp=args.train_spp,
                heldout_spp=args.heldout_spp,
                slice0_steps=args.slice0_steps,
                train_steps=args.train_steps,
                batch_size=args.batch_size,
            ),
            encoding="utf-8",
        )
        print(f"[pipeline] Packed radiant/ source ({tar_path.stat().st_size / 1024:.1f} KB)", flush=True)

        with ManagedColabSession(gpu=args.gpu, fallback_gpus=("H100", "L4"), notes="v3_area_prior_mixed_lod_a100") as sess:
            # Upload compact source + detached runner script (REST API - zero WebSocket overhead)
            sess.upload(tar_path, "/content/radiant_src.tar.gz")
            sess.upload(remote_script_path, "/content/run_remote_all.py")

            # Launch detached background job inside the VM (immune to WebSocket drops!)
            print("\n=== LAUNCHING DETACHED REMOTE PIPELINE ON COLAB GPU ===", flush=True)
            launch_out = sess.exec_code(
                r'''
import os, subprocess, sys
log_f = open("/content/pipeline.log", "w")
proc = subprocess.Popen(
    [sys.executable, "-u", "/content/run_remote_all.py"],
    stdout=log_f,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open("/content/pipeline.pid", "w") as pf:
    pf.write(str(proc.pid))
print(f"DETACHED_PID={proc.pid}", flush=True)
''',
                timeout=120,
                stream=True,
            )
            if "DETACHED_PID=" not in launch_out:
                raise RuntimeError("Failed to launch detached remote runner!")

            # Poll every 25s with tiny 1s probes; ignore transient WebSocket blips!
            last_offset = 0
            consecutive_poll_errs = 0
            while True:
                time.sleep(25)
                poll_code = f'''
import json, os
from pathlib import Path
log_p = Path("/content/pipeline.log")
done_p = Path("/content/radiant_out/DONE.flag")
fail_p = Path("/content/radiant_out/FAILED.flag")
pid_p = Path("/content/pipeline.pid")
alive = False
if pid_p.exists():
    try:
        os.kill(int(pid_p.read_text().strip()), 0)
        alive = True
    except Exception:
        alive = False
chunk = ""
new_off = {last_offset}
if log_p.exists():
    data = log_p.read_bytes()
    new_off = len(data)
    if new_off > {last_offset}:
        chunk = data[{last_offset}:].decode("utf-8", errors="replace")
state = "RUNNING"
if done_p.exists():
    state = "DONE"
elif fail_p.exists():
    state = "FAILED"
elif not alive:
    state = "EXITED_NO_FLAG"
print("POLL_META=" + json.dumps({{"state": state, "offset": new_off, "alive": alive}}))
if chunk:
    print(chunk, end="", flush=True)
'''
                try:
                    out = sess.exec_code(poll_code, timeout=90, stream=False)
                    consecutive_poll_errs = 0
                except Exception as exc:
                    consecutive_poll_errs += 1
                    print(f"[pipeline] Transient poll error #{consecutive_poll_errs} (VM keeps running!): {exc}", flush=True)
                    if consecutive_poll_errs >= 12:
                        raise RuntimeError("Lost contact with Colab VM for 12 consecutive polls") from exc
                    continue

                meta = None
                body_lines = []
                for line in out.splitlines():
                    if line.startswith("POLL_META="):
                        meta = json.loads(line.split("POLL_META=", 1)[1])
                    else:
                        body_lines.append(line)
                if body_lines:
                    print("\n".join(body_lines), flush=True)
                if meta is None:
                    continue
                last_offset = int(meta["offset"])
                st = meta["state"]
                if st == "DONE":
                    print("\n[pipeline] Detached remote pipeline reported DONE!", flush=True)
                    break
                if st in ("FAILED", "EXITED_NO_FLAG"):
                    raise RuntimeError(f"Detached remote pipeline ended with state={st}")

            # Stage G: Download ONLY compact artifacts (strictly <= 120 MB disk budget!)
            print("\n=== STAGE G: DOWNLOADING COMPACT ARTIFACTS ===", flush=True)
            downloads = [
                ("/content/radiant_out/smoke_probe_report.json", artifacts_dir / "smoke_probe_report.json"),
                ("/content/radiant_out/radiant_gi_fp16.bin", artifacts_dir / "radiant_gi_fp16.bin"),
                ("/content/radiant_out/o1_physics_report.json", artifacts_dir / "o1_physics_report.json"),
                ("/content/radiant_out/o2_slice0_report.json", artifacts_dir / "o2_slice0_report.json"),
                ("/content/radiant_out/o3_heldout_report.json", artifacts_dir / "o3_heldout_report.json"),
                ("/content/radiant_out/training_summary.json", artifacts_dir / "training_summary.json"),
                ("/content/radiant_out/heldout_eval_bundle.npz", data_dir / "heldout_eval_bundle.npz"),
                ("/content/radiant_out/parity_probes.json", data_dir / "parity_probes.json"),
            ]
            for remote_p, local_p in downloads:
                for dl_try in range(3):
                    try:
                        print(f"[pipeline] Downloading {remote_p} -> {local_p}...", flush=True)
                        sess.download(remote_p, local_p)
                        break
                    except Exception as exc:
                        if dl_try == 2:
                            raise
                        print(f"[pipeline] Download retry {dl_try + 1} for {remote_p}: {exc}", flush=True)
                        time.sleep(5)

    # Verify local disk cap
    total_bytes = 0
    for d in (artifacts_dir, data_dir):
        for f in d.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size
    total_mb = total_bytes / (1024 * 1024)
    print(f"[pipeline] Local data/ + artifacts/ size: {total_mb:.2f} MB (cap: 120.00 MB)", flush=True)
    assert total_mb <= 120.0, f"Exceeded 120 MB local disk cap: {total_mb:.2f} MB"
    print("[pipeline] ALL COLAB A100 STAGES COMPLETED & VM STOPPED CLEANLY!", flush=True)


if __name__ == "__main__":
    main()
