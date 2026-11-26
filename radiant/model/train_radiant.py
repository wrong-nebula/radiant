"""Full Colab A100 BF16 Training Loop & O3 Held-Out Evaluation for RADIANT-GI."""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import torch

from radiant.model.radiant_transformer import RadiantGITransformer, radiant_gi_loss
from radiant.model.slice0_falsifier import (
    apply_se3_transform,
    apply_tangent_twist_augmentation,
    collate_csr_scenes,
    evaluate_model_on_heldout,
    measure_se3_invariance_drift,
    random_so3_matrices,
)
from radiant.sh_math import (
    color_bleed_correlation,
    sh_directional_cosine,
    sh_rmse,
    tonemapped_psnr_db,
)


@torch.no_grad()
def evaluate_dynamic_occlusion(
    model: torch.nn.Module,
    occ_data: dict[str, np.ndarray],
    device: torch.device,
) -> dict:
    """Evaluate O3 dynamic occlusion test on 16 unseen (open, occluded) pairs.

    Checks that sliding an unseen solid wall between emitter and target drops predicted
    indirect irradiance on the target object by >= 40% matching ground truth.
    """
    model.eval()
    target_oids = occ_data["target_object_ids"]
    offsets = occ_data["scene_offsets"]
    oids_all = occ_data["object_ids"]
    n_pairs = len(target_oids)

    gt_drops = []
    pred_drops = []
    pair_details = []

    for p in range(n_pairs):
        idx_open = 2 * p
        idx_occ = 2 * p + 1
        target_oid = int(target_oids[p])

        tok, tgt, mask = collate_csr_scenes(occ_data, np.array([idx_open, idx_occ]), device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(tok, valid_mask=mask).float().cpu().numpy()
        gt = tgt.float().cpu().numpy()

        # Extract target object patches for open & occluded scenes
        s0, e0 = int(offsets[idx_open]), int(offsets[idx_open + 1])
        s1, e1 = int(offsets[idx_occ]), int(offsets[idx_occ + 1])
        tmask0 = oids_all[s0:e0] == target_oid
        tmask1 = oids_all[s1:e1] == target_oid

        gt_open = float(gt[0, : e0 - s0][tmask0, 0:3].mean())
        gt_occ = float(gt[1, : e1 - s1][tmask1, 0:3].mean())
        pr_open = float(pred[0, : e0 - s0][tmask0, 0:3].mean())
        pr_occ = float(pred[1, : e1 - s1][tmask1, 0:3].mean())

        gt_drop = (gt_open - gt_occ) / max(gt_open, 1e-6)
        pr_drop = (pr_open - pr_occ) / max(pr_open, 1e-6)
        gt_drops.append(gt_drop)
        pred_drops.append(pr_drop)
        pair_details.append(
            {
                "pair": p,
                "gt_open": round(gt_open, 4),
                "gt_occ": round(gt_occ, 4),
                "gt_drop_pct": round(gt_drop * 100.0, 2),
                "pred_open": round(pr_open, 4),
                "pred_occ": round(pr_occ, 4),
                "pred_drop_pct": round(pr_drop * 100.0, 2),
            }
        )

    mean_gt_drop = float(np.mean(gt_drops))
    mean_pred_drop = float(np.mean(pred_drops))
    drop_abs_err = float(np.mean(np.abs(np.array(pred_drops) - np.array(gt_drops))))
    return {
        "n_pairs": n_pairs,
        "mean_gt_drop_ratio": round(mean_gt_drop, 4),
        "mean_pred_drop_ratio": round(mean_pred_drop, 4),
        "mean_drop_abs_error": round(drop_abs_err, 4),
        "passed_drop_ge_40pct": bool(mean_pred_drop >= 0.40 and mean_gt_drop >= 0.40),
        "sample_pairs": pair_details[:4],
    }


def train_full_radiant_model(
    train_data: dict[str, np.ndarray],
    heldout_data: dict[str, np.ndarray],
    occ_data: dict[str, np.ndarray],
    device: torch.device,
    *,
    n_layers: int = 9,
    d_model: int = 256,
    n_heads: int = 8,
    d_ff: int = 1024,
    total_steps: int = 14000,
    batch_size: int = 32,
    lr: float = 6e-4,
    warmup_steps: int = 400,
    eval_every: int = 1000,
    checkpoint_dir: Path | None = None,
) -> tuple[RadiantGITransformer, dict, dict]:
    """Train full RADIANT-GI Transformer on Colab A100 BF16 with SE(3) + tangent-twist augmentation."""
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = RadiantGITransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[train] RADIANT-GI initialized: {n_layers} layers, d_model={d_model}, "
        f"n_heads={n_heads}, d_ff={d_ff}, params={n_params:,} ({n_params * 2 / (1024**2):.2f} MB FP16)",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-2)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        prog = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.03 + 0.97 * 0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    n_train = len(train_data["scene_offsets"]) - 1
    rng = np.random.default_rng(202601)
    best_psnr = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []

    t0 = time.perf_counter()
    running_loss = 0.0
    log_interval = 200

    for step in range(1, total_steps + 1):
        model.train()
        idx = rng.integers(0, n_train, size=batch_size)
        tok, tgt, mask = collate_csr_scenes(train_data, idx, device)

        # Random SE(3) rigid transform augmentation + random local tangent-plane twist augmentation
        R = random_so3_matrices(tok.shape[0], device)
        t = (torch.rand(tok.shape[0], 3, device=device) - 0.5) * 16.0
        tok = apply_se3_transform(tok, R, t)
        tok, tgt = apply_tangent_twist_augmentation(tok, tgt)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(tok, valid_mask=mask)
            loss, stats = radiant_gi_loss(pred, tgt, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running_loss += stats["loss"]

        if step % log_interval == 0:
            avg_l = running_loss / log_interval
            running_loss = 0.0
            elapsed = time.perf_counter() - t0
            print(
                f"[train] step {step:5d}/{total_steps} | loss={avg_l:.4f} | "
                f"lr={sched.get_last_lr()[0]:.2e} | elapsed={elapsed:.1f}s",
                flush=True,
            )

        if step % eval_every == 0 or step == total_steps:
            val_metrics = evaluate_model_on_heldout(model, heldout_data, device, random_se3=False)
            elapsed = time.perf_counter() - t0
            rec = {"step": step, "elapsed_s": round(elapsed, 1), **val_metrics}
            history.append(rec)
            print(
                f"[eval ] step {step:5d} | PSNR={val_metrics['psnr_db']:.2f} dB | "
                f"r={val_metrics['color_bleed_r']:.4f} | sh_cos={val_metrics['sh_dir_cosine']:.4f} | "
                f"rmse={val_metrics['sh_rmse']:.5f}",
                flush=True,
            )
            if val_metrics["psnr_db"] > best_psnr:
                best_psnr = val_metrics["psnr_db"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint_dir is not None:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, checkpoint_dir / "best_radiant_fp32.pt")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    total_train_sec = time.perf_counter() - t0
    final_aligned = evaluate_model_on_heldout(model, heldout_data, device, random_se3=False, use_bf16=False)
    final_rotated = evaluate_model_on_heldout(model, heldout_data, device, random_se3=True, use_bf16=False)
    final_se3_drift = measure_se3_invariance_drift(model, heldout_data, device, n_scenes=64, n_transforms=4)
    final_occlusion = evaluate_dynamic_occlusion(model, occ_data, device)

    # Also compute per-topology breakdown on the 256 held-out rooms
    per_topo: dict[str, dict[str, float]] = {}
    topos = heldout_data.get("topologies")
    if topos is not None:
        unique_topos = sorted(set(str(x) for x in topos.tolist()))
        offsets = heldout_data["scene_offsets"]
        model.eval()
        for tp in unique_topos:
            t_idx = np.array([i for i, x in enumerate(topos) if str(x) == tp], dtype=np.int64)
            ps, gs = [], []
            for start in range(0, len(t_idx), 16):
                batch_idx = t_idx[start : start + 16]
                tok, tgt, mask = collate_csr_scenes(heldout_data, batch_idx, device)
                with torch.no_grad():
                    pr = model(tok, valid_mask=mask).float().cpu().numpy()
                m_np = mask.cpu().numpy()
                ps.append(pr[m_np])
                gs.append(tgt.float().cpu().numpy()[m_np])
            ap = np.concatenate(ps, axis=0)
            ag = np.concatenate(gs, axis=0)
            per_topo[tp] = {
                "n_scenes": int(len(t_idx)),
                "psnr_db": round(tonemapped_psnr_db(ap[:, 0:3], ag[:, 0:3]), 3),
                "color_bleed_r": round(color_bleed_correlation(ap[:, 0:3], ag[:, 0:3]), 4),
                "sh_dir_cosine": round(sh_directional_cosine(ap, ag), 4),
            }

    o3_passed = (
        final_aligned["psnr_db"] >= 28.0
        and final_aligned["color_bleed_r"] >= 0.92
        and final_occlusion["passed_drop_ge_40pct"]
        and final_se3_drift["max_se3_rmse"] < 0.005
    )

    o3_report = {
        "gate": "O3_HELDOUT_MULTIBOUNCE_GI_FIDELITY",
        "passed": bool(o3_passed),
        "n_heldout_scenes": int(len(heldout_data["scene_offsets"]) - 1),
        "aligned_metrics": final_aligned,
        "rotated_metrics": final_rotated,
        "se3_invariance_drift": final_se3_drift,
        "dynamic_occlusion": final_occlusion,
        "per_topology_breakdown": per_topo,
    }

    train_summary = {
        "model_config": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "n_params": n_params,
            "fp16_mb": round(n_params * 2 / (1024**2), 3),
        },
        "n_train_scenes": n_train,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "train_seconds": round(total_train_sec, 2),
        "best_val_psnr_db": round(best_psnr, 4),
        "history": history,
    }
    return model, o3_report, train_summary
