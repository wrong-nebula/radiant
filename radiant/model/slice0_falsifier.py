"""Slice 0 SE(3) Falsification Experiment (O2): H1 (Naive XYZ) vs. H2 (RADIANT SE(3) Form-Factor).

Trains both models on the same subset of rooms with identical budget, then evaluates both on:
  1. Aligned held-out unseen rooms.
  2. Randomly rotated (SO(3)) & translated (R^3) held-out unseen rooms.
  3. Direct SE(3) invariance drift test: RMSE(f(T(x)), f(x)) across random rigid transforms.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
import numpy as np
import torch

from radiant.model.radiant_transformer import (
    NaiveXYZTransformer,
    RadiantGITransformer,
    radiant_gi_loss,
)
from radiant.sh_math import (
    color_bleed_correlation,
    sh_directional_cosine,
    sh_rmse,
    tonemapped_psnr_db,
)


def random_so3_matrices(batch_size: int, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniform random proper rotation matrices R in SO(3) [B, 3, 3]."""
    q = torch.randn(batch_size, 4, device=device, generator=generator)
    q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )
    return R


def apply_se3_transform(
    tokens: torch.Tensor,  # [B, N, 21]
    R: torch.Tensor,       # [B, 3, 3]
    t: torch.Tensor,       # [B, 3]
) -> torch.Tensor:
    """Apply global rigid transform (R, t) to pos, T, B, N while leaving intrinsic properties untouched."""
    out = tokens.clone()
    Rt = R.transpose(-1, -2)
    out[..., 0:3] = torch.bmm(tokens[..., 0:3].float(), Rt) + t.unsqueeze(1)
    out[..., 3:6] = torch.bmm(tokens[..., 3:6].float(), Rt)
    out[..., 6:9] = torch.bmm(tokens[..., 6:9].float(), Rt)
    out[..., 9:12] = torch.bmm(tokens[..., 9:12].float(), Rt)
    return out.to(tokens.dtype)


def apply_tangent_twist_augmentation(
    tokens: torch.Tensor,  # [B, N, 21]
    targets: torch.Tensor, # [B, N, 12]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly rotate each patch's local tangent pair (T, B) around N and rotate target (I_T, I_B) identically.

    Teaches exact in-plane tangent gauge equivariance!
    """
    B_sz, N_tok, _ = tokens.shape
    theta = torch.rand(B_sz, N_tok, 1, device=tokens.device, dtype=torch.float32) * (2.0 * math.pi)
    c = torch.cos(theta)
    s = torch.sin(theta)

    T = tokens[..., 3:6].float()
    Bv = tokens[..., 6:9].float()
    T_new = c * T + s * Bv
    B_new = -s * T + c * Bv

    tok_out = tokens.clone()
    tok_out[..., 3:6] = T_new.to(tokens.dtype)
    tok_out[..., 6:9] = B_new.to(tokens.dtype)

    it = targets[..., 3:6].float()
    ib = targets[..., 6:9].float()
    it_new = c * it + s * ib
    ib_new = -s * it + c * ib

    tgt_out = targets.clone()
    tgt_out[..., 3:6] = it_new.to(targets.dtype)
    tgt_out[..., 6:9] = ib_new.to(targets.dtype)
    return tok_out, tgt_out


def collate_csr_scenes(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice variable-length scenes from CSR RAM arrays into padded GPU tensors."""
    offsets = dataset["scene_offsets"]
    tok_all = dataset["tokens_fp16"]
    tgt_all = dataset["targets_fp16"]

    lengths = [int(offsets[i + 1] - offsets[i]) for i in indices]
    max_len = max(lengths)
    B = len(indices)

    tok_batch = np.zeros((B, max_len, 21), dtype=np.float32)
    tgt_batch = np.zeros((B, max_len, 12), dtype=np.float32)
    mask_batch = np.zeros((B, max_len), dtype=bool)

    for b, idx in enumerate(indices):
        s, e = int(offsets[idx]), int(offsets[idx + 1])
        L = e - s
        tok_batch[b, :L] = tok_all[s:e]
        tgt_batch[b, :L] = tgt_all[s:e]
        mask_batch[b, :L] = True

    return (
        torch.from_numpy(tok_batch).to(device, non_blocking=True),
        torch.from_numpy(tgt_batch).to(device, non_blocking=True),
        torch.from_numpy(mask_batch).to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate_model_on_heldout(
    model: torch.nn.Module,
    heldout: dict[str, np.ndarray],
    device: torch.device,
    *,
    random_se3: bool = False,
    seed: int = 1234,
    batch_size: int = 16,
    use_bf16: bool = True,
) -> dict[str, float]:
    model.eval()
    n_scenes = len(heldout["scene_offsets"]) - 1
    gen = torch.Generator(device=device).manual_seed(seed)

    preds_list, gts_list = [], []
    for start in range(0, n_scenes, batch_size):
        idx = np.arange(start, min(start + batch_size, n_scenes))
        tok, tgt, mask = collate_csr_scenes(heldout, idx, device)
        if random_se3:
            R = random_so3_matrices(tok.shape[0], device, generator=gen)
            t = (torch.rand(tok.shape[0], 3, device=device, generator=gen) - 0.5) * 40.0
            tok = apply_se3_transform(tok, R, t)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
            pred = model(tok, valid_mask=mask)

        m_np = mask.cpu().numpy()
        p_np = pred.float().cpu().numpy()
        g_np = tgt.float().cpu().numpy()
        preds_list.append(p_np[m_np])
        gts_list.append(g_np[m_np])

    all_p = np.concatenate(preds_list, axis=0)
    all_g = np.concatenate(gts_list, axis=0)
    return {
        "psnr_db": round(tonemapped_psnr_db(all_p[:, 0:3], all_g[:, 0:3]), 4),
        "color_bleed_r": round(color_bleed_correlation(all_p[:, 0:3], all_g[:, 0:3]), 5),
        "sh_dir_cosine": round(sh_directional_cosine(all_p, all_g), 5),
        "sh_rmse": round(sh_rmse(all_p, all_g), 6),
    }


@torch.no_grad()
def measure_se3_invariance_drift(
    model: torch.nn.Module,
    heldout: dict[str, np.ndarray],
    device: torch.device,
    n_scenes: int = 32,
    n_transforms: int = 4,
    seed: int = 4321,
) -> dict[str, float]:
    """Measure exact RMSE change in predicted local SH under arbitrary rigid transforms (R, t)."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    idx = np.arange(min(n_scenes, len(heldout["scene_offsets"]) - 1))
    tok, _, mask = collate_csr_scenes(heldout, idx, device)
    base_pred = model(tok, valid_mask=mask).float()

    rmses = []
    max_abs = 0.0
    for _ in range(n_transforms):
        R = random_so3_matrices(tok.shape[0], device, generator=gen)
        # Large arbitrary translation up to +/- 100 m + full SO(3) rotation!
        t = (torch.rand(tok.shape[0], 3, device=device, generator=gen) - 0.5) * 200.0
        tok_tr = apply_se3_transform(tok, R, t)
        tr_pred = model(tok_tr, valid_mask=mask).float()
        diff = (tr_pred - base_pred)[mask]
        rmses.append(float(torch.sqrt((diff * diff).mean()).item()))
        max_abs = max(max_abs, float(diff.abs().max().item()))

    return {
        "mean_se3_rmse": round(float(np.mean(rmses)), 7),
        "max_se3_rmse": round(float(np.max(rmses)), 7),
        "max_abs_diff": round(max_abs, 7),
    }


def _train_quick_slice0(
    model: torch.nn.Module,
    train_data: dict[str, np.ndarray],
    device: torch.device,
    *,
    n_train_subset: int = 4096,
    steps: int = 600,
    batch_size: int = 24,
    lr: float = 4e-4,
    label: str = "model",
) -> float:
    t0 = time.perf_counter()
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    n_avail = min(n_train_subset, len(train_data["scene_offsets"]) - 1)
    rng = np.random.default_rng(77)

    for step in range(1, steps + 1):
        idx = rng.integers(0, n_avail, size=batch_size)
        tok, tgt, mask = collate_csr_scenes(train_data, idx, device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(tok, valid_mask=mask)
            loss, _ = radiant_gi_loss(pred, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps:
            print(f"[slice0:{label}] step {step}/{steps} loss={loss.item():.4f}", flush=True)
    return time.perf_counter() - t0


def run_slice0_falsification(
    train_data: dict[str, np.ndarray],
    heldout_data: dict[str, np.ndarray],
    device: torch.device,
    out_path: Path | None = None,
    steps: int = 600,
) -> dict:
    """Compare H1 (Naive XYZ) vs H2 (RADIANT SE(3)) on aligned AND rotated/translated unseen rooms."""
    print("[slice0] Running H1 (Naive XYZ) vs H2 (RADIANT SE(3)) falsification...", flush=True)
    torch.manual_seed(2026)
    h1 = NaiveXYZTransformer(d_model=256, n_heads=8, n_layers=6, d_ff=1024)
    t_h1 = _train_quick_slice0(h1, train_data, device, steps=steps, label="H1_naive_xyz")

    torch.manual_seed(2026)
    h2 = RadiantGITransformer(d_model=256, n_heads=8, n_layers=6, d_ff=1024)
    t_h2 = _train_quick_slice0(h2, train_data, device, steps=steps, label="H2_radiant_se3")

    h1_aligned = evaluate_model_on_heldout(h1, heldout_data, device, random_se3=False)
    h1_rotated = evaluate_model_on_heldout(h1, heldout_data, device, random_se3=True)
    h1_drift = measure_se3_invariance_drift(h1, heldout_data, device)

    h2_aligned = evaluate_model_on_heldout(h2, heldout_data, device, random_se3=False)
    h2_rotated = evaluate_model_on_heldout(h2, heldout_data, device, random_se3=True)
    h2_drift = measure_se3_invariance_drift(h2, heldout_data, device)

    passed = (
        h2_drift["max_se3_rmse"] < 0.005
        and h2_rotated["psnr_db"] > h1_rotated["psnr_db"] + 2.0
        and h2_aligned["psnr_db"] >= h1_aligned["psnr_db"]
    )

    report = {
        "gate": "O2_SLICE0_SE3_FALSIFICATION",
        "passed": bool(passed),
        "steps_per_model": steps,
        "H1_naive_xyz": {
            "train_seconds": round(t_h1, 2),
            "aligned_heldout": h1_aligned,
            "rotated_heldout": h1_rotated,
            "se3_invariance_drift": h1_drift,
        },
        "H2_radiant_se3": {
            "train_seconds": round(t_h2, 2),
            "aligned_heldout": h2_aligned,
            "rotated_heldout": h2_rotated,
            "se3_invariance_drift": h2_drift,
        },
        "psnr_gain_on_rotated_db": round(h2_rotated["psnr_db"] - h1_rotated["psnr_db"], 4),
        "psnr_gain_on_aligned_db": round(h2_aligned["psnr_db"] - h1_aligned["psnr_db"], 4),
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2), flush=True)
    return report
