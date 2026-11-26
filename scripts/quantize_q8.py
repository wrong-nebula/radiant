"""Quantize RADIANT-GI FP16 (`RDGI`) to fused-QKV per-row symmetric INT8 (`RDG8`, ~6.7 MB)
and verify exact held-out O3 PSNR (256 unseen rooms @ 1024 spp) + O4 probe RMSE vs FP16.

Binary Format (`RDG8` v2):
  Header (64 bytes, little-endian):
    [0..4]   magic       = b'RDG8'
    [4..8]   version     = uint32(2)
    [8..12]  d_model     = uint32(256)
    [12..16] n_heads     = uint32(8)
    [16..20] n_layers    = uint32(9)
    [20..24] d_ff        = uint32(1024)
    [24..28] q8_u32_len  = uint32 (number of u32 words = q8_bytes / 4)
    [28..32] f32_len     = uint32 (number of float32 words in aux buffer)
    [32..36] pos_scale   = float32(4.0)
    [36..64] reserved    = zeros (28 bytes)

  Payload:
    1. `q8_u32_len * 4` bytes: packed signed int8 (`int8` in [-127, 127], 4 bytes per u32
       matching WGSL `unpack4x8snorm` which divides by 127.0!) for all large [out, in]
       matrices (in_dim % 4 == 0), with QKV+Geo FUSED into [624, 256] per layer:
         - stem2.weight [256, 256]
         - For layer l = 0..n_layers-1:
             qkv_geo.weight [624, 256]  (concat qSem[152], kSem[152], vInv[208], geoQ[56], geoK[56])
             w_o.weight     [256, 256]
             fc1.weight     [1024, 256]
             fc2.weight     [256, 1024]
         - head1.weight [256, 256]
    2. `f32_len * 4` bytes: exact FP32 auxiliary buffer (scales, biases, RMSNorm weights,
       and tiny sensitive stem1 [256, 13] + head2 [12, 256] weights):
         - stem1.weight [256*13], stem1.bias [256]
         - stem2.scale  [256],    stem2.bias [256]
         - For layer l = 0..n_layers-1:
             norm1.weight  [256]
             qkv_geo.scale [624], qkv_geo.bias [624]
             w_o.scale     [256], w_o.bias     [256]
             norm2.weight  [256]
             fc1.scale     [1024], fc1.bias    [1024]
             fc2.scale     [256],  fc2.bias    [256]
         - final_norm.weight [256]
         - head1.scale [256], head1.bias [256]
         - head2.weight [12*256], head2.bias [12]
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radiant.model.export_fp16 import load_radiant_fp16_bin
from radiant.model.slice0_falsifier import evaluate_model_on_heldout
from radiant.model.train_radiant import evaluate_dynamic_occlusion
from radiant.sh_math import POS_SCALE_METERS, sh_rmse, tonemapped_psnr_db


def quantize_rowwise_snorm8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize [out, in] float32 matrix to signed int8 in [-127, 127] matching WGSL unpack4x8snorm."""
    assert w.ndim == 2 and (w.shape[1] % 4 == 0), f"Bad shape for Q8: {w.shape}"
    scales = np.max(np.abs(w), axis=1).astype(np.float32)
    scales_safe = np.where(scales > 1e-8, scales, 1.0).astype(np.float32)
    q = np.clip(np.rint((w / scales_safe[:, None]) * 127.0), -127, 127).astype(np.int8)
    deq = (q.astype(np.float32) / 127.0) * scales[:, None]
    return q, scales, deq


def export_rdg8(fp16_bin: Path, out_bin: Path) -> tuple[dict, torch.nn.Module]:
    model = load_radiant_fp16_bin(fp16_bin, device="cpu")
    sd = {k: v.detach().cpu().float().numpy().copy() for k, v in model.state_dict().items()}
    deq_sd: dict[str, torch.Tensor] = {k: torch.from_numpy(v.copy()) for k, v in sd.items()}

    q8_chunks: list[np.ndarray] = []
    f32_chunks: list[np.ndarray] = []

    def push_f32(arr: np.ndarray) -> None:
        f32_chunks.append(np.ascontiguousarray(arr.reshape(-1), dtype="<f4"))

    def push_q8(w: np.ndarray, b: np.ndarray) -> np.ndarray:
        q, sc, deq = quantize_rowwise_snorm8(w)
        q8_chunks.append(np.ascontiguousarray(q.reshape(-1), dtype=np.int8))
        push_f32(sc)
        push_f32(b)
        return deq

    # Stem
    push_f32(sd["stem1.weight"])
    push_f32(sd["stem1.bias"])
    deq_sd["stem2.weight"] = torch.from_numpy(push_q8(sd["stem2.weight"], sd["stem2.bias"]))

    # Layers (with fused QKV+Geo)
    for l in range(model.n_layers):
        p = f"blocks.{l}."
        push_f32(sd[f"{p}norm1.weight"])

        qkv_names = ["attn.w_q_sem", "attn.w_k_sem", "attn.w_v_inv", "attn.w_geo_q", "attn.w_geo_k"]
        w_fused = np.concatenate([sd[f"{p}{n}.weight"] for n in qkv_names], axis=0)
        b_fused = np.concatenate([sd[f"{p}{n}.bias"] for n in qkv_names], axis=0)
        deq_fused = push_q8(w_fused, b_fused)
        off = 0
        for n in qkv_names:
            rows = sd[f"{p}{n}.weight"].shape[0]
            deq_sd[f"{p}{n}.weight"] = torch.from_numpy(deq_fused[off : off + rows].copy())
            off += rows

        deq_sd[f"{p}attn.w_o.weight"] = torch.from_numpy(push_q8(sd[f"{p}attn.w_o.weight"], sd[f"{p}attn.w_o.bias"]))
        push_f32(sd[f"{p}norm2.weight"])
        deq_sd[f"{p}fc1.weight"] = torch.from_numpy(push_q8(sd[f"{p}fc1.weight"], sd[f"{p}fc1.bias"]))
        deq_sd[f"{p}fc2.weight"] = torch.from_numpy(push_q8(sd[f"{p}fc2.weight"], sd[f"{p}fc2.bias"]))

    # Final norm + Head
    push_f32(sd["final_norm.weight"])
    deq_sd["head1.weight"] = torch.from_numpy(push_q8(sd["head1.weight"], sd["head1.bias"]))
    push_f32(sd["head2.weight"])
    push_f32(sd["head2.bias"])

    q8_bytes = np.concatenate(q8_chunks).tobytes()
    f32_bytes = np.concatenate(f32_chunks).tobytes()
    assert len(q8_bytes) % 4 == 0
    q8_u32_len = len(q8_bytes) // 4
    f32_len = len(f32_bytes) // 4

    header = struct.pack(
        "<4s7If28s",
        b"RDG8",
        2,
        model.d_model,
        model.n_heads,
        model.n_layers,
        model.d_ff,
        q8_u32_len,
        f32_len,
        float(POS_SCALE_METERS),
        b"\x00" * 28,
    )
    assert len(header) == 64

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with out_bin.open("wb") as f:
        f.write(header)
        f.write(q8_bytes)
        f.write(f32_bytes)

    model.load_state_dict(deq_sd)
    model.eval()
    size_bytes = out_bin.stat().st_size
    info = {
        "path": str(out_bin),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "q8_u32_len": q8_u32_len,
        "f32_len": f32_len,
    }
    return info, model


@torch.no_grad()
def main() -> None:
    fp16_bin = REPO_ROOT / "artifacts" / "radiant_gi_fp16.bin"
    q8_bin = REPO_ROOT / "artifacts" / "radiant_gi_q8.bin"

    info, q8_model = export_rdg8(fp16_bin, q8_bin)
    print(f"[quantize_q8] Exported {q8_bin.name}: {info['size_mb']:.3f} MB ({info['size_bytes']} bytes)", flush=True)

    # Check exact O4 probe diff vs FP16 reference
    probes_data = json.loads((REPO_ROOT / "data" / "parity_probes.json").read_text())
    worst_vs_fp16 = 0.0
    for p in probes_data["probes"]:
        tok = torch.tensor(p["tokens"], dtype=torch.float32).unsqueeze(0)
        ref = np.array(p["pytorch_fp16_ref_sh"], dtype=np.float32)
        gt = np.array(p["gt_sh"], dtype=np.float32)
        pred = q8_model(tok).squeeze(0).numpy().astype(np.float32)
        rmse_fp16 = sh_rmse(pred, ref)
        psnr_gt = tonemapped_psnr_db(pred[:, 0:3], gt[:, 0:3])
        worst_vs_fp16 = max(worst_vs_fp16, rmse_fp16)
        print(
            f"  probe {p['scene_index']} ({p['n_tokens']} tok): "
            f"Q8 vs FP16 RMSE={rmse_fp16:.6f} | Q8 PSNR={psnr_gt:.3f} dB (FP16={p['psnr_vs_gt_db']:.3f} dB)",
            flush=True,
        )

    # Full 256 unseen held-out rooms @ 1024 spp + 16 occlusion pairs!
    npz = np.load(REPO_ROOT / "data" / "heldout_eval_bundle.npz", allow_pickle=False)
    heldout = {k: npz[k] for k in ("tokens_fp16", "targets_fp16", "object_ids", "scene_offsets")}
    occ = {
        "tokens_fp16": npz["occ_tokens_fp16"],
        "targets_fp16": npz["occ_targets_fp16"],
        "object_ids": npz["occ_object_ids"],
        "scene_offsets": npz["occ_scene_offsets"],
        "target_object_ids": npz["occ_target_object_ids"],
    }
    res_aligned = evaluate_model_on_heldout(q8_model, heldout, torch.device("cpu"), random_se3=False, batch_size=16, use_bf16=False)
    res_occ = evaluate_dynamic_occlusion(q8_model, occ, torch.device("cpu"))
    print(f"[quantize_q8] O3 Held-Out 256 (INT8): {res_aligned}", flush=True)
    print(f"[quantize_q8] O3 Occlusion 16 (INT8): gt={res_occ['mean_gt_drop_ratio']:.4f}, pred={res_occ['mean_pred_drop_ratio']:.4f}", flush=True)
    print(f"[quantize_q8] Worst Q8-vs-FP16 probe RMSE: {worst_vs_fp16:.6f}", flush=True)


if __name__ == "__main__":
    main()
