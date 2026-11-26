"""Export RADIANT-GI weights to compact < 20 MB FP16 RDGI binary format + PyTorch parity probes.

Binary Format (`RDGI` v1):
  Header (64 bytes, little-endian uint32 / float32):
    [0..4]   magic       = b'RDGI'
    [4..8]   version     = uint32(1)
    [8..12]  d_model     = uint32(256)
    [12..16] n_heads     = uint32(8)
    [16..20] n_layers    = uint32(9)
    [20..24] d_ff        = uint32(1024)
    [24..28] n_tensors   = uint32(N_T)
    [28..32] flags       = uint32(1)  # 1 = fp16 payload
    [32..36] pos_scale   = float32(4.0)
    [36..64] reserved    = zeros (28 bytes)

  Followed immediately by contiguous IEEE-754 float16 tensor payloads in EXACT canonical order
  (all Linear weights stored as [out_features, in_features] row-major, matching PyTorch & WGSL!):
    1. stem1.weight [d_model, 13], stem1.bias [d_model]
    2. stem2.weight [d_model, d_model], stem2.bias [d_model]
    3. For layer l in 0..n_layers-1:
       - norm1.weight [d_model]
       - attn.w_q_sem.weight [n_heads*19, d_model], attn.w_q_sem.bias [n_heads*19]
       - attn.w_k_sem.weight [n_heads*19, d_model], attn.w_k_sem.bias [n_heads*19]
       - attn.w_v_inv.weight [n_heads*26, d_model], attn.w_v_inv.bias [n_heads*26]
       - attn.w_geo_q.weight [n_heads*7, d_model],  attn.w_geo_q.bias [n_heads*7]
       - attn.w_geo_k.weight [n_heads*7, d_model],  attn.w_geo_k.bias [n_heads*7]
       - attn.w_o.weight     [d_model, d_model],    attn.w_o.bias     [d_model]
       - norm2.weight [d_model]
       - fc1.weight   [d_ff, d_model],              fc1.bias          [d_ff]
       - fc2.weight   [d_model, d_ff],              fc2.bias          [d_model]
    4. final_norm.weight [d_model]
    5. head1.weight [d_model, d_model], head1.bias [d_model]
    6. head2.weight [12, d_model],      head2.bias [12]
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
import numpy as np
import torch

from radiant.model.radiant_transformer import RadiantGITransformer
from radiant.sh_math import POS_SCALE_METERS, sh_rmse, tonemapped_psnr_db


def canonical_tensor_names(n_layers: int) -> list[str]:
    names = [
        "stem1.weight",
        "stem1.bias",
        "stem2.weight",
        "stem2.bias",
    ]
    for l in range(n_layers):
        p = f"blocks.{l}."
        names += [
            f"{p}norm1.weight",
            f"{p}attn.w_q_sem.weight",
            f"{p}attn.w_q_sem.bias",
            f"{p}attn.w_k_sem.weight",
            f"{p}attn.w_k_sem.bias",
            f"{p}attn.w_v_inv.weight",
            f"{p}attn.w_v_inv.bias",
            f"{p}attn.w_geo_q.weight",
            f"{p}attn.w_geo_q.bias",
            f"{p}attn.w_geo_k.weight",
            f"{p}attn.w_geo_k.bias",
            f"{p}attn.w_o.weight",
            f"{p}attn.w_o.bias",
            f"{p}norm2.weight",
            f"{p}fc1.weight",
            f"{p}fc1.bias",
            f"{p}fc2.weight",
            f"{p}fc2.bias",
        ]
    names += [
        "final_norm.weight",
        "head1.weight",
        "head1.bias",
        "head2.weight",
        "head2.bias",
    ]
    return names


def export_radiant_fp16_bin(
    model: RadiantGITransformer,
    out_path: Path,
) -> dict:
    """Serialize RadiantGITransformer to compact FP16 RDGI binary and return metadata."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    names = canonical_tensor_names(model.n_layers)

    header = struct.pack(
        "<4s7If28s",
        b"RDGI",
        1,
        model.d_model,
        model.n_heads,
        model.n_layers,
        model.d_ff,
        len(names),
        1,  # flags: fp16
        float(POS_SCALE_METERS),
        b"\x00" * 28,
    )
    assert len(header) == 64

    total_elems = 0
    with out_path.open("wb") as f:
        f.write(header)
        for name in names:
            t = sd[name].detach().cpu().float().numpy().astype(np.float16)
            t = np.ascontiguousarray(t, dtype="<f2")
            f.write(t.tobytes())
            total_elems += int(t.size)

    size_bytes = out_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    assert size_mb < 20.0, f"Exported binary {size_mb:.2f} MB exceeds 20 MB cap!"
    return {
        "path": str(out_path),
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 3),
        "n_tensors": len(names),
        "total_params": total_elems,
    }


def load_radiant_fp16_bin(bin_path: Path, device: str | torch.device = "cpu") -> RadiantGITransformer:
    """Reload RDGI FP16 binary into a PyTorch RadiantGITransformer (dequantized to FP32 weights)."""
    raw = bin_path.read_bytes()
    magic, version, d_model, n_heads, n_layers, d_ff, n_tensors, flags, pos_scale, _ = struct.unpack(
        "<4s7If28s", raw[:64]
    )
    assert magic == b"RDGI" and version == 1, f"Invalid RDGI header: {magic} v{version}"
    assert abs(pos_scale - POS_SCALE_METERS) < 1e-4

    model = RadiantGITransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
    sd = model.state_dict()
    names = canonical_tensor_names(n_layers)
    assert len(names) == n_tensors

    off = 64
    new_sd = {}
    for name in names:
        shape = tuple(sd[name].shape)
        count = int(np.prod(shape))
        nbytes = count * 2
        arr = np.frombuffer(raw, dtype="<f2", count=count, offset=off).reshape(shape)
        new_sd[name] = torch.from_numpy(arr.astype(np.float32).copy())
        off += nbytes
    assert off == len(raw), f"Trailing bytes in RDGI file: {len(raw) - off}"
    model.load_state_dict(new_sd)
    return model.to(device).eval()


@torch.no_grad()
def export_parity_probes(
    fp16_bin_path: Path,
    heldout_data: dict[str, np.ndarray],
    out_path: Path,
    n_probes: int = 4,
) -> dict:
    """Generate locked PyTorch FP16-weight reference outputs on matched held-out scenes for O4 WGSL parity."""
    model = load_radiant_fp16_bin(fp16_bin_path, device="cpu")
    offsets = heldout_data["scene_offsets"]
    tok_all = heldout_data["tokens_fp16"]
    tgt_all = heldout_data["targets_fp16"]
    oids_all = heldout_data["object_ids"]

    probes = []
    for i in range(min(n_probes, len(offsets) - 1)):
        s, e = int(offsets[i]), int(offsets[i + 1])
        tok = tok_all[s:e].astype(np.float32)
        gt = tgt_all[s:e].astype(np.float32)
        oid = oids_all[s:e].astype(np.int32)
        pred = model(torch.from_numpy(tok).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
        probes.append(
            {
                "scene_index": i,
                "n_tokens": int(e - s),
                "psnr_vs_gt_db": round(tonemapped_psnr_db(pred[:, 0:3], gt[:, 0:3]), 3),
                "rmse_vs_gt": round(sh_rmse(pred, gt), 5),
                "tokens": np.round(tok, 6).tolist(),
                "object_ids": oid.tolist(),
                "gt_sh": np.round(gt, 6).tolist(),
                "pytorch_fp16_ref_sh": np.round(pred, 6).tolist(),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "RADIANT_O4_PARITY_PROBES_V1",
        "pos_scale_meters": POS_SCALE_METERS,
        "n_probes": len(probes),
        "probes": probes,
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    return {"path": str(out_path), "n_probes": len(probes), "size_mb": round(size_mb, 3)}
