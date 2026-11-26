"""Colab A100 BF16 PyTorch + NVIDIA Warp GPU BVH Ray-Tracing smoke probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.cli_client import ColabClient
from colab.session import ManagedColabSession

REMOTE_PROBE_CODE = r'''
import json, subprocess, sys, time

# Ensure warp-lang is installed
try:
    import warp as wp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "warp-lang"])
    import warp as wp

import numpy as np
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA is not available!"
gpu_name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
vram_gb = round(props.total_memory / (1024 ** 3), 2)
bf16_supported = bool(torch.cuda.is_bf16_supported())

# 1. BF16 SDPA benchmark
B, H, N, D = 16, 8, 640, 48
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, N, 32, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    out = F.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
sdpa_ms = (time.perf_counter() - t0) * 1000.0 / 50.0

# 2. NVIDIA Warp GPU BVH raycast smoke test (unit box facing +Z at z=2.0)
wp.init()
wp_device = "cuda:0"
verts = np.array([
    [-1.0, -1.0, 2.0],
    [ 1.0, -1.0, 2.0],
    [ 1.0,  1.0, 2.0],
    [-1.0,  1.0, 2.0],
], dtype=np.float32)
faces = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
mesh = wp.Mesh(
    points=wp.array(verts, dtype=wp.vec3, device=wp_device),
    indices=wp.array(faces, dtype=int, device=wp_device),
)

@wp.kernel
def trace_smoke(mesh_id: wp.uint64, out_t: wp.array(dtype=float)):
    tid = wp.tid()
    ro = wp.vec3(0.0, 0.0, 0.0)
    rd = wp.vec3(0.0, 0.0, 1.0)
    query = wp.mesh_query_ray(mesh_id, ro, rd, 100.0)
    if query.result:
        out_t[tid] = query.t
    else:
        out_t[tid] = -1.0

n_rays = 65536
out_t = wp.zeros(n_rays, dtype=float, device=wp_device)
wp.launch(trace_smoke, dim=n_rays, inputs=[mesh.id, out_t], device=wp_device)
wp.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    wp.launch(trace_smoke, dim=n_rays, inputs=[mesh.id, out_t], device=wp_device)
wp.synchronize()
warp_ms = (time.perf_counter() - t0) * 1000.0 / 20.0
t_vals = out_t.numpy()
ray_hit_error = float(np.max(np.abs(t_vals - 2.0)))

report = {
    "gpu_name": gpu_name,
    "vram_gb": vram_gb,
    "bf16_supported": bf16_supported,
    "torch_version": torch.__version__,
    "warp_version": wp.__version__,
    "sdpa_bf16_ms_per_iter": round(sdpa_ms, 4),
    "warp_65k_rays_ms": round(warp_ms, 4),
    "warp_ray_hit_max_error": ray_hit_error,
    "passed": bf16_supported and ray_hit_error < 1e-5,
}
print("SMOKE_PROBE_JSON=" + json.dumps(report))
assert report["passed"], f"Smoke probe failed: {report}"
'''


def run_probe_on_session(session_name: str, client: ColabClient | None = None) -> dict:
    cli = client or ColabClient()
    out = cli.exec_code(session_name, REMOTE_PROBE_CODE, timeout=600, stream=True)
    for line in out.splitlines():
        if line.startswith("SMOKE_PROBE_JSON="):
            return json.loads(line.split("SMOKE_PROBE_JSON=", 1)[1])
    raise RuntimeError("Did not find SMOKE_PROBE_JSON marker in probe output")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Colab A100 BF16 + NVIDIA Warp GPU BVH")
    parser.add_argument("--session", type=str, default=None, help="Existing warm Colab session name")
    parser.add_argument("--gpu", type=str, default="A100", help="GPU type if creating ephemeral session")
    args = parser.parse_args()

    if args.session:
        report = run_probe_on_session(args.session)
    else:
        with ManagedColabSession(gpu=args.gpu, notes="smoke_probe") as sess:
            report = run_probe_on_session(sess.name, sess.client)

    out_path = REPO_ROOT / "artifacts" / "smoke_probe_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[smoke_probe] Saved report to {out_path}")


if __name__ == "__main__":
    main()
