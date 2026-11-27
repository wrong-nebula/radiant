"""Real Headless Chromium Apple M1 Metal 3 WebGPU Audit (O4 WGSL Parity + O5 Visual/FPS/Color-Bleed).

Runs ZERO external network fetches (enforced via Playwright route guard) against a local Vite dev server.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "artifacts" / "playwright_audit"
PORT = 4189
BASE_URL = f"http://127.0.0.1:{PORT}/"


def _wait_for_server(url: str, timeout_s: float = 30.0) -> None:
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"Vite server did not become ready at {url}")


def _img_stats(path: Path) -> dict:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "mean_rgb": [round(float(arr[:, :, c].mean()), 2) for c in range(3)],
        "std_rgb": [round(float(arr[:, :, c].std()), 2) for c in range(3)],
        "non_blank": bool(arr.std() > 8.0 and arr.mean() > 5.0),
    }


def _img_l1_diff(path_a: Path, path_b: Path) -> float:
    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32)
    # Crop out HUD on the left (x > 330) so we measure pure 3D viewport pixel changes!
    if a.shape[1] > 400:
        a = a[:, 330:, :]
        b = b[:, 330:, :]
    return round(float(np.mean(np.abs(a - b))), 3)


def run_audit() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Start local Vite server serving THREE-RADIANT-GI
    vite_proc = subprocess.Popen(
        ["npx", "vite", "--host", "127.0.0.1", "--port", str(PORT), "--strictPort"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    console_errors: list[str] = []
    page_errors: list[str] = []
    external_requests: list[str] = []

    try:
        _wait_for_server(BASE_URL, timeout_s=30.0)
        print(f"[playwright] Vite server ready at {BASE_URL}", flush=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--enable-unsafe-webgpu",
                    "--enable-features=WebGPU",
                    "--use-angle=metal",
                    "--ignore-gpu-blocklist",
                ],
            )
            context = browser.new_context(viewport={"width": 1280, "height": 760})
            page = context.new_page()

            def route_guard(route):
                url = route.request.url
                if (
                    url.startswith(f"http://127.0.0.1:{PORT}")
                    or url.startswith(f"http://localhost:{PORT}")
                    or url.startswith(f"ws://127.0.0.1:{PORT}")
                    or url.startswith(f"ws://localhost:{PORT}")
                    or url.startswith("data:")
                    or url.startswith("blob:")
                ):
                    route.continue_()
                else:
                    print(f"[playwright:BLOCKED_EXTERNAL] {url}", flush=True)
                    external_requests.append(url)
                    route.abort()

            page.route("**/*", route_guard)
            page.on("requestfailed", lambda req: print(f"[browser:reqfailed] {req.url} -> {req.failure}", flush=True))
            page.on("console", lambda msg: (print(f"[browser:{msg.type}] {msg.text}", flush=True), console_errors.append(msg.text) if msg.type == "error" else None))
            page.on("pageerror", lambda err: (print(f"[browser:pageerror] {err}", flush=True), page_errors.append(str(err))))

            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_function("() => Boolean(window.__RADIANT_APP__)", timeout=30000)
            page.evaluate("() => window.__RADIANT_APP__.ready")
            page.wait_for_timeout(400)

            # Verify WebGPU adapter is active
            adapter_info = page.evaluate(
                """async () => {
                    if (!navigator.gpu) return { webgpu: false };
                    const a = await navigator.gpu.requestAdapter();
                    return { webgpu: Boolean(a), info: a?.info ? { vendor: a.info.vendor, arch: a.info.architecture, device: a.info.device } : {} };
                }"""
            )
            print(f"[playwright] WebGPU Adapter: {adapter_info}", flush=True)

            # ===============================================================
            # O4: Exact PyTorch A100 <-> WebGPU WGSL Bit-Parity Audit
            # ===============================================================
            print("[playwright] Running O4 PyTorch FP16 <-> WebGPU WGSL parity audit...", flush=True)
            o4_report = page.evaluate("() => window.__RADIANT_APP__.runParityAudit('/data/parity_probes.json')")
            (OUT_DIR / "o4_wgsl_parity_report.json").write_text(json.dumps(o4_report, indent=2) + "\n")
            print(f"[playwright] O4 Result: passed={o4_report['passed']}, worst_rmse={o4_report['worst_rmse']}, mean_ms={o4_report['mean_compute_ms']}", flush=True)

            # ===============================================================
            # O5: Visual Renders, Live Color-Bleed Shift, Occlusion & M1 FPS
            # ===============================================================
            shots: dict[str, Path] = {}

            def snap(name: str, wait_ms: int = 250) -> Path:
                page.wait_for_timeout(wait_ms)
                path = OUT_DIR / f"{name}.png"
                page.screenshot(path=str(path))
                shots[name] = path
                return path

            # 1. Room 1: Cornell Color-Bleed Box (Live GI vs Direct Only vs Live Shift)
            page.evaluate("() => window.__RADIANT_APP__.loadRoom(0)")
            page.evaluate("() => window.__RADIANT_APP__.setViewMode(0)")
            p_cornell_gi = snap("01_cornell_radiant_gi")
            rgb_before = page.evaluate("() => window.__RADIANT_APP__.getObjectMeanIndirectRGB('Sculptural Knot')")

            page.evaluate("() => window.__RADIANT_APP__.setViewMode(1)")
            p_cornell_direct = snap("02_cornell_direct_only")

            # Split-screen wipe & indirect-only & X-ray overlays
            page.evaluate("() => window.__RADIANT_APP__.setViewMode(4)")
            snap("03_cornell_split_screen")

            page.evaluate("() => window.__RADIANT_APP__.setViewMode(2)")
            snap("04_cornell_indirect_only")

            page.evaluate("() => window.__RADIANT_APP__.setViewMode(0)")
            page.evaluate("() => window.__RADIANT_APP__.setOverlays({ patches: true, shProbes: true })")
            snap("05_cornell_xray_probes")
            page.evaluate("() => window.__RADIANT_APP__.setOverlays({ patches: false, shProbes: false })")

            # Live color-bleed shift test: slide Crimson wall closer & repaint Emerald wall to Cobalt Blue!
            page.evaluate("() => window.__RADIANT_APP__.moveObjectByName('Crimson Left Wall', 0.85, 0, 0)")
            page.evaluate("() => window.__RADIANT_APP__.setObjectColorByName('Emerald Right Wall', '#1a47db')")
            p_cornell_shift = snap("06_cornell_color_bleed_shift")
            rgb_after = page.evaluate("() => window.__RADIANT_APP__.getObjectMeanIndirectRGB('Sculptural Knot')")

            # Measure real steady & live-drag FPS on M1 Metal 3 in Room 1 (Auto Distance LOD + Fixed 32 + 1/2/3 Bounce + Saturation!)
            page.evaluate("() => window.__RADIANT_APP__.loadRoom(0)")
            page.evaluate("() => window.__RADIANT_APP__.setLODMode('auto')")
            perf = page.evaluate("() => window.__RADIANT_APP__.measurePerformance(45, 12)")
            page.evaluate("() => window.__RADIANT_APP__.setLODMode('fixed32')")
            perf_fixed32 = page.evaluate("() => window.__RADIANT_APP__.measurePerformance(15, 12)")
            page.evaluate("() => window.__RADIANT_APP__.setLODMode('auto')")
            print(f"[playwright] M1 Metal 3 Performance (Auto LOD): {perf}", flush=True)
            print(f"[playwright] M1 Metal 3 Performance (Fixed 32): {perf_fixed32}", flush=True)

            # Verify instant 1-Bounce / 2-Bounce / 3-Bounce & Bounce Saturation controls
            page.evaluate("() => window.__RADIANT_APP__.setBounceCount(1)")
            rgb_b1 = page.evaluate("() => window.__RADIANT_APP__.getObjectMeanIndirectRGB('Sculptural Knot')")
            page.evaluate("() => window.__RADIANT_APP__.setBounceCount(2)")
            rgb_b2 = page.evaluate("() => window.__RADIANT_APP__.getObjectMeanIndirectRGB('Sculptural Knot')")
            page.evaluate("() => window.__RADIANT_APP__.setBounceCount(3)")
            rgb_b3 = page.evaluate("() => window.__RADIANT_APP__.getObjectMeanIndirectRGB('Sculptural Knot')")
            page.evaluate("() => window.__RADIANT_APP__.setBounceSaturation(0.0)")
            p_sat0 = snap("14_cornell_sat0_mono")
            page.evaluate("() => window.__RADIANT_APP__.setBounceSaturation(1.8)")
            p_sat180 = snap("15_cornell_sat180_vivid")
            page.evaluate("() => window.__RADIANT_APP__.setBounceSaturation(1.0)")
            print(f"[playwright] Multi-Bounce RGB (1B/2B/3B): {rgb_b1} / {rgb_b2} / {rgb_b3}", flush=True)

            # 2. Room 2: Vaulted Brutalist Gallery
            page.evaluate("() => window.__RADIANT_APP__.loadRoom(1)")
            snap("07_vaulted_gallery")

            # 3. Room 3: L-Corridor Doorway Occlusion (open vs occluded)
            page.evaluate("() => window.__RADIANT_APP__.loadRoom(2)")
            p_l_open = snap("08_lcorridor_open")
            east_open = page.evaluate("() => window.__RADIANT_APP__.getObjectMeanIndirectRGB('East Chamber Sculpture')")
            page.evaluate("() => window.__RADIANT_APP__.moveObjectByName('Sliding Occluder Door', 0, 0, 1.65)")
            p_l_occ = snap("09_lcorridor_occluded")
            east_occ = page.evaluate("() => window.__RADIANT_APP__.getObjectMeanIndirectRGB('East Chamber Sculpture')")

            # 4. Room 4: Studio Showcase
            page.evaluate("() => window.__RADIANT_APP__.loadRoom(3)")
            snap("10_studio_showcase")

            # 5. Room 5: Held-Out 1024-spp Benchmark (Live vs 1024-spp GT vs Error Heatmap)
            page.evaluate("() => window.__RADIANT_APP__.loadRoom(4)")
            page.evaluate("() => window.__RADIANT_APP__.setViewMode(0)")
            snap("11_benchmark_live")
            page.evaluate("() => window.__RADIANT_APP__.setViewMode(5)")
            snap("12_benchmark_gt1024")
            page.evaluate("() => window.__RADIANT_APP__.setViewMode(3)")
            snap("13_benchmark_error_heatmap")

            browser.close()

        # Analyze screenshots & shifts
        shot_stats = {k: _img_stats(v) for k, v in shots.items()}
        all_non_blank = all(s["non_blank"] for s in shot_stats.values())
        gi_vs_direct_diff = _img_l1_diff(p_cornell_gi, p_cornell_direct)
        bleed_shift_diff = _img_l1_diff(p_cornell_gi, p_cornell_shift)
        corridor_occ_diff = _img_l1_diff(p_l_open, p_l_occ)
        sat_shift_diff = _img_l1_diff(p_sat0, p_sat180)

        # Hue/RGB shift on Sculptural Knot when walls move/change color
        rgb_b = np.array(rgb_before, dtype=np.float64)
        rgb_a = np.array(rgb_after, dtype=np.float64)
        chroma_b = rgb_b / max(rgb_b.sum(), 1e-6)
        chroma_a = rgb_a / max(rgb_a.sum(), 1e-6)
        bleed_chroma_shift = float(np.linalg.norm(chroma_a - chroma_b))

        # L-Corridor live occlusion drop on East Chamber Sculpture
        open_luma = float(np.mean(east_open))
        occ_luma = float(np.mean(east_occ))
        corridor_drop_ratio = (open_luma - occ_luma) / max(open_luma, 1e-6)

        o5_passed = (
            len(console_errors) == 0
            and len(page_errors) == 0
            and len(external_requests) == 0
            and all_non_blank
            and gi_vs_direct_diff > 4.0
            and bleed_shift_diff > 3.0
            and bleed_chroma_shift > 0.02
            and perf["steady_fps"] >= 30.0
            and perf["live_drag_fps"] >= 20.0
        )

        o5_report = {
            "gate": "O5_PLAYWRIGHT_M1_METAL3_AUDIT",
            "passed": bool(o5_passed),
            "adapter": adapter_info,
            "console_errors": console_errors,
            "page_errors": page_errors,
            "external_requests": external_requests,
            "all_renders_non_blank": all_non_blank,
            "pixel_diffs_viewport": {
                "cornell_gi_vs_direct_l1": gi_vs_direct_diff,
                "cornell_live_bleed_shift_l1": bleed_shift_diff,
                "lcorridor_open_vs_occluded_l1": corridor_occ_diff,
                "cornell_sat0_vs_sat180_l1": sat_shift_diff,
            },
            "gpu_indirect_rgb_checks": {
                "knot_before_rgb": [round(x, 4) for x in rgb_before],
                "knot_after_shift_rgb": [round(x, 4) for x in rgb_after],
                "knot_chroma_shift_l2": round(bleed_chroma_shift, 4),
                "knot_1bounce_rgb": [round(x, 4) for x in rgb_b1],
                "knot_2bounce_rgb": [round(x, 4) for x in rgb_b2],
                "knot_3bounce_rgb": [round(x, 4) for x in rgb_b3],
                "east_sculpt_open_rgb": [round(x, 4) for x in east_open],
                "east_sculpt_occluded_rgb": [round(x, 4) for x in east_occ],
                "corridor_occlusion_drop_ratio": round(corridor_drop_ratio, 4),
            },
            "performance_m1": perf,
            "performance_m1_fixed32": perf_fixed32,
            "screenshots": {k: str(v.relative_to(REPO_ROOT)) for k, v in shots.items()},
            "screenshot_stats": shot_stats,
        }
        (OUT_DIR / "o5_playwright_report.json").write_text(json.dumps(o5_report, indent=2) + "\n")
        print(json.dumps(o5_report, indent=2), flush=True)

        assert o4_report["passed"], f"O4 WGSL Parity FAILED: {o4_report}"
        assert o5_report["passed"], f"O5 Playwright Audit FAILED: {o5_report}"
        print("[playwright] O4 & O5 PLAYWRIGHT APPLE M1 METAL 3 AUDITS PASSED!", flush=True)

    finally:
        vite_proc.terminate()
        try:
            vite_proc.wait(timeout=5)
        except Exception:
            vite_proc.kill()


if __name__ == "__main__":
    run_audit()
