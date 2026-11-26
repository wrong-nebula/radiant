/**
 * Dedicated off-main-thread Web Worker for the CPU fallback (`WebGL2 · 9L CPU GI`).
 * Keeps Firefox macOS orbiting & UI at 60 fps with zero main-thread stutter while
 * running the L1-cache-blocked 9-layer RADIANT SE(3) Form-Factor Transformer!
 */

import {
  CPUWeightLayout,
  runRadiantCPU,
  unpackRDG8ToCanonicalF32,
  unpackRDGIWeightsF32,
} from './cpuFallback.ts';
import type { RDGIHeader } from './RadiantComputeEngine.ts';

let weightsF32: Float32Array | null = null;
let weightLayout: CPUWeightLayout | null = null;
let modelHeader: RDGIHeader | null = null;

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      const buf: ArrayBuffer = msg.buffer;
      modelHeader = msg.header as RDGIHeader;
      if (msg.isQ8) {
        const unpacked = unpackRDG8ToCanonicalF32(buf, modelHeader, msg.q8U32Len, msg.f32Len);
        weightsF32 = unpacked.weights;
        weightLayout = unpacked.layout;
      } else {
        weightsF32 = unpackRDGIWeightsF32(buf);
        weightLayout = msg.layout as CPUWeightLayout;
      }
      self.postMessage({ type: 'init_done', id: msg.id });
    } else if (msg.type === 'infer') {
      if (!weightsF32 || !weightLayout || !modelHeader) {
        self.postMessage({ type: 'infer_error', id: msg.id, error: 'Worker weights not initialized' });
        return;
      }
      const t0 = performance.now();
      const tokens = new Float32Array(msg.tokens);
      const result = await runRadiantCPU(tokens, msg.nTokens, modelHeader, weightsF32, weightLayout);
      const computeMs = performance.now() - t0;
      (self as unknown as Worker).postMessage(
        { type: 'infer_done', id: msg.id, result: result.buffer, computeMs },
        [result.buffer],
      );
    }
  } catch (err) {
    self.postMessage({ type: 'infer_error', id: msg.id, error: String(err) });
  }
};
