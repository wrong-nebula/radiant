/**
 * Exact FP32 CPU fallback for THREE-RADIANT-GI when a browser lacks a WebGPU adapter
 * (e.g. Firefox macOS where `navigator.gpu.requestAdapter()` returns null and
 * Three.js `WebGPURenderer` automatically falls back to `WebGLBackend`).
 *
 * Implements the exact same 9-layer SE(3)-Invariant Form-Factor Transformer math as
 * `RadiantComputeEngine.ts` WGSL compute shaders, unpacking the identical `RDGI` FP16 binary.
 */

import { SH_STRIDE, TOKEN_STRIDE } from './shMath.ts';
import type { RDGIHeader } from './RadiantComputeEngine.ts';

// Precomputed exact IEEE-754 binary16 -> float32 lookup table (65536 entries = 256 KB)
let f16Lut: Float32Array | null = null;

function getF16Lut(): Float32Array {
  if (f16Lut) return f16Lut;
  const lut = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = (h >> 15) & 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    let v: number;
    if (e === 0) {
      v = (m === 0 ? 0 : m * Math.pow(2, -24));
    } else if (e === 31) {
      v = m === 0 ? Infinity : NaN;
    } else {
      v = Math.pow(2, e - 15) * (1 + m / 1024);
    }
    lut[h] = s ? -v : v;
  }
  f16Lut = lut;
  return lut;
}

export function unpackRDGIWeightsF32(buffer: ArrayBuffer): Float32Array {
  const lut = getF16Lut();
  const u16 = new Uint16Array(buffer, 64);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    out[i] = lut[u16[i]];
  }
  return out;
}

/**
 * Unpack `RDG8` v2 (per-row symmetric INT8 + FP32 aux) into the exact same canonical
 * dequantized FP32 layout used by `runRadiantCPU`, so Firefox/WebGL2 also benefits from
 * the 2x smaller 6.76 MB `radiant_gi_q8.bin` download!
 */
export function unpackRDG8ToCanonicalF32(
  buffer: ArrayBuffer,
  header: RDGIHeader,
  q8U32Len: number,
  f32Len: number,
): { weights: Float32Array; layout: CPUWeightLayout } {
  const { dModel, nHeads, nLayers, dFF } = header;
  const q8Bytes = new Int8Array(buffer, 64, q8U32Len * 4);
  const f32Aux = new Float32Array(buffer, 64 + q8U32Len * 4, f32Len);

  // Allocate canonical float array matching RDGI layout
  const head2OutDim = header.isMultiBounce ? 36 : 12;
  let totalElems = dModel * 13 + dModel + dModel * dModel + dModel;
  const qkvRows = nHeads * 78;
  totalElems +=
    nLayers *
    (dModel + qkvRows * dModel + qkvRows + dModel * dModel + dModel + dModel + dFF * dModel + dFF + dModel * dFF + dModel);
  totalElems += dModel + dModel * dModel + dModel + head2OutDim * dModel + head2OutDim;

  const out = new Float32Array(totalElems);
  let outPos = 0;
  let q8Pos = 0;
  let auxPos = 0;

  const alloc = (count: number): number => {
    const u32Off = outPos >> 1;
    outPos += count;
    return u32Off;
  };
  const copyAux = (count: number): number => {
    const off = alloc(count);
    out.set(f32Aux.subarray(auxPos, auxPos + count), off * 2);
    auxPos += count;
    return off;
  };
  const deqMat = (rows: number, cols: number): { wOff: number; bOff: number } => {
    const wOff = alloc(rows * cols);
    const bOff = alloc(rows);
    const scales = f32Aux.subarray(auxPos, auxPos + rows);
    auxPos += rows;
    out.set(f32Aux.subarray(auxPos, auxPos + rows), bOff * 2);
    auxPos += rows;
    const dstBase = wOff * 2;
    const inv127 = 1.0 / 127.0;
    for (let r = 0; r < rows; r++) {
      const s = scales[r] * inv127;
      const rDst = dstBase + r * cols;
      for (let c = 0; c < cols; c++) {
        out[rDst + c] = q8Bytes[q8Pos++] * s;
      }
    }
    return { wOff, bOff };
  };

  const stem1W = copyAux(dModel * 13);
  const stem1B = copyAux(dModel);
  const stem2 = deqMat(dModel, dModel);

  const layers: CPULayerOffsets[] = [];
  for (let l = 0; l < nLayers; l++) {
    const norm1W = copyAux(dModel);
    // Unpack fused QKV+Geo [624, dModel] into 5 slices matching CPULayerOffsets
    const qkvScales = f32Aux.subarray(auxPos, auxPos + qkvRows);
    auxPos += qkvRows;
    const qkvBias = f32Aux.subarray(auxPos, auxPos + qkvRows);
    auxPos += qkvRows;

    const sliceMat = (rows: number, rowStart: number): { wOff: number; bOff: number } => {
      const wOff = alloc(rows * dModel);
      const bOff = alloc(rows);
      out.set(qkvBias.subarray(rowStart, rowStart + rows), bOff * 2);
      const dstBase = wOff * 2;
      const inv127 = 1.0 / 127.0;
      for (let r = 0; r < rows; r++) {
        const s = qkvScales[rowStart + r] * inv127;
        const rDst = dstBase + r * dModel;
        for (let c = 0; c < dModel; c++) {
          out[rDst + c] = q8Bytes[q8Pos++] * s;
        }
      }
      return { wOff, bOff };
    };

    const qSem = sliceMat(nHeads * 19, 0);
    const kSem = sliceMat(nHeads * 19, nHeads * 19);
    const vInv = sliceMat(nHeads * 26, nHeads * 38);
    const geoQ = sliceMat(nHeads * 7, nHeads * 64);
    const geoK = sliceMat(nHeads * 7, nHeads * 71);
    const wO = deqMat(dModel, dModel);
    const norm2W = copyAux(dModel);
    const fc1 = deqMat(dFF, dModel);
    const fc2 = deqMat(dModel, dFF);

    layers.push({
      norm1W,
      qSemW: qSem.wOff, qSemB: qSem.bOff,
      kSemW: kSem.wOff, kSemB: kSem.bOff,
      vInvW: vInv.wOff, vInvB: vInv.bOff,
      geoQW: geoQ.wOff, geoQB: geoQ.bOff,
      geoKW: geoK.wOff, geoKB: geoK.bOff,
      wOW: wO.wOff, wOB: wO.bOff,
      norm2W,
      fc1W: fc1.wOff, fc1B: fc1.bOff,
      fc2W: fc2.wOff, fc2B: fc2.bOff,
    });
  }

  const finalNormW = copyAux(dModel);
  const head1 = deqMat(dModel, dModel);
  const head2W = copyAux(head2OutDim * dModel);
  const head2B = copyAux(head2OutDim);

  return {
    weights: out,
    layout: {
      stem1W, stem1B,
      stem2W: stem2.wOff, stem2B: stem2.bOff,
      layers,
      finalNormW,
      head1W: head1.wOff, head1B: head1.bOff,
      head2W, head2B,
    },
  };
}

function geluTanh(x: number): number {
  const arg = Math.max(-15.0, Math.min(15.0, 0.7978845608028654 * (x + 0.044715 * x * x * x)));
  return 0.5 * x * (1.0 + Math.tanh(arg));
}

function softplusClamped(x: number, cap: number): number {
  const sp = x > 20.0 ? x : Math.log(1.0 + Math.exp(x));
  return sp > cap ? cap : sp < 0 ? 0 : sp;
}

function rmsNorm(
  xIn: Float32Array,
  w: Float32Array,
  wOff: number,
  xOut: Float32Array,
  nTokens: number,
  dim: number,
): void {
  for (let t = 0; t < nTokens; t++) {
    const base = t * dim;
    let sq = 0.0;
    for (let d = 0; d < dim; d++) {
      const v = xIn[base + d];
      sq += v * v;
    }
    const invRms = 1.0 / Math.sqrt(sq / dim + 1e-5);
    for (let d = 0; d < dim; d++) {
      xOut[base + d] = xIn[base + d] * invRms * w[wOff + d];
    }
  }
}

function linearLayer(
  xIn: Float32Array,
  w: Float32Array,
  wOff: number,
  bOff: number,
  resIn: Float32Array | null,
  yOut: Float32Array,
  nTokens: number,
  inDim: number,
  outDim: number,
  actGelu: boolean,
): void {
  // L1-cache-resident output-outer 4-output x 4-token blocked GEMM:
  // Reads each 1 MB weight matrix ONCE from RAM instead of 88x (huge Firefox/CPU speedup!)
  const outDim4 = outDim & ~3;
  const nTok4 = nTokens & ~3;
  const inDim4 = inDim & ~3;

  for (let o = 0; o < outDim4; o += 4) {
    const wb0 = wOff + (o + 0) * inDim;
    const wb1 = wOff + (o + 1) * inDim;
    const wb2 = wOff + (o + 2) * inDim;
    const wb3 = wOff + (o + 3) * inDim;
    const b0 = w[bOff + o], b1 = w[bOff + o + 1], b2 = w[bOff + o + 2], b3 = w[bOff + o + 3];

    for (let t = 0; t < nTok4; t += 4) {
      const x0 = t * inDim, x1 = (t + 1) * inDim, x2 = (t + 2) * inDim, x3 = (t + 3) * inDim;
      let s00 = b0, s01 = b1, s02 = b2, s03 = b3;
      let s10 = b0, s11 = b1, s12 = b2, s13 = b3;
      let s20 = b0, s21 = b1, s22 = b2, s23 = b3;
      let s30 = b0, s31 = b1, s32 = b2, s33 = b3;

      let k = 0;
      for (; k < inDim4; k += 4) {
        const w0a = w[wb0 + k], w0b = w[wb0 + k + 1], w0c = w[wb0 + k + 2], w0d = w[wb0 + k + 3];
        const w1a = w[wb1 + k], w1b = w[wb1 + k + 1], w1c = w[wb1 + k + 2], w1d = w[wb1 + k + 3];
        const w2a = w[wb2 + k], w2b = w[wb2 + k + 1], w2c = w[wb2 + k + 2], w2d = w[wb2 + k + 3];
        const w3a = w[wb3 + k], w3b = w[wb3 + k + 1], w3c = w[wb3 + k + 2], w3d = w[wb3 + k + 3];

        const xa0 = xIn[x0 + k], xb0 = xIn[x0 + k + 1], xc0 = xIn[x0 + k + 2], xd0 = xIn[x0 + k + 3];
        s00 += xa0 * w0a + xb0 * w0b + xc0 * w0c + xd0 * w0d;
        s01 += xa0 * w1a + xb0 * w1b + xc0 * w1c + xd0 * w1d;
        s02 += xa0 * w2a + xb0 * w2b + xc0 * w2c + xd0 * w2d;
        s03 += xa0 * w3a + xb0 * w3b + xc0 * w3c + xd0 * w3d;

        const xa1 = xIn[x1 + k], xb1 = xIn[x1 + k + 1], xc1 = xIn[x1 + k + 2], xd1 = xIn[x1 + k + 3];
        s10 += xa1 * w0a + xb1 * w0b + xc1 * w0c + xd1 * w0d;
        s11 += xa1 * w1a + xb1 * w1b + xc1 * w1c + xd1 * w1d;
        s12 += xa1 * w2a + xb1 * w2b + xc1 * w2c + xd1 * w2d;
        s13 += xa1 * w3a + xb1 * w3b + xc1 * w3c + xd1 * w3d;

        const xa2 = xIn[x2 + k], xb2 = xIn[x2 + k + 1], xc2 = xIn[x2 + k + 2], xd2 = xIn[x2 + k + 3];
        s20 += xa2 * w0a + xb2 * w0b + xc2 * w0c + xd2 * w0d;
        s21 += xa2 * w1a + xb2 * w1b + xc2 * w1c + xd2 * w1d;
        s22 += xa2 * w2a + xb2 * w2b + xc2 * w2c + xd2 * w2d;
        s23 += xa2 * w3a + xb2 * w3b + xc2 * w3c + xd2 * w3d;

        const xa3 = xIn[x3 + k], xb3 = xIn[x3 + k + 1], xc3 = xIn[x3 + k + 2], xd3 = xIn[x3 + k + 3];
        s30 += xa3 * w0a + xb3 * w0b + xc3 * w0c + xd3 * w0d;
        s31 += xa3 * w1a + xb3 * w1b + xc3 * w1c + xd3 * w1d;
        s32 += xa3 * w2a + xb3 * w2b + xc3 * w2c + xd3 * w2d;
        s33 += xa3 * w3a + xb3 * w3b + xc3 * w3c + xd3 * w3d;
      }
      for (; k < inDim; k++) {
        const w0k = w[wb0 + k], w1k = w[wb1 + k], w2k = w[wb2 + k], w3k = w[wb3 + k];
        const xv0 = xIn[x0 + k], xv1 = xIn[x1 + k], xv2 = xIn[x2 + k], xv3 = xIn[x3 + k];
        s00 += xv0 * w0k; s01 += xv0 * w1k; s02 += xv0 * w2k; s03 += xv0 * w3k;
        s10 += xv1 * w0k; s11 += xv1 * w1k; s12 += xv1 * w2k; s13 += xv1 * w3k;
        s20 += xv2 * w0k; s21 += xv2 * w1k; s22 += xv2 * w2k; s23 += xv2 * w3k;
        s30 += xv3 * w0k; s31 += xv3 * w1k; s32 += xv3 * w2k; s33 += xv3 * w3k;
      }

      if (actGelu) {
        s00 = geluTanh(s00); s01 = geluTanh(s01); s02 = geluTanh(s02); s03 = geluTanh(s03);
        s10 = geluTanh(s10); s11 = geluTanh(s11); s12 = geluTanh(s12); s13 = geluTanh(s13);
        s20 = geluTanh(s20); s21 = geluTanh(s21); s22 = geluTanh(s22); s23 = geluTanh(s23);
        s30 = geluTanh(s30); s31 = geluTanh(s31); s32 = geluTanh(s32); s33 = geluTanh(s33);
      }
      const y0 = t * outDim + o, y1 = (t + 1) * outDim + o, y2 = (t + 2) * outDim + o, y3 = (t + 3) * outDim + o;
      if (resIn) {
        s00 += resIn[y0]; s01 += resIn[y0 + 1]; s02 += resIn[y0 + 2]; s03 += resIn[y0 + 3];
        s10 += resIn[y1]; s11 += resIn[y1 + 1]; s12 += resIn[y1 + 2]; s13 += resIn[y1 + 3];
        s20 += resIn[y2]; s21 += resIn[y2 + 1]; s22 += resIn[y2 + 2]; s23 += resIn[y2 + 3];
        s30 += resIn[y3]; s31 += resIn[y3 + 1]; s32 += resIn[y3 + 2]; s33 += resIn[y3 + 3];
      }
      yOut[y0] = s00; yOut[y0 + 1] = s01; yOut[y0 + 2] = s02; yOut[y0 + 3] = s03;
      yOut[y1] = s10; yOut[y1 + 1] = s11; yOut[y1 + 2] = s12; yOut[y1 + 3] = s13;
      yOut[y2] = s20; yOut[y2 + 1] = s21; yOut[y2 + 2] = s22; yOut[y2 + 3] = s23;
      yOut[y3] = s30; yOut[y3 + 1] = s31; yOut[y3 + 2] = s32; yOut[y3 + 3] = s33;
    }

    for (let t = nTok4; t < nTokens; t++) {
      const xb = t * inDim;
      let s0 = b0, s1 = b1, s2 = b2, s3 = b3;
      for (let k = 0; k < inDim; k++) {
        const xv = xIn[xb + k];
        s0 += xv * w[wb0 + k]; s1 += xv * w[wb1 + k]; s2 += xv * w[wb2 + k]; s3 += xv * w[wb3 + k];
      }
      if (actGelu) { s0 = geluTanh(s0); s1 = geluTanh(s1); s2 = geluTanh(s2); s3 = geluTanh(s3); }
      const yb = t * outDim + o;
      if (resIn) { s0 += resIn[yb]; s1 += resIn[yb + 1]; s2 += resIn[yb + 2]; s3 += resIn[yb + 3]; }
      yOut[yb] = s0; yOut[yb + 1] = s1; yOut[yb + 2] = s2; yOut[yb + 3] = s3;
    }
  }

  for (let o = outDim4; o < outDim; o++) {
    const wb = wOff + o * inDim;
    const b = w[bOff + o];
    for (let t = 0; t < nTokens; t++) {
      const xb = t * inDim;
      let acc = b;
      for (let k = 0; k < inDim; k++) acc += xIn[xb + k] * w[wb + k];
      if (actGelu) acc = geluTanh(acc);
      const yb = t * outDim + o;
      if (resIn) acc += resIn[yb];
      yOut[yb] = acc;
    }
  }
}

export interface CPULayerOffsets {
  norm1W: number;
  qSemW: number; qSemB: number;
  kSemW: number; kSemB: number;
  vInvW: number; vInvB: number;
  geoQW: number; geoQB: number;
  geoKW: number; geoKB: number;
  wOW: number; wOB: number;
  norm2W: number;
  fc1W: number; fc1B: number;
  fc2W: number; fc2B: number;
}

export interface CPUWeightLayout {
  stem1W: number; stem1B: number;
  stem2W: number; stem2B: number;
  layers: CPULayerOffsets[];
  finalNormW: number;
  head1W: number; head1B: number;
  head2W: number; head2B: number;
}

/**
 * Run exact 9-layer FP32 RADIANT SE(3) Form-Factor Transformer on CPU.
 * Note: layout offsets are in `u32` pairs (so float index = offset * 2).
 */
export async function runRadiantCPU(
  tokens: Float32Array,
  nTokens: number,
  header: RDGIHeader,
  weights: Float32Array,
  layout: CPUWeightLayout,
): Promise<Float32Array> {
  const { dModel, nHeads, nLayers, dFF, posScale } = header;
  const invScale = 1.0 / posScale;

  // 1. Scene centroid
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < nTokens; i++) {
    const tb = i * TOKEN_STRIDE;
    cx += tokens[tb];
    cy += tokens[tb + 1];
    cz += tokens[tb + 2];
  }
  const invN = 1.0 / Math.max(nTokens, 1);
  cx *= invN; cy *= invN; cz *= invN;

  // 2. Geo [N, 16] and Intr [N, 13]
  const useAreaPrior = Boolean(header.hasAreaPrior);
  const geo = new Float32Array(nTokens * 16);
  const intr = new Float32Array(nTokens * 13);
  for (let i = 0; i < nTokens; i++) {
    const tb = i * TOKEN_STRIDE;
    const px = (tokens[tb] - cx) * invScale;
    const py = (tokens[tb + 1] - cy) * invScale;
    const pz = (tokens[tb + 2] - cz) * invScale;
    const area = Math.max(tokens[tb + 12], 0.0);
    const gb = i * 16;
    geo[gb] = px; geo[gb + 1] = py; geo[gb + 2] = pz; geo[gb + 3] = px * px + py * py + pz * pz;
    geo[gb + 4] = tokens[tb + 3]; geo[gb + 5] = tokens[tb + 4]; geo[gb + 6] = tokens[tb + 5];
    geo[gb + 7] = useAreaPrior ? Math.log(Math.max(area, 1e-4)) : 0.0;
    geo[gb + 8] = tokens[tb + 6]; geo[gb + 9] = tokens[tb + 7]; geo[gb + 10] = tokens[tb + 8];
    geo[gb + 12] = tokens[tb + 9]; geo[gb + 13] = tokens[tb + 10]; geo[gb + 14] = tokens[tb + 11];

    const ar = Math.max(0, Math.min(1, tokens[tb + 13]));
    const ag = Math.max(0, Math.min(1, tokens[tb + 14]));
    const ab = Math.max(0, Math.min(1, tokens[tb + 15]));
    const rough = Math.max(0, Math.min(1, tokens[tb + 16]));
    const metal = Math.max(0, Math.min(1, tokens[tb + 17]));
    const er = Math.max(tokens[tb + 18], 0.0);
    const eg = Math.max(tokens[tb + 19], 0.0);
    const eb = Math.max(tokens[tb + 20], 0.0);
    const isEm = Math.max(er, eg, eb) > 1e-3 ? 1.0 : 0.0;

    const ib = i * 13;
    intr[ib] = Math.log(1.0 + area);
    intr[ib + 1] = ar; intr[ib + 2] = ag; intr[ib + 3] = ab;
    intr[ib + 4] = rough; intr[ib + 5] = metal;
    intr[ib + 6] = Math.log(1.0 + er);
    intr[ib + 7] = Math.log(1.0 + eg);
    intr[ib + 8] = Math.log(1.0 + eb);
    intr[ib + 9] = isEm;
    intr[ib + 10] = Math.log(1.0 + er * area);
    intr[ib + 11] = Math.log(1.0 + eg * area);
    intr[ib + 12] = Math.log(1.0 + eb * area);
  }

  // Scratch buffers
  const xA = new Float32Array(nTokens * dModel);
  const xB = new Float32Array(nTokens * dModel);
  const xNorm = new Float32Array(nTokens * dModel);
  const ffn = new Float32Array(nTokens * dFF);
  const qSem = new Float32Array(nTokens * nHeads * 19);
  const kSem = new Float32Array(nTokens * nHeads * 19);
  const vInv = new Float32Array(nTokens * nHeads * 26);
  const geoQ = new Float32Array(nTokens * nHeads * 7);
  const geoK = new Float32Array(nTokens * nHeads * 7);
  const Q = new Float32Array(nTokens * nHeads * 32);
  const K = new Float32Array(nTokens * nHeads * 32);
  const V = new Float32Array(nTokens * nHeads * 32);
  const merged = new Float32Array(nTokens * dModel);
  const accVec = new Float32Array(32);

  // 3. Stem
  linearLayer(intr, weights, layout.stem1W * 2, layout.stem1B * 2, null, xB, nTokens, 13, dModel, true);
  linearLayer(xB, weights, layout.stem2W * 2, layout.stem2B * 2, null, xA, nTokens, dModel, dModel, false);

  const semScale = 0.22941573387056177;

  // 4. Transformer Layers (yields briefly per layer so WebGL2 browsers stay responsive!)
  for (let l = 0; l < nLayers; l++) {
    if (l > 0) {
      await new Promise((r) => setTimeout(r, 0));
    }
    const L = layout.layers[l];
    rmsNorm(xA, weights, L.norm1W * 2, xNorm, nTokens, dModel);

    linearLayer(xNorm, weights, L.qSemW * 2, L.qSemB * 2, null, qSem, nTokens, dModel, nHeads * 19, false);
    linearLayer(xNorm, weights, L.kSemW * 2, L.kSemB * 2, null, kSem, nTokens, dModel, nHeads * 19, false);
    linearLayer(xNorm, weights, L.vInvW * 2, L.vInvB * 2, null, vInv, nTokens, dModel, nHeads * 26, false);
    linearLayer(xNorm, weights, L.geoQW * 2, L.geoQB * 2, null, geoQ, nTokens, dModel, nHeads * 7, false);
    linearLayer(xNorm, weights, L.geoKW * 2, L.geoKB * 2, null, geoK, nTokens, dModel, nHeads * 7, false);

    // Build SE(3)-invariant Q, K, V
    for (let i = 0; i < nTokens; i++) {
      const gb = i * 16;
      const px = geo[gb], py = geo[gb + 1], pz = geo[gb + 2], pn2 = geo[gb + 3];
      const tx = geo[gb + 4], ty = geo[gb + 5], tz = geo[gb + 6];
      const bx = geo[gb + 8], by = geo[gb + 9], bz = geo[gb + 10];
      const nx = geo[gb + 12], ny = geo[gb + 13], nz = geo[gb + 14];

      for (let h = 0; h < nHeads; h++) {
        const ob = (i * nHeads + h) * 32;
        const sb = (i * nHeads + h) * 19;
        for (let d = 0; d < 19; d++) {
          Q[ob + d] = qSem[sb + d] * semScale;
          K[ob + d] = kSem[sb + d];
        }

        const gqb = (i * nHeads + h) * 7;
        const alpha = softplusClamped(geoQ[gqb], 20.0);
        const ux = geoQ[gqb + 1] * tx + geoQ[gqb + 2] * bx + geoQ[gqb + 3] * nx;
        const uy = geoQ[gqb + 1] * ty + geoQ[gqb + 2] * by + geoQ[gqb + 3] * ny;
        const uz = geoQ[gqb + 1] * tz + geoQ[gqb + 2] * bz + geoQ[gqb + 3] * nz;
        const mx = geoQ[gqb + 4] * tx + geoQ[gqb + 5] * bx + geoQ[gqb + 6] * nx;
        const my = geoQ[gqb + 4] * ty + geoQ[gqb + 5] * by + geoQ[gqb + 6] * ny;
        const mz = geoQ[gqb + 4] * tz + geoQ[gqb + 5] * bz + geoQ[gqb + 6] * nz;

        Q[ob + 19] = 2.0 * alpha * px + ux;
        Q[ob + 20] = 2.0 * alpha * py + uy;
        Q[ob + 21] = 2.0 * alpha * pz + uz;
        Q[ob + 22] = -alpha * pn2 - (ux * px + uy * py + uz * pz);
        Q[ob + 23] = -alpha;
        Q[ob + 24] = px; Q[ob + 25] = py; Q[ob + 26] = pz;
        Q[ob + 27] = 1.0; Q[ob + 28] = pn2;
        Q[ob + 29] = mx; Q[ob + 30] = my; Q[ob + 31] = mz;

        const gkb = (i * nHeads + h) * 7;
        const beta = softplusClamped(geoK[gkb], 20.0);
        const vx = geoK[gkb + 1] * tx + geoK[gkb + 2] * bx + geoK[gkb + 3] * nx;
        const vy = geoK[gkb + 1] * ty + geoK[gkb + 2] * by + geoK[gkb + 3] * ny;
        const vz = geoK[gkb + 1] * tz + geoK[gkb + 2] * bz + geoK[gkb + 3] * nz;
        const rx = geoK[gkb + 4] * tx + geoK[gkb + 5] * bx + geoK[gkb + 6] * nx;
        const ry = geoK[gkb + 4] * ty + geoK[gkb + 5] * by + geoK[gkb + 6] * ny;
        const rz = geoK[gkb + 4] * tz + geoK[gkb + 5] * bz + geoK[gkb + 6] * nz;

        K[ob + 19] = px; K[ob + 20] = py; K[ob + 21] = pz;
        K[ob + 22] = 1.0; K[ob + 23] = pn2;
        K[ob + 24] = 2.0 * beta * px + vx;
        K[ob + 25] = 2.0 * beta * py + vy;
        K[ob + 26] = 2.0 * beta * pz + vz;
        K[ob + 27] = -beta * pn2 - (vx * px + vy * py + vz * pz) + geo[gb + 7];
        K[ob + 28] = -beta;
        K[ob + 29] = rx; K[ob + 30] = ry; K[ob + 31] = rz;

        const vb = (i * nHeads + h) * 26;
        for (let d = 0; d < 26; d++) V[ob + d] = vInv[vb + d];
        V[ob + 26] = px; V[ob + 27] = py; V[ob + 28] = pz;
        V[ob + 29] = nx; V[ob + 30] = ny; V[ob + 31] = nz;
      }
    }

    // Online softmax SDPA per head
    for (let h = 0; h < nHeads; h++) {
      for (let i = 0; i < nTokens; i++) {
        const qb = (i * nHeads + h) * 32;
        let mMax = -1e30;
        let lSum = 0.0;
        accVec.fill(0.0);

        for (let j = 0; j < nTokens; j++) {
          const kb = (j * nHeads + h) * 32;
          let score = 0.0;
          for (let d = 0; d < 32; d += 4) {
            score +=
              Q[qb + d] * K[kb + d] +
              Q[qb + d + 1] * K[kb + d + 1] +
              Q[qb + d + 2] * K[kb + d + 2] +
              Q[qb + d + 3] * K[kb + d + 3];
          }
          if (j === 0) {
            mMax = score;
            lSum = 1.0;
            for (let d = 0; d < 32; d++) accVec[d] = V[kb + d];
          } else if (score <= mMax) {
            const diff = score - mMax;
            if (diff > -14.0) {
              const wt = Math.exp(diff);
              lSum += wt;
              for (let d = 0; d < 32; d++) accVec[d] += V[kb + d] * wt;
            }
          } else {
            const rescale = Math.exp(Math.max(mMax - score, -80.0));
            lSum = lSum * rescale + 1.0;
            for (let d = 0; d < 32; d++) accVec[d] = accVec[d] * rescale + V[kb + d];
            mMax = score;
          }
        }

        const invL = 1.0 / Math.max(lSum, 1e-12);
        const ob = i * dModel + h * 32;
        for (let d = 0; d < 26; d++) merged[ob + d] = accVec[d] * invL;

        const gb = i * 16;
        const dpx = accVec[26] * invL - geo[gb];
        const dpy = accVec[27] * invL - geo[gb + 1];
        const dpz = accVec[28] * invL - geo[gb + 2];
        const nbx = accVec[29] * invL;
        const nby = accVec[30] * invL;
        const nbz = accVec[31] * invL;
        merged[ob + 26] = dpx * geo[gb + 4] + dpy * geo[gb + 5] + dpz * geo[gb + 6];
        merged[ob + 27] = dpx * geo[gb + 8] + dpy * geo[gb + 9] + dpz * geo[gb + 10];
        merged[ob + 28] = dpx * geo[gb + 12] + dpy * geo[gb + 13] + dpz * geo[gb + 14];
        merged[ob + 29] = nbx * geo[gb + 4] + nby * geo[gb + 5] + nbz * geo[gb + 6];
        merged[ob + 30] = nbx * geo[gb + 8] + nby * geo[gb + 9] + nbz * geo[gb + 10];
        merged[ob + 31] = nbx * geo[gb + 12] + nby * geo[gb + 13] + nbz * geo[gb + 14];
      }
    }

    // Residual + FFN
    linearLayer(merged, weights, L.wOW * 2, L.wOB * 2, xA, xB, nTokens, dModel, dModel, false);
    rmsNorm(xB, weights, L.norm2W * 2, xNorm, nTokens, dModel);
    linearLayer(xNorm, weights, L.fc1W * 2, L.fc1B * 2, null, ffn, nTokens, dModel, dFF, true);
    linearLayer(ffn, weights, L.fc2W * 2, L.fc2B * 2, xB, xA, nTokens, dFF, dModel, false);
  }

  // 5. Final RMSNorm + Output Head + Physical SH Decode (12-ch single OR 36-ch [b1, b2, b3] Neumann bounces)
  const isMultiBounce = Boolean(header.isMultiBounce);
  const head2OutDim = isMultiBounce ? 36 : SH_STRIDE;
  rmsNorm(xA, weights, layout.finalNormW * 2, xNorm, nTokens, dModel);
  linearLayer(xNorm, weights, layout.head1W * 2, layout.head1B * 2, null, xB, nTokens, dModel, dModel, true);
  const raw = new Float32Array(nTokens * head2OutDim);
  linearLayer(xB, weights, layout.head2W * 2, layout.head2B * 2, null, raw, nTokens, dModel, head2OutDim, false);

  const shOut = new Float32Array(nTokens * head2OutDim);
  const nBlocks = isMultiBounce ? 3 : 1;
  for (let i = 0; i < nTokens; i++) {
    for (let blk = 0; blk < nBlocks; blk++) {
      const b = i * head2OutDim + blk * SH_STRIDE;
      const i0r = softplusClamped(raw[b], 64.0);
      const i0g = softplusClamped(raw[b + 1], 64.0);
      const i0b = softplusClamped(raw[b + 2], 64.0);
      shOut[b] = i0r;
      shOut[b + 1] = i0g;
      shOut[b + 2] = i0b;
      shOut[b + 3] = i0r * Math.tanh(Math.max(-15, Math.min(15, raw[b + 3])));
      shOut[b + 4] = i0g * Math.tanh(Math.max(-15, Math.min(15, raw[b + 4])));
      shOut[b + 5] = i0b * Math.tanh(Math.max(-15, Math.min(15, raw[b + 5])));
      shOut[b + 6] = i0r * Math.tanh(Math.max(-15, Math.min(15, raw[b + 6])));
      shOut[b + 7] = i0g * Math.tanh(Math.max(-15, Math.min(15, raw[b + 7])));
      shOut[b + 8] = i0b * Math.tanh(Math.max(-15, Math.min(15, raw[b + 8])));
      shOut[b + 9] = i0r / (1.0 + Math.exp(-Math.max(-20, Math.min(20, raw[b + 9]))));
      shOut[b + 10] = i0g / (1.0 + Math.exp(-Math.max(-20, Math.min(20, raw[b + 10]))));
      shOut[b + 11] = i0b / (1.0 + Math.exp(-Math.max(-20, Math.min(20, raw[b + 11]))));
    }
  }
  return shOut;
}

/** Combine decoded [N, 36] per-bounce SH into [N, 12] for any bounceCount in [1, 3] (instant 0ms switch!). */
export function combineBounceSH(decoded: Float32Array, nTokens: number, bounceCount: number, isMultiBounce: boolean): Float32Array {
  if (!isMultiBounce) {
    // Graceful fallback for legacy 3-bounce-only binaries: scale smoothly
    return decoded;
  }
  const w2 = Math.max(0, Math.min(1, bounceCount - 1.0));
  const w3 = Math.max(0, Math.min(1, bounceCount - 2.0));
  const out = new Float32Array(nTokens * SH_STRIDE);
  for (let i = 0; i < nTokens; i++) {
    const src = i * 36;
    const dst = i * 12;
    for (let k = 0; k < 12; k++) {
      out[dst + k] = decoded[src + k] + w2 * decoded[src + 12 + k] + w3 * decoded[src + 24 + k];
    }
  }
  return out;
}
