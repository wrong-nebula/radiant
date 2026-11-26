/**
 * Exact Order-1 Spherical Harmonics math & Frisvad local tangent frame utilities.
 * Matches `radiant/sh_math.py` bit-for-bit.
 */

export const POS_SCALE_METERS = 4.0;
export const PATCHES_PER_OBJECT = 32;
export const TOKEN_STRIDE = 21; // pos(3), T(3), B(3), N(3), area(1), albedo(3), rough(1), metal(1), emit(3)
export const SH_STRIDE = 12;    // I0(3), IT(3), IB(3), IN(3)

export interface Basis3 {
  tx: number; ty: number; tz: number;
  bx: number; by: number; bz: number;
  nx: number; ny: number; nz: number;
}

/**
 * Duff/Frisvad branchless orthonormal tangent frame (T, B, N) from normal (nx, ny, nz).
 */
export function buildFrisvadBasis(nxIn: number, nyIn: number, nzIn: number): Basis3 {
  const nLen = Math.hypot(nxIn, nyIn, nzIn) || 1.0;
  const nx = nxIn / nLen;
  const ny = nyIn / nLen;
  const nz = nzIn / nLen;

  const sign = nz >= 0.0 ? 1.0 : -1.0;
  const a = -1.0 / (sign + nz);
  const b = nx * ny * a;

  let tx = 1.0 + sign * nx * nx * a;
  let ty = sign * b;
  let tz = -sign * nx;
  const tLen = Math.hypot(tx, ty, tz) || 1.0;
  tx /= tLen;
  ty /= tLen;
  tz /= tLen;

  // B = normalize(N x T)
  let bx = ny * tz - nz * ty;
  let by = nz * tx - nx * tz;
  let bz = nx * ty - ny * tx;
  const bLen = Math.hypot(bx, by, bz) || 1.0;
  bx /= bLen;
  by /= bLen;
  bz /= bLen;

  return { tx, ty, tz, bx, by, bz, nx, ny, nz };
}

export function tonemapReinhardSRGB(x: number): number {
  const v = Math.max(0.0, x);
  const tm = Math.min(1.0, Math.max(0.0, v / (1.0 + v)));
  return Math.pow(tm, 1.0 / 2.2);
}

export function computeSHRMSE(pred: Float32Array | number[], ref: Float32Array | number[]): number {
  const n = Math.min(pred.length, ref.length);
  if (n === 0) return 0;
  let sum2 = 0;
  for (let i = 0; i < n; i++) {
    const d = pred[i] - ref[i];
    sum2 += d * d;
  }
  return Math.sqrt(sum2 / n);
}

export function computeTonemappedPSNR(predSH: Float32Array | number[], gtSH: Float32Array | number[]): number {
  const nTokens = Math.floor(Math.min(predSH.length, gtSH.length) / SH_STRIDE);
  if (nTokens === 0) return 0;
  let mse = 0;
  for (let i = 0; i < nTokens; i++) {
    const off = i * SH_STRIDE;
    for (let c = 0; c < 3; c++) {
      const d = tonemapReinhardSRGB(predSH[off + c]) - tonemapReinhardSRGB(gtSH[off + c]);
      mse += d * d;
    }
  }
  mse /= nTokens * 3;
  if (mse <= 1e-12) return 60.0;
  return 10.0 * Math.log10(1.0 / mse);
}
