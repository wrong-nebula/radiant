/**
 * Discretization-Agnostic 32-Patch Local-Space Mesh Tokenizer for Three.js 0.187.0.
 *
 * Samples any Three.js Mesh / .glb (12 to 500,000+ triangles) ONCE in local space via
 * deterministic area-weighted Farthest-Point Sampling + normal-aware Lloyd clustering into
 * 32 surface-patch tokens. Moving or rotating the mesh simply rigid-transforms its 32 local
 * tokens in < 0.02 ms!
 */

import * as THREE from 'three/webgpu';
import { PATCHES_PER_OBJECT, TOKEN_STRIDE, buildFrisvadBasis } from './shMath.ts';

export const LOD_PATCH_COUNTS = [4, 8, 16, 32, 64] as const;
export type PatchLOD = (typeof LOD_PATCH_COUNTS)[number];
export type LODMode = 'auto' | 'hero' | 'fixed32' | 'eco';

export interface LocalPatch {
  px: number; py: number; pz: number;
  tx: number; ty: number; tz: number;
  bx: number; by: number; bz: number;
  nx: number; ny: number; nz: number;
  area: number;
  /** Optional per-patch linear RGB sampled from material.map texture (for multi-colored .glb meshes!) */
  albedo?: [number, number, number];
}

export interface LocalObjectTokens {
  objectId: number;
  mesh: THREE.Mesh;
  patches: LocalPatch[]; // active LOD patches (default 32)
  /** Precomputed Multi-LOD patch pyramid (4, 8, 16, 32, 64) for zero-cost distance switching */
  lods?: Partial<Record<PatchLOD, LocalPatch[]>>;
  activeLod: PatchLOD;
  patchStart: number;
  localRadius: number;
  albedo: [number, number, number];
  roughness: number;
  metallic: number;
  emission: [number, number, number];
  /** Optional explicit override for dynamic material editing in viewer */
  materialOverride?: {
    albedo?: [number, number, number];
    roughness?: number;
    metallic?: number;
    emission?: [number, number, number];
  };
}

interface TriData {
  cx: Float32Array; cy: Float32Array; cz: Float32Array;
  nx: Float32Array; ny: Float32Array; nz: Float32Array;
  area: Float32Array;
  u: Float32Array | null;
  v: Float32Array | null;
  count: number;
}

function rasterizeTextureMap(map: THREE.Texture | null | undefined): { data: Uint8ClampedArray; size: number } | null {
  if (!map || !map.image) return null;
  try {
    const img: any = map.image;
    const size = 64;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(img, 0, 0, size, size);
    const imgData = ctx.getImageData(0, 0, size, size);
    return { data: imgData.data, size };
  } catch {
    return null;
  }
}

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/**
 * Extract triangle centroids, unit normals, and areas from any BufferGeometry.
 * Subdivides tiny meshes (< 128 tris) and deterministically strided-subsamples giant meshes (> 8192 tris).
 */
function extractLocalTriangles(geometry: THREE.BufferGeometry): TriData {
  const posAttr = geometry.getAttribute('position');
  if (!posAttr) {
    throw new Error('sampleMeshPatches: BufferGeometry missing position attribute');
  }
  const uvAttr = geometry.getAttribute('uv');
  const idxAttr = geometry.getIndex();
  const rawTriCount = idxAttr ? Math.floor(idxAttr.count / 3) : Math.floor(posAttr.count / 3);

  // Gather raw triangle vertices (strided if > 8192 tris for instant tokenization of 500k-tri GLBs)
  const maxSampleTris = 8192;
  const stride = rawTriCount > maxSampleTris ? rawTriCount / maxSampleTris : 1;
  const initCount = rawTriCount > maxSampleTris ? maxSampleTris : rawTriCount;

  let v0x = new Float32Array(initCount), v0y = new Float32Array(initCount), v0z = new Float32Array(initCount);
  let v1x = new Float32Array(initCount), v1y = new Float32Array(initCount), v1z = new Float32Array(initCount);
  let v2x = new Float32Array(initCount), v2y = new Float32Array(initCount), v2z = new Float32Array(initCount);
  let triU: Float32Array | null = uvAttr ? new Float32Array(initCount) : null;
  let triV: Float32Array | null = uvAttr ? new Float32Array(initCount) : null;

  for (let i = 0; i < initCount; i++) {
    const t = Math.min(rawTriCount - 1, Math.floor(i * stride));
    const i0 = idxAttr ? idxAttr.getX(t * 3 + 0) : t * 3 + 0;
    const i1 = idxAttr ? idxAttr.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idxAttr ? idxAttr.getX(t * 3 + 2) : t * 3 + 2;
    v0x[i] = posAttr.getX(i0); v0y[i] = posAttr.getY(i0); v0z[i] = posAttr.getZ(i0);
    v1x[i] = posAttr.getX(i1); v1y[i] = posAttr.getY(i1); v1z[i] = posAttr.getZ(i1);
    v2x[i] = posAttr.getX(i2); v2y[i] = posAttr.getY(i2); v2z[i] = posAttr.getZ(i2);
    if (uvAttr && triU && triV) {
      triU[i] = (uvAttr.getX(i0) + uvAttr.getX(i1) + uvAttr.getX(i2)) / 3.0;
      triV[i] = (uvAttr.getY(i0) + uvAttr.getY(i1) + uvAttr.getY(i2)) / 3.0;
    }
  }

  // Subdivide low-poly meshes (like 12-tri BoxGeometry walls) so up to 64 LOD patches tile faces evenly
  let count = initCount;
  const areaScaleFactor = rawTriCount > maxSampleTris ? rawTriCount / initCount : 1.0;
  while (count < 256 && count > 0) {
    const next = count * 4;
    const n0x = new Float32Array(next), n0y = new Float32Array(next), n0z = new Float32Array(next);
    const n1x = new Float32Array(next), n1y = new Float32Array(next), n1z = new Float32Array(next);
    const n2x = new Float32Array(next), n2y = new Float32Array(next), n2z = new Float32Array(next);

    for (let i = 0; i < count; i++) {
      const ax = v0x[i], ay = v0y[i], az = v0z[i];
      const bx = v1x[i], by = v1y[i], bz = v1z[i];
      const cx = v2x[i], cy = v2y[i], cz = v2z[i];
      const m01x = 0.5 * (ax + bx), m01y = 0.5 * (ay + by), m01z = 0.5 * (az + bz);
      const m12x = 0.5 * (bx + cx), m12y = 0.5 * (by + cy), m12z = 0.5 * (bz + cz);
      const m20x = 0.5 * (cx + ax), m20y = 0.5 * (cy + ay), m20z = 0.5 * (cz + az);

      const b0 = i * 4;
      n0x[b0] = ax; n0y[b0] = ay; n0z[b0] = az;
      n1x[b0] = m01x; n1y[b0] = m01y; n1z[b0] = m01z;
      n2x[b0] = m20x; n2y[b0] = m20y; n2z[b0] = m20z;

      const b1 = b0 + 1;
      n0x[b1] = m01x; n0y[b1] = m01y; n0z[b1] = m01z;
      n1x[b1] = bx; n1y[b1] = by; n1z[b1] = bz;
      n2x[b1] = m12x; n2y[b1] = m12y; n2z[b1] = m12z;

      const b2 = b0 + 2;
      n0x[b2] = m20x; n0y[b2] = m20y; n0z[b2] = m20z;
      n1x[b2] = m12x; n1y[b2] = m12y; n1z[b2] = m12z;
      n2x[b2] = cx; n2y[b2] = cy; n2z[b2] = cz;

      const b3 = b0 + 3;
      n0x[b3] = m01x; n0y[b3] = m01y; n0z[b3] = m01z;
      n1x[b3] = m12x; n1y[b3] = m12y; n1z[b3] = m12z;
      n2x[b3] = m20x; n2y[b3] = m20y; n2z[b3] = m20z;
    }
    v0x = n0x; v0y = n0y; v0z = n0z;
    v1x = n1x; v1y = n1y; v1z = n1z;
    v2x = n2x; v2y = n2y; v2z = n2z;
    if (triU && triV) {
      const nu = new Float32Array(next);
      const nv = new Float32Array(next);
      for (let i = 0; i < count; i++) {
        nu[i * 4] = nu[i * 4 + 1] = nu[i * 4 + 2] = nu[i * 4 + 3] = triU[i];
        nv[i * 4] = nv[i * 4 + 1] = nv[i * 4 + 2] = nv[i * 4 + 3] = triV[i];
      }
      triU = nu;
      triV = nv;
    }
    count = next;
  }

  const cx = new Float32Array(count), cy = new Float32Array(count), cz = new Float32Array(count);
  const nx = new Float32Array(count), ny = new Float32Array(count), nz = new Float32Array(count);
  const area = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    cx[i] = (v0x[i] + v1x[i] + v2x[i]) / 3.0;
    cy[i] = (v0y[i] + v1y[i] + v2y[i]) / 3.0;
    cz[i] = (v0z[i] + v1z[i] + v2z[i]) / 3.0;

    const e1x = v1x[i] - v0x[i], e1y = v1y[i] - v0y[i], e1z = v1z[i] - v0z[i];
    const e2x = v2x[i] - v0x[i], e2y = v2y[i] - v0y[i], e2z = v2z[i] - v0z[i];
    const crx = e1y * e2z - e1z * e2y;
    const cry = e1z * e2x - e1x * e2z;
    const crz = e1x * e2y - e1y * e2x;
    const dbl = Math.hypot(crx, cry, crz);
    area[i] = 0.5 * dbl * areaScaleFactor;
    if (dbl > 1e-12) {
      nx[i] = crx / dbl;
      ny[i] = cry / dbl;
      nz[i] = crz / dbl;
    } else {
      nx[i] = 0; ny[i] = 1; nz[i] = 0;
    }
  }
  return { cx, cy, cz, nx, ny, nz, area, u: triU, v: triV, count };
}

/**
 * Extract linear PBR parameters from any Three.js Mesh material.
 */
export function extractMaterialProps(mesh: THREE.Mesh): {
  albedo: [number, number, number];
  roughness: number;
  metallic: number;
  emission: [number, number, number];
} {
  const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  const anyMat = mat as any;

  let albedo: [number, number, number] = [0.75, 0.75, 0.75];
  if (anyMat && anyMat.color && typeof anyMat.color.r === 'number') {
    albedo = [
      Math.min(1, Math.max(0, anyMat.color.r)),
      Math.min(1, Math.max(0, anyMat.color.g)),
      Math.min(1, Math.max(0, anyMat.color.b)),
    ];
  }
  const roughness = typeof anyMat?.roughness === 'number' ? Math.min(1, Math.max(0.05, anyMat.roughness)) : 0.8;
  const metallic = typeof anyMat?.metalness === 'number' ? Math.min(1, Math.max(0, anyMat.metalness)) : 0.0;

  let emission: [number, number, number] = [0, 0, 0];
  if (anyMat && anyMat.emissive && typeof anyMat.emissive.r === 'number') {
    const intensity = typeof anyMat.emissiveIntensity === 'number' ? anyMat.emissiveIntensity : 1.0;
    emission = [
      Math.max(0, anyMat.emissive.r * intensity),
      Math.max(0, anyMat.emissive.g * intensity),
      Math.max(0, anyMat.emissive.b * intensity),
    ];
  }
  return { albedo, roughness, metallic, emission };
}

/**
 * Deterministically cluster any Three.js Mesh into 32 local-space surface-patch tokens.
 */
function clusterTrianglesToPatches(
  tris: TriData,
  mesh: THREE.Mesh,
  matProps: ReturnType<typeof extractMaterialProps>,
  texRaster: { data: Uint8ClampedArray; size: number } | null,
  K: number,
  lloydIters: number,
  normalWeight: number,
): { patches: LocalPatch[]; localRadius: number } {
  const F = tris.count;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < F; i++) {
    if (tris.cx[i] < minX) minX = tris.cx[i];
    if (tris.cy[i] < minY) minY = tris.cy[i];
    if (tris.cz[i] < minZ) minZ = tris.cz[i];
    if (tris.cx[i] > maxX) maxX = tris.cx[i];
    if (tris.cy[i] > maxY) maxY = tris.cy[i];
    if (tris.cz[i] > maxZ) maxZ = tris.cz[i];
  }
  const diag = Math.max(Math.hypot(maxX - minX, maxY - minY, maxZ - minZ), 1e-3);
  const nw = normalWeight * diag;

  // Center features: [px, py, pz, nx*nw, ny*nw, nz*nw]
  const cPx = new Float32Array(K), cPy = new Float32Array(K), cPz = new Float32Array(K);
  const cNx = new Float32Array(K), cNy = new Float32Array(K), cNz = new Float32Array(K);

  // Face-normal stratified seeding: guarantee every distinct major normal direction
  // (e.g. thin 0.2m wall edges or cylinder back sides) gets patch coverage before FPS!
  let first = 0;
  let bestArea = -1;
  let totalArea = 0;
  for (let i = 0; i < F; i++) {
    totalArea += tris.area[i];
    if (tris.area[i] > bestArea) {
      bestArea = tris.area[i];
      first = i;
    }
  }
  cPx[0] = tris.cx[first]; cPy[0] = tris.cy[first]; cPz[0] = tris.cz[first];
  cNx[0] = tris.nx[first] * nw; cNy[0] = tris.ny[first] * nw; cNz[0] = tris.nz[first] * nw;

  const seedNormals: [number, number, number][] = [[tris.nx[first], tris.ny[first], tris.nz[first]]];
  let seededCount = 1;
  for (let s = 0; s < Math.min(6, K - 1); s++) {
    let bestOrth = -1;
    let bestOrthArea = totalArea * 0.0005;
    for (let i = 0; i < F; i++) {
      let maxDot = -1;
      for (const sn of seedNormals) {
        const d = tris.nx[i] * sn[0] + tris.ny[i] * sn[1] + tris.nz[i] * sn[2];
        if (d > maxDot) maxDot = d;
      }
      if (maxDot < 0.35 && tris.area[i] > bestOrthArea) {
        bestOrthArea = tris.area[i];
        bestOrth = i;
      }
    }
    if (bestOrth < 0) break;
    cPx[seededCount] = tris.cx[bestOrth]; cPy[seededCount] = tris.cy[bestOrth]; cPz[seededCount] = tris.cz[bestOrth];
    cNx[seededCount] = tris.nx[bestOrth] * nw; cNy[seededCount] = tris.ny[bestOrth] * nw; cNz[seededCount] = tris.nz[bestOrth] * nw;
    seedNormals.push([tris.nx[bestOrth], tris.ny[bestOrth], tris.nz[bestOrth]]);
    seededCount++;
  }

  // Area-weighted Farthest-Point Sampling for remaining patches
  const minDist2 = new Float32Array(F).fill(Infinity);
  const areaW = new Float32Array(F);
  for (let i = 0; i < F; i++) {
    areaW[i] = Math.sqrt(Math.max(tris.area[i], 1e-10));
    for (let k = 0; k < seededCount; k++) {
      const dx = tris.cx[i] - cPx[k], dy = tris.cy[i] - cPy[k], dz = tris.cz[i] - cPz[k];
      const dnx = tris.nx[i] * nw - cNx[k], dny = tris.ny[i] * nw - cNy[k], dnz = tris.nz[i] * nw - cNz[k];
      const d2 = dx * dx + dy * dy + dz * dz + dnx * dnx + dny * dny + dnz * dnz;
      if (d2 < minDist2[i]) minDist2[i] = d2;
    }
  }

  for (let k = seededCount; k < K; k++) {
    let nxt = 0;
    let bestScore = -1;
    for (let i = 0; i < F; i++) {
      const sc = minDist2[i] * areaW[i];
      if (sc > bestScore) {
        bestScore = sc;
        nxt = i;
      }
    }
    cPx[k] = tris.cx[nxt]; cPy[k] = tris.cy[nxt]; cPz[k] = tris.cz[nxt];
    cNx[k] = tris.nx[nxt] * nw; cNy[k] = tris.ny[nxt] * nw; cNz[k] = tris.nz[nxt] * nw;

    for (let i = 0; i < F; i++) {
      const dx = tris.cx[i] - cPx[k], dy = tris.cy[i] - cPy[k], dz = tris.cz[i] - cPz[k];
      const dnx = tris.nx[i] * nw - cNx[k], dny = tris.ny[i] * nw - cNy[k], dnz = tris.nz[i] * nw - cNz[k];
      const d2 = dx * dx + dy * dy + dz * dz + dnx * dnx + dny * dny + dnz * dnz;
      if (d2 < minDist2[i]) minDist2[i] = d2;
    }
  }

  // Normal-aware Lloyd iterations
  const labels = new Int32Array(F);
  const sumPx = new Float32Array(K), sumPy = new Float32Array(K), sumPz = new Float32Array(K);
  const sumNx = new Float32Array(K), sumNy = new Float32Array(K), sumNz = new Float32Array(K);
  const sumA = new Float32Array(K);

  for (let iter = 0; iter <= lloydIters; iter++) {
    for (let i = 0; i < F; i++) {
      const fx = tris.cx[i], fy = tris.cy[i], fz = tris.cz[i];
      const fnx = tris.nx[i] * nw, fny = tris.ny[i] * nw, fnz = tris.nz[i] * nw;
      let bestK = 0;
      let bestD2 = Infinity;
      for (let k = 0; k < K; k++) {
        const dx = fx - cPx[k], dy = fy - cPy[k], dz = fz - cPz[k];
        const dnx = fnx - cNx[k], dny = fny - cNy[k], dnz = fnz - cNz[k];
        const d2 = dx * dx + dy * dy + dz * dz + dnx * dnx + dny * dny + dnz * dnz;
        if (d2 < bestD2) {
          bestD2 = d2;
          bestK = k;
        }
      }
      labels[i] = bestK;
    }

    sumPx.fill(0); sumPy.fill(0); sumPz.fill(0);
    sumNx.fill(0); sumNy.fill(0); sumNz.fill(0);
    sumA.fill(0);

    for (let i = 0; i < F; i++) {
      const k = labels[i];
      const w = tris.area[i];
      sumA[k] += w;
      sumPx[k] += tris.cx[i] * w;
      sumPy[k] += tris.cy[i] * w;
      sumPz[k] += tris.cz[i] * w;
      sumNx[k] += tris.nx[i] * w;
      sumNy[k] += tris.ny[i] * w;
      sumNz[k] += tris.nz[i] * w;
    }

    if (iter < lloydIters) {
      for (let k = 0; k < K; k++) {
        if (sumA[k] > 1e-10) {
          cPx[k] = sumPx[k] / sumA[k];
          cPy[k] = sumPy[k] / sumA[k];
          cPz[k] = sumPz[k] / sumA[k];
          const nLen = Math.hypot(sumNx[k], sumNy[k], sumNz[k]) || 1.0;
          cNx[k] = (sumNx[k] / nLen) * nw;
          cNy[k] = (sumNy[k] / nLen) * nw;
          cNz[k] = (sumNz[k] / nLen) * nw;
        }
      }
    }
  }

  // Surface-snap each cluster center to its nearest member triangle centroid
  const bestSnapIdx = new Int32Array(K).fill(-1);
  const bestSnapD2 = new Float32Array(K).fill(Infinity);
  for (let k = 0; k < K; k++) {
    if (sumA[k] > 1e-10) {
      sumPx[k] /= sumA[k];
      sumPy[k] /= sumA[k];
      sumPz[k] /= sumA[k];
    }
  }
  for (let i = 0; i < F; i++) {
    const k = labels[i];
    const dx = tris.cx[i] - sumPx[k], dy = tris.cy[i] - sumPy[k], dz = tris.cz[i] - sumPz[k];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < bestSnapD2[k]) {
      bestSnapD2[k] = d2;
      bestSnapIdx[k] = i;
    }
  }

  // Fallback cluster if any cluster was empty
  let fallbackK = 0;
  for (let k = 1; k < K; k++) {
    if (sumA[k] > sumA[fallbackK]) fallbackK = k;
  }

  // Optional per-patch albedo sampling from material.map texture (so textured .glb meshes bounce true local colors!)
  void mesh;
  let patchTexR: Float32Array | null = null;
  let patchTexG: Float32Array | null = null;
  let patchTexB: Float32Array | null = null;
  if (texRaster && tris.u && tris.v) {
    patchTexR = new Float32Array(K);
    patchTexG = new Float32Array(K);
    patchTexB = new Float32Array(K);
    const sz = texRaster.size;
    for (let i = 0; i < F; i++) {
      const k = labels[i];
      const w = tris.area[i];
      const uu = ((tris.u[i] % 1) + 1) % 1;
      const vv = ((1.0 - (tris.v[i] % 1)) + 1) % 1;
      const px = Math.min(sz - 1, Math.floor(uu * sz));
      const py = Math.min(sz - 1, Math.floor(vv * sz));
      const idx4 = (py * sz + px) * 4;
      patchTexR[k] += srgbToLinear(texRaster.data[idx4] / 255.0) * w;
      patchTexG[k] += srgbToLinear(texRaster.data[idx4 + 1] / 255.0) * w;
      patchTexB[k] += srgbToLinear(texRaster.data[idx4 + 2] / 255.0) * w;
    }
  }

  const patches: LocalPatch[] = [];
  for (let k = 0; k < K; k++) {
    const useK = sumA[k] > 1e-10 ? k : fallbackK;
    const snap = bestSnapIdx[useK] >= 0 ? bestSnapIdx[useK] : 0;
    const px = tris.cx[snap];
    const py = tris.cy[snap];
    const pz = tris.cz[snap];
    let nx = sumNx[useK], ny = sumNy[useK], nz = sumNz[useK];
    if (Math.hypot(nx, ny, nz) < 1e-4) {
      nx = tris.nx[snap]; ny = tris.ny[snap]; nz = tris.nz[snap];
    }
    const basis = buildFrisvadBasis(nx, ny, nz);
    let patchAlb: [number, number, number] | undefined = undefined;
    if (patchTexR && patchTexG && patchTexB && sumA[useK] > 1e-10) {
      patchAlb = [
        Math.min(1, (patchTexR[useK] / sumA[useK]) * matProps.albedo[0]),
        Math.min(1, (patchTexG[useK] / sumA[useK]) * matProps.albedo[1]),
        Math.min(1, (patchTexB[useK] / sumA[useK]) * matProps.albedo[2]),
      ];
    }
    patches.push({
      px, py, pz,
      tx: basis.tx, ty: basis.ty, tz: basis.tz,
      bx: basis.bx, by: basis.by, bz: basis.bz,
      nx: basis.nx, ny: basis.ny, nz: basis.nz,
      area: sumA[k] > 1e-10 ? sumA[k] : 1e-4,
      albedo: patchAlb,
    });
  }

  return { patches, localRadius: Math.max(0.5 * diag, 0.15) };
}

/**
 * Deterministically cluster any Three.js Mesh into a single fixed patch count (default 32).
 */
export function sampleMeshPatches(
  mesh: THREE.Mesh,
  objectId: number,
  nPatches: PatchLOD = PATCHES_PER_OBJECT as PatchLOD,
  lloydIters: number = 6,
  normalWeight: number = 0.45,
): LocalObjectTokens {
  const tris = extractLocalTriangles(mesh.geometry);
  const matProps = extractMaterialProps(mesh);
  const rawMat: any = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  const texRaster = tris.u && tris.v ? rasterizeTextureMap(rawMat?.map) : null;
  const { patches, localRadius } = clusterTrianglesToPatches(
    tris, mesh, matProps, texRaster, nPatches, lloydIters, normalWeight,
  );
  return {
    objectId,
    mesh,
    patches,
    lods: { [nPatches]: patches },
    activeLod: nPatches,
    patchStart: objectId * nPatches,
    localRadius,
    ...matProps,
  };
}

/**
 * Precompute the full Distance-LOD patch pyramid (4, 8, 16, 32, 64) ONCE per mesh in local space
 * so switching LODs while orbiting the camera takes 0 ms with zero geometry re-clustering!
 */
export function sampleMeshMultiLOD(
  mesh: THREE.Mesh,
  objectId: number,
  lloydIters: number = 5,
  normalWeight: number = 0.45,
): LocalObjectTokens {
  const tris = extractLocalTriangles(mesh.geometry);
  const matProps = extractMaterialProps(mesh);
  const rawMat: any = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  const texRaster = tris.u && tris.v ? rasterizeTextureMap(rawMat?.map) : null;
  const lods: Partial<Record<PatchLOD, LocalPatch[]>> = {};
  let localRadius = 0.5;
  for (const k of LOD_PATCH_COUNTS) {
    const res = clusterTrianglesToPatches(tris, mesh, matProps, texRaster, k, lloydIters, normalWeight);
    lods[k] = res.patches;
    localRadius = res.localRadius;
  }
  const patches = lods[32]!;
  return {
    objectId,
    mesh,
    patches,
    lods,
    activeLod: 32,
    patchStart: objectId * 32,
    localRadius,
    ...matProps,
  };
}

const _lodCenter = new THREE.Vector3();
const _lodScale = new THREE.Vector3();

/**
 * Select distance/screen-coverage Patch LOD (4, 8, 16, 32, 64) with 15% hysteresis band
 * so objects never flicker between LODs as the user orbits the camera.
 */
export function selectObjectLOD(
  obj: LocalObjectTokens,
  camera: THREE.Camera,
  mode: LODMode,
  isEmitter = false,
): PatchLOD {
  if (mode === 'fixed32') return 32;

  obj.mesh.updateWorldMatrix(true, false);
  _lodCenter.setFromMatrixPosition(obj.mesh.matrixWorld);
  _lodScale.setFromMatrixScale(obj.mesh.matrixWorld);
  const maxS = Math.max(Math.abs(_lodScale.x), Math.abs(_lodScale.y), Math.abs(_lodScale.z), 1e-4);
  const worldRad = obj.localRadius * maxS;
  const dist = Math.max(_lodCenter.distanceTo(camera.position), 0.35);
  const cov = worldRad / dist;

  const prev = obj.activeLod;
  if (mode === 'hero') {
    if (isEmitter) return cov >= 0.25 ? 32 : 16;
    if (cov >= (prev === 64 ? 0.42 : 0.50)) return 64;
    if (cov >= (prev >= 32 ? 0.18 : 0.22)) return 32;
    return 16;
  }
  if (mode === 'eco') {
    if (isEmitter) return cov >= 0.22 ? 16 : 8;
    if (cov >= (prev >= 16 ? 0.30 : 0.36)) return 16;
    if (cov >= (prev >= 8 ? 0.14 : 0.18)) return 8;
    return 4;
  }
  // mode === 'auto': Distance-adaptive LOD (64 very close hero -> 32 near -> 16 mid -> 8 far -> 4 distant)
  if (isEmitter) {
    if (cov >= (prev >= 32 ? 0.28 : 0.34)) return 32;
    if (cov >= (prev >= 16 ? 0.13 : 0.16)) return 16;
    return 8;
  }
  if (cov >= (prev === 64 ? 0.82 : 0.96)) return 64;
  if (cov >= (prev >= 32 ? 0.36 : 0.43)) return 32;
  if (cov >= (prev >= 16 ? 0.19 : 0.23)) return 16;
  if (cov >= (prev >= 8 ? 0.095 : 0.115)) return 8;
  return 4;
}

const _tmpPos = new THREE.Vector3();
const _tmpN = new THREE.Vector3();
const _tmpT = new THREE.Vector3();
const _tmpB = new THREE.Vector3();
const _tmpScale = new THREE.Vector3();
const _tmpQuat = new THREE.Quaternion();
const _tmpTrans = new THREE.Vector3();
const _normalMat = new THREE.Matrix3();

/**
 * Rigid/affine-transform an object's 32 local-space tokens into the packed world token array.
 */
export function writeWorldTokens(
  obj: LocalObjectTokens,
  outTokens: Float32Array,
  tokenStartIdx: number,
): void {
  obj.mesh.updateWorldMatrix(true, false);
  const mw = obj.mesh.matrixWorld;
  mw.decompose(_tmpTrans, _tmpQuat, _tmpScale);
  _normalMat.getNormalMatrix(mw);

  const sx = Math.abs(_tmpScale.x);
  const sy = Math.abs(_tmpScale.y);
  const sz = Math.abs(_tmpScale.z);

  const matLive = extractMaterialProps(obj.mesh);
  const alb = obj.materialOverride?.albedo ?? matLive.albedo;
  const rough = obj.materialOverride?.roughness ?? matLive.roughness;
  const metal = obj.materialOverride?.metallic ?? matLive.metallic;
  const emit = obj.materialOverride?.emission ?? matLive.emission;

  const activePatches = obj.lods?.[obj.activeLod] ?? obj.patches;
  for (let i = 0; i < activePatches.length; i++) {
    const p = activePatches[i];
    _tmpPos.set(p.px, p.py, p.pz).applyMatrix4(mw);

    // Transform normal by inverse-transpose (exact for both rigid & non-uniform scale)
    _tmpN.set(p.nx, p.ny, p.nz).applyMatrix3(_normalMat).normalize();

    // Transform tangent & re-orthogonalize rigidly to maintain exact SE(3) invariance!
    _tmpT.set(p.tx, p.ty, p.tz).transformDirection(mw);
    // Gram-Schmidt orthogonalize T against N
    const dotTN = _tmpT.dot(_tmpN);
    _tmpT.addScaledVector(_tmpN, -dotTN);
    if (_tmpT.lengthSq() < 1e-8) {
      const fb = buildFrisvadBasis(_tmpN.x, _tmpN.y, _tmpN.z);
      _tmpT.set(fb.tx, fb.ty, fb.tz);
      _tmpB.set(fb.bx, fb.by, fb.bz);
    } else {
      _tmpT.normalize();
      _tmpB.crossVectors(_tmpN, _tmpT).normalize();
    }

    // Exact area scaling under axis scale (sx, sy, sz) for local normal (nx, ny, nz)
    const aScale = Math.hypot(p.nx * sy * sz, p.ny * sx * sz, p.nz * sx * sy);
    const worldArea = p.area * Math.max(aScale, 1e-6);

    const off = (tokenStartIdx + i) * TOKEN_STRIDE;
    outTokens[off + 0] = _tmpPos.x;
    outTokens[off + 1] = _tmpPos.y;
    outTokens[off + 2] = _tmpPos.z;
    outTokens[off + 3] = _tmpT.x;
    outTokens[off + 4] = _tmpT.y;
    outTokens[off + 5] = _tmpT.z;
    outTokens[off + 6] = _tmpB.x;
    outTokens[off + 7] = _tmpB.y;
    outTokens[off + 8] = _tmpB.z;
    outTokens[off + 9] = _tmpN.x;
    outTokens[off + 10] = _tmpN.y;
    outTokens[off + 11] = _tmpN.z;
    const pAlb = obj.materialOverride?.albedo ?? p.albedo ?? alb;
    outTokens[off + 12] = worldArea;
    outTokens[off + 13] = pAlb[0];
    outTokens[off + 14] = pAlb[1];
    outTokens[off + 15] = pAlb[2];
    outTokens[off + 16] = rough;
    outTokens[off + 17] = metal;
    outTokens[off + 18] = emit[0];
    outTokens[off + 19] = emit[1];
    outTokens[off + 20] = emit[2];
  }
}
