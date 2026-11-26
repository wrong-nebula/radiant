/**
 * Drop-In Three.js 0.187.0 `three/tsl` Node Helper: `applyRadiantGI(material, engine, objectId)`.
 *
 * Injects smooth, crease-safe directional Order-1 SH indirect bounce light into any
 * `MeshStandardNodeMaterial` or `MeshPhysicalNodeMaterial`:
 *   - Bilateral Gaussian distance + softened cos^2.5 normal-alignment gating (zero light bleed
 *     across 90-deg creases AND zero dark starvation on curved cylinders or thin edges!)
 *   - Per-pixel texture albedo support (`material.map`) & normal-mapped world normals (`normalWorld`)
 *   - Directional glossy specular indirect bounce lobe driven by predicted world-space L1 SH vectors
 *   - Live X-Ray & Split-Screen visualization modes:
 *       0 = Live RADIANT-GI (Direct + Multi-Bounce Indirect)
 *       1 = Direct Light Only (GI disabled)
 *       2 = Indirect Bounce Only (pure Order-1 SH irradiance)
 *       3 = Per-Patch Error Heatmap (vs. 1024-spp Reference)
 *       4 = Split-Screen Wipe (left = Direct Only, right = Live RADIANT-GI)
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  Loop,
  cameraPosition,
  clamp,
  dot,
  exp,
  float,
  int,
  length,
  max,
  mix,
  normalWorld,
  positionWorld,
  pow,
  reflect,
  screenUV,
  smoothstep,
  storage,
  texture,
  uniform,
  uv,
  vec3,
  vec4,
} from 'three/tsl';
import { PATCHES_PER_OBJECT } from './shMath.ts';
import {
  MAX_RENDER_PATCHES,
  RadiantComputeEngine,
  WORLD_PATCH_VEC4S,
} from './RadiantComputeEngine.ts';

export function createRadiantGIUniforms() {
  return {
    giIntensity: uniform(1.0, 'float'),
    viewMode: uniform(0.0, 'float'),
    splitX: uniform(0.5, 'float'),
  };
}

export type RadiantGIUniforms = ReturnType<typeof createRadiantGIUniforms>;

/**
 * Evaluate bilateral normal-gated Order-1 SH indirect irradiance (diffuse + directional glossy)
 * and diagnostic error for a shading point on `objectId`.
 */
export function buildRadiantGINode(
  engine: RadiantComputeEngine,
  objectId: number,
  sharedUniforms: RadiantGIUniforms,
  roughnessVal = 0.8,
  metalnessVal = 0.0,
) {
  const patchBuf = storage(
    engine.worldPatchStorage,
    'vec4',
    MAX_RENDER_PATCHES * WORLD_PATCH_VEC4S,
  ).toReadOnly();

  const patchStart = int(objectId * PATCHES_PER_OBJECT);
  const roughF = float(Math.min(1.0, Math.max(0.08, roughnessVal)));
  const metalF = float(Math.min(1.0, Math.max(0.0, metalnessVal)));

  // Plain TSL builder function (inlined cleanly inside giContribution Fn!)
  const evalBilateralSH = () => {
    const wp = positionWorld;
    const wn = normalWorld.normalize();
    const viewDir = cameraPosition.sub(wp).normalize();
    const reflDir = reflect(viewDir.negate(), wn).normalize();

    const accumDiff = vec3(0.0, 0.0, 0.0).toVar();
    const accumSpec = vec3(0.0, 0.0, 0.0).toVar();
    const accumErr = float(0.0).toVar();
    const accumW = float(0.0).toVar();

    // Roughness-controlled specular lobe sharpness (sharp on smooth chrome, broad on matte)
    const specExp = mix(float(18.0), float(2.0), roughF.mul(roughF));
    const specNorm = specExp.add(2.0).div(6.2831853);

    Loop(PATCHES_PER_OBJECT, ({ i }) => {
      const pIdx = patchStart.add(i);
      const base = pIdx.mul(WORLD_PATCH_VEC4S);

      const p0 = patchBuf.element(base.add(0)); // pos.xyz, radius
      const p1 = patchBuf.element(base.add(1)); // N.xyz, area
      const p2 = patchBuf.element(base.add(2)); // I0.rgb, IN.r
      const p3 = patchBuf.element(base.add(3)); // I1_world_R.xyz, IN.g
      const p4 = patchBuf.element(base.add(4)); // I1_world_G.xyz, IN.b
      const p5 = patchBuf.element(base.add(5)); // I1_world_B.xyz, errVal

      const pPos = p0.xyz;
      const pRad = max(p0.w, 0.16);
      const pN = p1.xyz;

      // 1. Adaptive spatial Gaussian weight
      const diff = wp.sub(pPos);
      const dist2 = dot(diff, diff);
      const sigma2 = pRad.mul(pRad).mul(3.0).add(0.06);
      const wDist = exp(dist2.negate().div(sigma2));

      // 2. Crease-safe softened normal gate: blocks 90-deg bleed while keeping curved columns & thin bevels smooth!
      const ndot = max(dot(wn, pN), 0.0);
      const wNorm = pow(max(ndot, float(1e-4)), float(2.5))
        .mul(smoothstep(float(0.0), float(0.02), ndot))
        .add(smoothstep(float(0.02), float(0.35), ndot).mul(0.025));

      // 3. Tangent-plane distance penalty (prevents light leaking across thin walls)
      const planeDist = dot(diff, pN);
      const wPlane = exp(planeDist.mul(planeDist).negate().div(0.14));

      const w = wDist.mul(wNorm).mul(wPlane).add(1e-7);

      // Exact Order-1 SH directional diffuse irradiance at shading normal wn:
      const i0 = p2.xyz;
      const inVec = vec3(p2.w, p3.w, p4.w);
      const proj1 = vec3(dot(p3.xyz, wn), dot(p4.xyz, wn), dot(p5.xyz, wn));
      const twoMinusC = float(2.0).sub(clamp(dot(wn, pN), 0.0, 1.0));
      const irrPatch = max(
        i0.mul(twoMinusC).add(proj1.sub(inVec).mul(1.5)),
        vec3(0.0, 0.0, 0.0),
      );

      // Directional glossy specular indirect bounce from world-space L1 SH dominant vectors!
      const lenR = max(length(p3.xyz), 1e-4);
      const lenG = max(length(p4.xyz), 1e-4);
      const lenB = max(length(p5.xyz), 1e-4);
      const dirR = p3.xyz.div(lenR);
      const dirG = p4.xyz.div(lenG);
      const dirB = p5.xyz.div(lenB);
      const lobeR = pow(max(dot(reflDir, dirR), float(1e-4)), specExp);
      const lobeG = pow(max(dot(reflDir, dirG), float(1e-4)), specExp);
      const lobeB = pow(max(dot(reflDir, dirB), float(1e-4)), specExp);
      const specPatch = vec3(
        i0.x.mul(lobeR),
        i0.y.mul(lobeG),
        i0.z.mul(lobeB),
      ).mul(specNorm).mul(float(0.32));

      accumDiff.addAssign(irrPatch.mul(w));
      accumSpec.addAssign(specPatch.mul(w));
      accumErr.addAssign(p5.w.mul(w));
      accumW.addAssign(w);
    });

    const invW = float(1.0).div(max(accumW, 1e-6));
    const diffIrr = accumDiff.mul(invW);
    const specIrr = accumSpec.mul(invW);
    const err = accumErr.mul(invW);

    const glossyWeight = mix(float(1.0).sub(roughF).mul(0.55), float(0.85), metalF);
    return {
      diffIrr,
      specIrr: specIrr.mul(glossyWeight),
      err,
      packed: vec4(diffIrr.x, diffIrr.y, diffIrr.z, err),
    };
  };

  // Perceptual error heatmap for X-Ray diagnostic lens
  const heatmapColor = Fn(([val_immutable]: [any]) => {
    const t = clamp(float(val_immutable).mul(4.0), 0.0, 1.0);
    const c0 = vec3(0.08, 0.14, 0.42);
    const c1 = vec3(0.10, 0.68, 0.62);
    const c2 = vec3(0.96, 0.82, 0.16);
    const c3 = vec3(0.95, 0.18, 0.16);
    const a = mix(c0, c1, clamp(t.mul(3.0), 0.0, 1.0));
    const b = mix(c1, c2, clamp(t.mul(3.0).sub(1.0), 0.0, 1.0));
    const c = mix(c2, c3, clamp(t.mul(3.0).sub(2.0), 0.0, 1.0));
    const ab = mix(a, b, clamp(t.mul(3.0).sub(0.99).mul(100.0), 0.0, 1.0));
    return mix(ab, c, clamp(t.mul(3.0).sub(1.99).mul(100.0), 0.0, 1.0));
  });

  return { evalBilateralSH, heatmapColor, sharedUniforms };
}

/**
 * Drop-in helper: attach real-time zero-bake RADIANT-GI indirect bounce lighting
 * (diffuse + directional glossy specular + per-pixel texture albedo) to any Three.js
 * `MeshStandardNodeMaterial` or `MeshPhysicalNodeMaterial`.
 */
export function applyRadiantGI(
  material: THREE.MeshStandardNodeMaterial | THREE.MeshPhysicalNodeMaterial,
  engine: RadiantComputeEngine,
  objectId: number,
  sharedUniforms: RadiantGIUniforms,
  options?: { isEmitter?: boolean },
): THREE.MeshStandardNodeMaterial | THREE.MeshPhysicalNodeMaterial {
  const roughVal = typeof material.roughness === 'number' ? material.roughness : 0.8;
  const metalVal = typeof material.metalness === 'number' ? material.metalness : 0.0;
  const { evalBilateralSH, heatmapColor } = buildRadiantGINode(
    engine,
    objectId,
    sharedUniforms,
    roughVal,
    metalVal,
  );
  const isEmitter = options?.isEmitter ?? false;

  const baseColor = material.color.clone();
  const baseAlbedoScalar = vec3(baseColor.r, baseColor.g, baseColor.b);
  const albedoMap = material.map;
  const baseEmissive = material.emissive
    ? vec3(
        material.emissive.r * material.emissiveIntensity,
        material.emissive.g * material.emissiveIntensity,
        material.emissive.b * material.emissiveIntensity,
      )
    : vec3(0.0, 0.0, 0.0);
  const metalF = float(metalVal);

  const giContribution = Fn(() => {
    const sh = evalBilateralSH();
    const indirectDiff = sh.diffIrr.mul(sharedUniforms.giIntensity);
    const indirectSpec = sh.specIrr.mul(sharedUniforms.giIntensity);
    const errVal = sh.err;

    // Support per-pixel diffuse texture maps on dropped-in .glb assets!
    const fragAlbedo = albedoMap
      ? baseAlbedoScalar.mul(texture(albedoMap, uv()).rgb)
      : baseAlbedoScalar;

    // Energy-conserving PBR diffuse + Fresnel-tinted glossy indirect bounce
    const f0 = mix(vec3(0.04, 0.04, 0.04), fragAlbedo, metalF);
    const diffuseBounce = fragAlbedo.mul(float(1.0).sub(metalF.mul(0.65))).mul(indirectDiff);
    const glossyBounce = f0.mul(indirectSpec).add(
      fragAlbedo.mul(metalF.mul(0.35)).mul(indirectDiff),
    );
    const totalBounce = diffuseBounce.add(glossyBounce);

    const mode = sharedUniforms.viewMode;
    const splitMask = clamp(screenUV.x.sub(sharedUniforms.splitX).mul(500.0).add(0.5), 0.0, 1.0);

    const isDirectOnly = clamp(float(1.0).sub(mode.sub(1.0).abs().mul(10.0)), 0.0, 1.0);
    const isIndirectOnly = clamp(float(1.0).sub(mode.sub(2.0).abs().mul(10.0)), 0.0, 1.0);
    const isHeatmap = clamp(float(1.0).sub(mode.sub(3.0).abs().mul(10.0)), 0.0, 1.0);
    const isSplit = clamp(float(1.0).sub(mode.sub(4.0).abs().mul(10.0)), 0.0, 1.0);

    let giTerm = totalBounce;
    if (isEmitter) {
      giTerm = baseEmissive;
    } else {
      const normalModeGI = baseEmissive.add(totalBounce);
      const splitGI = baseEmissive.add(totalBounce.mul(splitMask));
      const heatCol = heatmapColor(errVal);
      giTerm = normalModeGI;
      giTerm = mix(giTerm, baseEmissive, isDirectOnly);
      giTerm = mix(giTerm, indirectDiff.add(indirectSpec.mul(0.5)), isIndirectOnly);
      giTerm = mix(giTerm, heatCol, isHeatmap);
      giTerm = mix(giTerm, splitGI, isSplit);
    }

    return giTerm;
  });

  material.emissiveNode = giContribution();
  material.needsUpdate = true;
  return material;
}
