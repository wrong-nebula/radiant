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
  ivec2,
  length,
  max,
  mix,
  normalWorld,
  positionWorld,
  pow,
  reflect,
  screenUV,
  smoothstep,
  sqrt,
  storage,
  texture,
  textureLoad,
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
  const useTextureFallback = engine.useTextureLookup;
  const patchBuf = useTextureFallback
    ? null
    : storage(
        engine.worldPatchStorage,
        'vec4',
        MAX_RENDER_PATCHES * WORLD_PATCH_VEC4S,
      ).toReadOnly();
  const patchTex = engine.worldPatchTexture;
  // Use uniforms (not compile-time constants!) so all objects in a room share ONE compiled shader program!
  const objRow = uniform(objectId, 'int');
  const patchStart = uniform(objectId * PATCHES_PER_OBJECT, 'int');
  const roughF = uniform(Math.min(1.0, Math.max(0.08, roughnessVal)), 'float');
  const metalF = uniform(Math.min(1.0, Math.max(0.0, metalnessVal)), 'float');

  // Plain TSL builder function (inlined cleanly inside giContribution Fn!)
  const evalBilateralSH = () => {
    const wp = positionWorld;
    const wn = normalWorld.normalize();
    const viewDir = cameraPosition.sub(wp).normalize();
    const reflDir = reflect(viewDir.negate(), wn).normalize();

    const accumDiff = vec3(0.0, 0.0, 0.0).toVar();
    const accumI0 = vec3(0.0, 0.0, 0.0).toVar();
    const accumI1R = vec3(0.0, 0.0, 0.0).toVar();
    const accumI1G = vec3(0.0, 0.0, 0.0).toVar();
    const accumI1B = vec3(0.0, 0.0, 0.0).toVar();
    const accumErr = float(0.0).toVar();
    const accumW = float(0.0).toVar();

    // Roughness-controlled specular lobe sharpness (sharp on smooth chrome, broad on matte)
    const specExp = mix(float(18.0), float(2.0), roughF.mul(roughF));
    const specNorm = specExp.add(2.0).div(6.2831853);

    Loop(PATCHES_PER_OBJECT, ({ i }) => {
      let p0: any, p1: any, p2: any, p3: any, p4: any, p5: any;
      if (useTextureFallback) {
        const colBase = i.mul(WORLD_PATCH_VEC4S);
        p0 = textureLoad(patchTex, ivec2(colBase.add(0), objRow));
        p1 = textureLoad(patchTex, ivec2(colBase.add(1), objRow));
        p2 = textureLoad(patchTex, ivec2(colBase.add(2), objRow));
        p3 = textureLoad(patchTex, ivec2(colBase.add(3), objRow));
        p4 = textureLoad(patchTex, ivec2(colBase.add(4), objRow));
        p5 = textureLoad(patchTex, ivec2(colBase.add(5), objRow));
      } else {
        const pIdx = patchStart.add(i);
        const base = pIdx.mul(WORLD_PATCH_VEC4S);
        p0 = patchBuf!.element(base.add(0)); // pos.xyz, radius
        p1 = patchBuf!.element(base.add(1)); // N.xyz, area
        p2 = patchBuf!.element(base.add(2)); // I0.rgb, IN.r
        p3 = patchBuf!.element(base.add(3)); // I1_world_R.xyz, IN.g
        p4 = patchBuf!.element(base.add(4)); // I1_world_G.xyz, IN.b
        p5 = patchBuf!.element(base.add(5)); // I1_world_B.xyz, errVal
      }

      const pPos = p0.xyz;
      const pRad = max(p0.w, 0.16);
      const pN = p1.xyz;

      // 1. Fused spatial Gaussian + tangent-plane penalty: exp(-a) * exp(-b) == exp(-(a + b)) (half the exp() ops!)
      const diff = wp.sub(pPos);
      const dist2 = dot(diff, diff);
      const sigma2 = pRad.mul(pRad).mul(3.0).add(0.06);
      const planeDist = dot(diff, pN);
      const wSpatial = exp(dist2.div(sigma2).add(planeDist.mul(planeDist).div(0.14)).negate());

      // 2. Crease-safe softened normal gate: ndot^2.5 == ndot * ndot * sqrt(ndot) (zero pow() inside loop!)
      const ndot = max(dot(wn, pN), 0.0);
      const wNorm = ndot.mul(ndot).mul(sqrt(ndot)).add(smoothstep(float(0.02), float(0.35), ndot).mul(0.025));

      const w = wSpatial.mul(wNorm).add(1e-7);

      // Exact Order-1 SH directional diffuse irradiance at shading normal wn:
      const i0 = p2.xyz;
      const inVec = vec3(p2.w, p3.w, p4.w);
      const proj1 = vec3(dot(p3.xyz, wn), dot(p4.xyz, wn), dot(p5.xyz, wn));
      const twoMinusC = float(2.0).sub(clamp(ndot, 0.0, 1.0));
      const irrPatch = max(
        i0.mul(twoMinusC).add(proj1.sub(inVec).mul(1.5)),
        vec3(0.0, 0.0, 0.0),
      );

      accumDiff.addAssign(irrPatch.mul(w));
      accumI0.addAssign(i0.mul(w));
      accumI1R.addAssign(p3.xyz.mul(w));
      accumI1G.addAssign(p4.xyz.mul(w));
      accumI1B.addAssign(p5.xyz.mul(w));
      accumErr.addAssign(p5.w.mul(w));
      accumW.addAssign(w);
    });

    const invW = float(1.0).div(max(accumW, 1e-6));
    const diffIrr = accumDiff.mul(invW);
    const meanI0 = accumI0.mul(invW);
    const err = accumErr.mul(invW);

    // Evaluate directional glossy specular lobe ONCE per pixel (32x fewer pow()/sqrt() ops!)
    const dirR = accumI1R.div(max(length(accumI1R), 1e-4));
    const dirG = accumI1G.div(max(length(accumI1G), 1e-4));
    const dirB = accumI1B.div(max(length(accumI1B), 1e-4));
    const lobeR = pow(max(dot(reflDir, dirR), float(1e-4)), specExp);
    const lobeG = pow(max(dot(reflDir, dirG), float(1e-4)), specExp);
    const lobeB = pow(max(dot(reflDir, dirB), float(1e-4)), specExp);
    const specIrr = vec3(
      meanI0.x.mul(lobeR),
      meanI0.y.mul(lobeG),
      meanI0.z.mul(lobeB),
    ).mul(specNorm).mul(float(0.32));

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
  const baseAlbedoScalar = uniform(new THREE.Vector3(baseColor.r, baseColor.g, baseColor.b));
  const albedoMap = material.map;
  const baseEmissive = uniform(
    material.emissive
      ? new THREE.Vector3(
          material.emissive.r * material.emissiveIntensity,
          material.emissive.g * material.emissiveIntensity,
          material.emissive.b * material.emissiveIntensity,
        )
      : new THREE.Vector3(0.0, 0.0, 0.0),
  );
  const metalF = uniform(metalVal, 'float');

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
