/**
 * THREE-RADIANT-GI Live 60 fps Three.js 0.187.0 WebGPU Interactive Showcase.
 *
 * Features:
 *   - Pure `three/webgpu` + `three/tsl` renderer (`await renderer.init()`, sync `renderer.render()`)
 *   - Native Apple M1 Metal 3 WGSL Compute Transformer (`RadiantComputeEngine`)
 *   - 4 unseen playable rooms + Held-Out 1024-spp Reference Benchmark room
 *   - Drag-and-drop `.glb` mesh support (instant 32-patch sampling + live GI)
 *   - Live TransformControls dragging & bounce color tinting at 60 fps (0 ms when static!)
 *   - Split-Screen Wipe, Indirect Only, 1024-spp Reference, Error Heatmap, Patch Spheres & SH Probes
 *   - Clean `window.__RADIANT_APP__` audit API for Playwright O4/O5 verification
 */

import './styles.css';
import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

import {
  LocalObjectTokens,
  PATCHES_PER_OBJECT,
  RadiantComputeEngine,
  SH_STRIDE,
  TOKEN_STRIDE,
  applyRadiantGI,
  computeSHRMSE,
  computeTonemappedPSNR,
  createRadiantGIUniforms,
  sampleMeshPatches,
  writeWorldTokens,
} from '../radiant/index.ts';
import {
  RoomBuildResult,
  RoomObjectSpec,
  buildCornellRoom,
  buildHeldoutBenchmarkRoom,
  buildLCorridorRoom,
  buildStudioShowcaseRoom,
  buildVaultedGalleryRoom,
} from './rooms.ts';

const BASE_URL: string = (import.meta as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? './';
const WEIGHTS_URL = `${BASE_URL}artifacts/radiant_gi_fp16.bin`;
const PROBES_URL = `${BASE_URL}data/parity_probes.json`;

class RadiantShowcaseApp {
  private canvas: HTMLCanvasElement;
  private renderer!: THREE.WebGPURenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private orbit!: OrbitControls;
  private transform!: TransformControls;
  private engine!: RadiantComputeEngine;
  private giUniforms = createRadiantGIUniforms();
  private raycaster = new THREE.Raycaster();
  private gltfLoader = new GLTFLoader();

  private roomGroup = new THREE.Group();
  private overlayGroup = new THREE.Group();
  private patchSpheresMesh: THREE.InstancedMesh | null = null;
  private shProbeLines: THREE.LineSegments | null = null;

  private roomObjects: RoomObjectSpec[] = [];
  private localTokens: LocalObjectTokens[] = [];
  private roomLights: THREE.Light[] = [];
  private roomLightBaseIntensity: number[] = [];
  private worldTokens = new Float32Array(0);

  private currentRoomIdx = 0;
  private currentViewMode = 0; // 0=FullGI, 1=Direct, 2=Indirect, 3=Heatmap, 4=Split, 5=Reference1024
  private showPatches = false;
  private showSHProbes = false;
  private selectedSpec: RoomObjectSpec | null = null;

  private isBenchmarkRoom = false;
  private benchmarkTokens: Float32Array | null = null;
  private benchmarkGtSH: Float32Array | null = null;
  private lastPredSH: Float32Array = new Float32Array(0);

  private giDirty = false;
  private computeInFlight = false;
  private roomGeneration = 0;
  private frameCount = 0;
  private fpsAccumTime = 0;
  private lastFrameTs = performance.now();
  private currentFps = 60;
  private isDraggingGizmo = false;

  public ready: Promise<void>;
  private resolveReady!: () => void;

  constructor() {
    this.canvas = document.getElementById('viewport') as HTMLCanvasElement;
    this.ready = new Promise((res) => {
      this.resolveReady = res;
    });
    this._init().catch((err) => {
      console.error('[THREE-RADIANT-GI] Init error:', err);
      this._showToast(`Init error: ${String(err)}`);
    });
  }

  private async _init(): Promise<void> {
    // Firefox macOS WebGPU (wgpu) still has swap-chain/TSL blank-canvas bugs in Three.js WebGPURenderer;
    // render via Firefox's rock-solid ANGLE Metal WebGL2 backend while still using WebGPU Compute (or CPU fallback)!
    const isFirefox = typeof navigator !== 'undefined' && /Firefox\//i.test(navigator.userAgent);
    this.renderer = new THREE.WebGPURenderer({
      canvas: this.canvas,
      antialias: true,
      powerPreference: 'high-performance',
      forceWebGL: isFirefox,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    await this.renderer.init();

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0c0e13);
    this.scene.add(this.roomGroup);
    this.scene.add(this.overlayGroup);

    // Subtle ambient fill (very low so multi-bounce GI is unmistakably dominant!)
    const amb = new THREE.AmbientLight(0xffffff, 0.03);
    this.scene.add(amb);

    this.camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.05, 80);
    this.orbit = new OrbitControls(this.camera, this.canvas);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.08;

    this.transform = new TransformControls(this.camera, this.canvas);
    this.transform.addEventListener('dragging-changed', (e: any) => {
      this.orbit.enabled = !e.value;
      this.isDraggingGizmo = Boolean(e.value);
    });
    this.transform.addEventListener('objectChange', () => {
      this._syncSelectedLight();
      this.giDirty = true;
    });
    this.scene.add(this.transform.getHelper());

    // Create WebGPU Compute Engine immediately so Room 1 renders on Frame 1 (zero blank screen!)
    this.engine = await RadiantComputeEngine.createUnloadedFromRenderer(this.renderer);
    this._wireUI();
    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });

    // Show Cornell Box immediately with direct light while FP16 weights stream in
    await this.loadRoom(0);
    this._animate();

    const elSt = document.getElementById('m-state');
    const toastEl = document.getElementById('toast');
    if (elSt) elSt.textContent = 'LOADING';
    if (toastEl) {
      toastEl.textContent = 'Loading RADIANT FP16 weights (13.2 MB)...';
      toastEl.classList.remove('hidden');
    }

    await this.engine.loadWeightsFromUrl(WEIGHTS_URL, (loaded, total) => {
      const mbLoaded = (loaded / (1024 * 1024)).toFixed(1);
      const mbTotal = total > 0 ? (total / (1024 * 1024)).toFixed(1) : '13.2';
      if (toastEl) {
        toastEl.textContent = `Loading RADIANT FP16 weights (${mbLoaded} / ${mbTotal} MB)...`;
        toastEl.classList.remove('hidden');
      }
    });

    const badge = document.getElementById('gpu-badge');
    if (badge) {
      if (this.engine.isCPUFallback) {
        badge.textContent = `WebGL2 · ${this.engine.header.nLayers}L CPU GI`;
      } else if (this.engine.useTextureLookup) {
        badge.textContent = `WebGL2 + WebGPU · ${this.engine.header.nLayers}L`;
      } else {
        badge.textContent = `WebGPU · ${this.engine.header.nLayers}L FP16`;
      }
    }

    // Compute live 3-bounce indirect GI the instant weights finish downloading!
    await this.recomputeGI();
    this._showToast(
      this.engine.isCPUFallback
        ? 'WebGL2 + CPU GI Ready (open in Chrome/Safari for GPU WebGPU Compute)'
        : 'RADIANT-GI Ready — drag any wall, lamp, or sculpture live!',
    );
    this.resolveReady();
  }

  private _syncSelectedLight(): void {
    if (!this.selectedSpec?.linkedLight || !this.selectedSpec.lightOffset) return;
    this.selectedSpec.linkedLight.position
      .copy(this.selectedSpec.mesh.position)
      .add(this.selectedSpec.lightOffset);
  }

  public async loadRoom(roomIdx: number): Promise<void> {
    this.roomGeneration++;
    this.giDirty = false;
    this.currentRoomIdx = roomIdx;
    this._deselect();

    // Dispose old room meshes & lights
    while (this.roomGroup.children.length > 0) {
      const ch = this.roomGroup.children[0];
      this.roomGroup.remove(ch);
    }
    this._clearOverlays();
    this.roomObjects = [];
    this.localTokens = [];
    this.roomLights = [];
    this.roomLightBaseIntensity = [];
    this.isBenchmarkRoom = false;
    this.benchmarkTokens = null;
    this.benchmarkGtSH = null;

    let built: RoomBuildResult;
    if (roomIdx === 0) built = buildCornellRoom();
    else if (roomIdx === 1) built = buildVaultedGalleryRoom();
    else if (roomIdx === 2) built = buildLCorridorRoom();
    else if (roomIdx === 3) built = buildStudioShowcaseRoom();
    else built = await buildHeldoutBenchmarkRoom(PROBES_URL);

    this.isBenchmarkRoom = Boolean(built.isBenchmarkProbe);
    this.benchmarkTokens = built.benchmarkTokens ?? null;
    this.benchmarkGtSH = built.benchmarkGtSH ?? null;

    this.camera.position.copy(built.cameraPos);
    this.orbit.target.copy(built.cameraTarget);
    this.orbit.update();

    // Register lights
    for (const l of built.lights) {
      this.roomGroup.add(l);
      this.roomLights.push(l);
      this.roomLightBaseIntensity.push(l.intensity);
    }

    // Tokenize every object into 32 local patches & attach drop-in `applyRadiantGI` TSL node
    for (let i = 0; i < built.objects.length; i++) {
      const spec = built.objects[i];
      const mat = spec.mesh.material as THREE.MeshStandardNodeMaterial | THREE.MeshPhysicalNodeMaterial;
      applyRadiantGI(mat, this.engine, i, this.giUniforms, { isEmitter: spec.isEmitter });
      this.roomGroup.add(spec.mesh);
      this.roomObjects.push(spec);
      this.localTokens.push(sampleMeshPatches(spec.mesh, i, PATCHES_PER_OBJECT));
    }

    this._updateRoomButtons(roomIdx);
    this.setViewMode(this.currentViewMode === 5 && !this.isBenchmarkRoom ? 0 : this.currentViewMode);
    await this.recomputeGI();
    this._showToast(`Loaded: ${built.title}`);
  }

  public async recomputeGI(): Promise<{ computeMs: number; nTokens: number }> {
    const gen = this.roomGeneration;
    let tokensToRun: Float32Array;
    if (this.isBenchmarkRoom && this.benchmarkTokens) {
      tokensToRun = this.benchmarkTokens;
    } else {
      const nTokens = this.localTokens.length * PATCHES_PER_OBJECT;
      if (this.worldTokens.length !== nTokens * TOKEN_STRIDE) {
        this.worldTokens = new Float32Array(nTokens * TOKEN_STRIDE);
      }
      for (let i = 0; i < this.localTokens.length; i++) {
        writeWorldTokens(this.localTokens[i], this.worldTokens, i * PATCHES_PER_OBJECT);
      }
      tokensToRun = this.worldTokens;
    }

    const nTokens = Math.floor(tokensToRun.length / TOKEN_STRIDE);
    const predSH = await this.engine.inferTokens(tokensToRun, false);
    if (gen !== this.roomGeneration) {
      return { computeMs: this.engine.lastComputeMs, nTokens };
    }
    this.lastPredSH = predSH;

    const displaySH =
      this.currentViewMode === 5 && this.benchmarkGtSH ? this.benchmarkGtSH : predSH;
    this.engine.uploadWorldSHToTSL(tokensToRun, displaySH, nTokens, this.benchmarkGtSH);

    this._rebuildOverlays(tokensToRun, displaySH, nTokens);
    this._updateMetricsHUD(this.engine.lastComputeMs, nTokens, 'COMPUTED');

    return { computeMs: this.engine.lastComputeMs, nTokens };
  }

  public setViewMode(mode: number): void {
    if (mode === 5 && !this.benchmarkGtSH) {
      this._showToast('Switch to Room 5 (Held-Out Benchmark) to view 1024-spp Ground Truth Reference');
      return;
    }
    this.currentViewMode = mode;

    // Mode 5 displays exact 1024-spp GT in Full-GI shading (TSL mode 0)
    const shaderMode = mode === 5 ? 0 : mode;
    this.giUniforms.viewMode.value = shaderMode;

    // Dim direct Three.js lights in Indirect-Only (2) and Error-Heatmap (3) modes
    const directScale = mode === 2 || mode === 3 ? 0.0 : 1.0;
    for (let i = 0; i < this.roomLights.length; i++) {
      this.roomLights[i].intensity = this.roomLightBaseIntensity[i] * directScale;
    }

    // Re-upload SH buffer if switching to/from 1024-spp GT Reference
    if (this.lastPredSH.length > 0) {
      const tokensSrc = this.isBenchmarkRoom && this.benchmarkTokens ? this.benchmarkTokens : this.worldTokens;
      const nTokens = Math.floor(tokensSrc.length / TOKEN_STRIDE);
      const displaySH = mode === 5 && this.benchmarkGtSH ? this.benchmarkGtSH : this.lastPredSH;
      this.engine.uploadWorldSHToTSL(tokensSrc, displaySH, nTokens, this.benchmarkGtSH);
      this._rebuildOverlays(tokensSrc, displaySH, nTokens);
    }

    // Update UI buttons & split-screen divider
    document.querySelectorAll('#mode-buttons button').forEach((btn) => {
      const m = Number((btn as HTMLElement).dataset.mode);
      btn.classList.toggle('active', m === mode);
    });
    const splitEl = document.getElementById('split-divider');
    const sliderRow = document.getElementById('split-slider-row');
    splitEl?.classList.toggle('hidden', mode !== 4);
    sliderRow?.classList.toggle('hidden', mode !== 4);
  }

  public setOverlays(opts: { patches?: boolean; shProbes?: boolean }): void {
    if (opts.patches !== undefined) this.showPatches = opts.patches;
    if (opts.shProbes !== undefined) this.showSHProbes = opts.shProbes;
    const chkP = document.getElementById('chk-patches') as HTMLInputElement | null;
    const chkS = document.getElementById('chk-sh-probes') as HTMLInputElement | null;
    if (chkP) chkP.checked = this.showPatches;
    if (chkS) chkS.checked = this.showSHProbes;

    const tokensSrc = this.isBenchmarkRoom && this.benchmarkTokens ? this.benchmarkTokens : this.worldTokens;
    const nTokens = Math.floor(tokensSrc.length / TOKEN_STRIDE);
    const displaySH = this.currentViewMode === 5 && this.benchmarkGtSH ? this.benchmarkGtSH : this.lastPredSH;
    if (nTokens > 0 && displaySH.length > 0) {
      this._rebuildOverlays(tokensSrc, displaySH, nTokens);
    }
  }

  private _clearOverlays(): void {
    if (this.patchSpheresMesh) {
      this.overlayGroup.remove(this.patchSpheresMesh);
      this.patchSpheresMesh.geometry.dispose();
      (this.patchSpheresMesh.material as THREE.Material).dispose();
      this.patchSpheresMesh = null;
    }
    if (this.shProbeLines) {
      this.overlayGroup.remove(this.shProbeLines);
      this.shProbeLines.geometry.dispose();
      (this.shProbeLines.material as THREE.Material).dispose();
      this.shProbeLines = null;
    }
  }

  private _rebuildOverlays(tokens: Float32Array, shLocal: Float32Array, nTokens: number): void {
    this._clearOverlays();
    if (!this.showPatches && !this.showSHProbes) return;

    if (this.showPatches) {
      const sphereGeo = new THREE.SphereGeometry(0.055, 10, 8);
      const sphereMat = new THREE.MeshBasicNodeMaterial({ color: 0xffffff });
      const inst = new THREE.InstancedMesh(sphereGeo, sphereMat, nTokens);
      const dummy = new THREE.Object3D();
      const col = new THREE.Color();

      for (let i = 0; i < nTokens; i++) {
        const tb = i * TOKEN_STRIDE;
        const sb = i * SH_STRIDE;
        dummy.position.set(
          tokens[tb + 0] + tokens[tb + 9] * 0.03,
          tokens[tb + 1] + tokens[tb + 10] * 0.03,
          tokens[tb + 2] + tokens[tb + 11] * 0.03,
        );
        dummy.updateMatrix();
        inst.setMatrixAt(i, dummy.matrix);
        // Tone-mapped indirect bounce color for clear visual readout
        const r = Math.min(1, shLocal[sb + 0] / (0.6 + shLocal[sb + 0]));
        const g = Math.min(1, shLocal[sb + 1] / (0.6 + shLocal[sb + 1]));
        const b = Math.min(1, shLocal[sb + 2] / (0.6 + shLocal[sb + 2]));
        col.setRGB(Math.pow(r, 1 / 2.2), Math.pow(g, 1 / 2.2), Math.pow(b, 1 / 2.2));
        inst.setColorAt(i, col);
      }
      inst.instanceMatrix.needsUpdate = true;
      if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
      this.patchSpheresMesh = inst;
      this.overlayGroup.add(inst);
    }

    if (this.showSHProbes) {
      const verts = new Float32Array(nTokens * 2 * 3);
      const colors = new Float32Array(nTokens * 2 * 3);
      for (let i = 0; i < nTokens; i++) {
        const tb = i * TOKEN_STRIDE;
        const sb = i * SH_STRIDE;
        const px = tokens[tb + 0], py = tokens[tb + 1], pz = tokens[tb + 2];
        const tx = tokens[tb + 3], ty = tokens[tb + 4], tz = tokens[tb + 5];
        const bx = tokens[tb + 6], by = tokens[tb + 7], bz = tokens[tb + 8];
        const nx = tokens[tb + 9], ny = tokens[tb + 10], nz = tokens[tb + 11];

        // Luminance-weighted Order-1 SH dominant incoming bounce direction in world space
        const iT = 0.2126 * shLocal[sb + 3] + 0.7152 * shLocal[sb + 4] + 0.0722 * shLocal[sb + 5];
        const iB = 0.2126 * shLocal[sb + 6] + 0.7152 * shLocal[sb + 7] + 0.0722 * shLocal[sb + 8];
        const iN = 0.2126 * shLocal[sb + 9] + 0.7152 * shLocal[sb + 10] + 0.0722 * shLocal[sb + 11];

        let dx = iT * tx + iB * bx + iN * nx;
        let dy = iT * ty + iB * by + iN * ny;
        let dz = iT * tz + iB * bz + iN * nz;
        const len = Math.hypot(dx, dy, dz) || 1.0;
        const arrowLen = Math.min(0.45, 0.12 + 0.30 * Math.min(1, len));
        dx = (dx / len) * arrowLen;
        dy = (dy / len) * arrowLen;
        dz = (dz / len) * arrowLen;

        const vOff = i * 6;
        verts[vOff + 0] = px; verts[vOff + 1] = py; verts[vOff + 2] = pz;
        verts[vOff + 3] = px + dx; verts[vOff + 4] = py + dy; verts[vOff + 5] = pz + dz;

        const r = Math.min(1, shLocal[sb + 0] * 1.4);
        const g = Math.min(1, shLocal[sb + 1] * 1.4);
        const b = Math.min(1, shLocal[sb + 2] * 1.4);
        colors[vOff + 0] = 0.2; colors[vOff + 1] = 0.2; colors[vOff + 2] = 0.25;
        colors[vOff + 3] = Math.max(0.25, r); colors[vOff + 4] = Math.max(0.25, g); colors[vOff + 5] = Math.max(0.25, b);
      }
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
      lineGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      const lineMat = new THREE.LineBasicMaterial({ vertexColors: true });
      this.shProbeLines = new THREE.LineSegments(lineGeo, lineMat);
      this.overlayGroup.add(this.shProbeLines);
    }
  }

  public async dropGLBBuffer(arrayBuffer: ArrayBuffer, filename: string): Promise<void> {
    if (this.isBenchmarkRoom) {
      await this.loadRoom(3);
    }
    const gltf = await this.gltfLoader.parseAsync(arrayBuffer, '');
    const root = gltf.scene;

    // Normalize dropped .glb to ~1.2m tall resting on floor/stage
    const box = new THREE.Box3().setFromObject(root);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    const maxDim = Math.max(size.x, size.y, size.z, 1e-4);
    const scale = 1.25 / maxDim;

    let added = 0;
    root.traverse((child: any) => {
      if (child.isMesh && this.roomObjects.length < 30) {
        const clonedGeom = child.geometry.clone();
        clonedGeom.translate(-center.x, -box.min.y, -center.z);
        clonedGeom.scale(scale, scale, scale);
        clonedGeom.computeVertexNormals();

        const srcMat: any = Array.isArray(child.material) ? child.material[0] : child.material;
        const nodeMat = new THREE.MeshStandardNodeMaterial({
          color: srcMat?.color ? srcMat.color.clone() : new THREE.Color(0xe0dcd5),
          roughness: typeof srcMat?.roughness === 'number' ? srcMat.roughness : 0.65,
          metalness: typeof srcMat?.metalness === 'number' ? srcMat.metalness : 0.1,
          map: srcMat?.map ?? null,
          normalMap: srcMat?.normalMap ?? null,
        });

        const objId = this.roomObjects.length;
        applyRadiantGI(nodeMat, this.engine, objId, this.giUniforms);
        const mesh = new THREE.Mesh(clonedGeom, nodeMat);
        mesh.name = `${filename} #${added + 1}`;
        mesh.position.set(0, 0.5, 0.3);
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        this.roomGroup.add(mesh);
        const spec: RoomObjectSpec = { name: mesh.name, mesh, draggable: true, isEmitter: false };
        this.roomObjects.push(spec);
        this.localTokens.push(sampleMeshPatches(mesh, objId, PATCHES_PER_OBJECT));
        this._selectSpec(spec);
        added++;
      }
    });

    if (added > 0) {
      await this.recomputeGI();
      this._showToast(`Added ${filename} (${added * 32} live GI patches)`);
    } else {
      this._showToast(`No meshes found in ${filename}`);
    }
  }

  // -------------------------------------------------------------------------
  // Interactive Selection, Painting & Programmatic Manipulation
  // -------------------------------------------------------------------------

  private _selectSpec(spec: RoomObjectSpec | null): void {
    this.selectedSpec = spec;
    const label = document.getElementById('selected-name');
    if (spec && spec.draggable) {
      this.transform.attach(spec.mesh);
      if (label) label.textContent = `Selected: ${spec.name}`;
    } else {
      this.transform.detach();
      if (label) label.textContent = 'Selected: None';
    }
  }

  private _deselect(): void {
    this._selectSpec(null);
  }

  public async moveObjectByName(name: string, dx: number, dy: number, dz: number): Promise<boolean> {
    const spec = this.roomObjects.find((o) => o.name === name);
    if (!spec) return false;
    spec.mesh.position.x += dx;
    spec.mesh.position.y += dy;
    spec.mesh.position.z += dz;
    if (spec.linkedLight && spec.lightOffset) {
      spec.linkedLight.position.copy(spec.mesh.position).add(spec.lightOffset);
    }
    await this.recomputeGI();
    return true;
  }

  public async setObjectColorByName(name: string, hexColor: string): Promise<boolean> {
    const idx = this.roomObjects.findIndex((o) => o.name === name);
    if (idx < 0) return false;
    const spec = this.roomObjects[idx];
    const col = new THREE.Color(hexColor);
    const mat = spec.mesh.material as THREE.MeshStandardNodeMaterial;
    mat.color.copy(col);
    this.localTokens[idx].materialOverride = {
      ...(this.localTokens[idx].materialOverride ?? {}),
      albedo: [col.r, col.g, col.b],
    };
    // Re-apply TSL node so diffuse bounce picks up new base albedo immediately
    applyRadiantGI(mat, this.engine, idx, this.giUniforms, { isEmitter: spec.isEmitter });
    await this.recomputeGI();
    return true;
  }

  public getObjectMeanIndirectRGB(name: string): [number, number, number] {
    const idx = this.roomObjects.findIndex((o) => o.name === name);
    if (idx < 0 || this.lastPredSH.length === 0) return [0, 0, 0];
    const start = idx * PATCHES_PER_OBJECT;
    let r = 0, g = 0, b = 0;
    for (let k = 0; k < PATCHES_PER_OBJECT; k++) {
      const off = (start + k) * SH_STRIDE;
      r += this.lastPredSH[off + 0];
      g += this.lastPredSH[off + 1];
      b += this.lastPredSH[off + 2];
    }
    return [r / PATCHES_PER_OBJECT, g / PATCHES_PER_OBJECT, b / PATCHES_PER_OBJECT];
  }

  /**
   * O4 Verification Helper: Run pure WebGPU WGSL Compute Transformer on locked PyTorch FP16
   * parity probes (`parity_probes.json`) and report exact RMSE / max_abs / PSNR / compute ms!
   */
  public async runParityAudit(probesUrl = PROBES_URL): Promise<any> {
    const resp = await fetch(probesUrl);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${probesUrl}`);
    const data = await resp.json();
    const results = [];
    let worstRMSE = 0;
    let totalComputeMs = 0;

    for (const p of data.probes) {
      const nTokens: number = p.n_tokens;
      const tokens = new Float32Array(nTokens * TOKEN_STRIDE);
      const refSH = new Float32Array(nTokens * SH_STRIDE);
      const gtSH = new Float32Array(nTokens * SH_STRIDE);

      for (let i = 0; i < nTokens; i++) {
        for (let k = 0; k < TOKEN_STRIDE; k++) tokens[i * TOKEN_STRIDE + k] = p.tokens[i][k];
        for (let k = 0; k < SH_STRIDE; k++) {
          refSH[i * SH_STRIDE + k] = p.pytorch_fp16_ref_sh[i][k];
          gtSH[i * SH_STRIDE + k] = p.gt_sh[i][k];
        }
      }

      const wgslSH = await this.engine.inferTokens(tokens, false);
      const rmse = computeSHRMSE(wgslSH, refSH);
      let maxAbs = 0;
      for (let i = 0; i < wgslSH.length; i++) {
        const d = Math.abs(wgslSH[i] - refSH[i]);
        if (d > maxAbs) maxAbs = d;
      }
      const psnrVsGt = computeTonemappedPSNR(wgslSH, gtSH);
      worstRMSE = Math.max(worstRMSE, rmse);
      totalComputeMs += this.engine.lastComputeMs;

      results.push({
        scene_index: p.scene_index,
        n_tokens: nTokens,
        wgsl_vs_pytorch_rmse: Number(rmse.toFixed(6)),
        wgsl_vs_pytorch_max_abs: Number(maxAbs.toFixed(6)),
        wgsl_psnr_vs_gt_db: Number(psnrVsGt.toFixed(3)),
        pytorch_psnr_vs_gt_db: p.psnr_vs_gt_db,
        compute_ms: Number(this.engine.lastComputeMs.toFixed(2)),
      });
    }

    // Restore current room SH after running probes
    await this.recomputeGI();

    return {
      gate: 'O4_PYTORCH_TO_WEBGPU_WGSL_PARITY',
      passed: worstRMSE <= 0.005,
      worst_rmse: Number(worstRMSE.toFixed(6)),
      mean_compute_ms: Number((totalComputeMs / Math.max(results.length, 1)).toFixed(2)),
      probes: results,
    };
  }

  /**
   * Measure real steady & live-drag frame rates on Apple M1 Metal 3!
   */
  public async measurePerformance(steadyFrames = 45, dragFrames = 36): Promise<any> {
    // 1. Steady-state FPS (0 ms GI compute!)
    const t0 = performance.now();
    for (let i = 0; i < steadyFrames; i++) {
      await new Promise((r) => requestAnimationFrame(r));
    }
    const steadyMs = performance.now() - t0;
    const steadyFps = (steadyFrames / steadyMs) * 1000;

    // 2. Live-drag FPS (real non-blocking dirty-flag drag loop!) + true GPU GI compute latency
    const spec = this.roomObjects[3] ?? this.roomObjects[0];
    const startX = spec.mesh.position.x;
    const computeTimes: number[] = [];
    // Measure 6 direct GPU compute latencies first
    for (let k = 0; k < 6; k++) {
      spec.mesh.position.x = startX + (k % 2 === 0 ? 0.25 : 0.0);
      await this.recomputeGI();
      computeTimes.push(this.engine.lastComputeMs);
    }
    // Measure real interactive rAF drag loop FPS while continuously perturbing the wall
    const td0 = performance.now();
    for (let i = 0; i < dragFrames; i++) {
      spec.mesh.position.x = startX + 0.35 * Math.sin(i * 0.25);
      this.giDirty = true;
      await new Promise((r) => requestAnimationFrame(r));
    }
    const dragMs = performance.now() - td0;
    const dragFps = (dragFrames / dragMs) * 1000;
    spec.mesh.position.x = startX;
    this.giDirty = false;
    await this.recomputeGI();

    const meanComputeMs = computeTimes.reduce((a, b) => a + b, 0) / Math.max(computeTimes.length, 1);

    return {
      steady_fps: Number(steadyFps.toFixed(1)),
      live_drag_fps: Number(dragFps.toFixed(1)),
      mean_gi_compute_ms: Number(meanComputeMs.toFixed(2)),
      n_tokens: this.engine.activeTokenCount,
    };
  }

  // -------------------------------------------------------------------------
  // UI Wiring & Main Render Loop
  // -------------------------------------------------------------------------

  private _wireUI(): void {
    document.querySelectorAll('#room-buttons button').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = Number((btn as HTMLElement).dataset.room);
        this.loadRoom(idx);
      });
    });

    document.querySelectorAll('#mode-buttons button').forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = Number((btn as HTMLElement).dataset.mode);
        this.setViewMode(mode);
      });
    });

    const splitSlider = document.getElementById('split-slider') as HTMLInputElement;
    const splitDiv = document.getElementById('split-divider') as HTMLElement;
    splitSlider?.addEventListener('input', () => {
      const v = Number(splitSlider.value);
      this.giUniforms.splitX.value = v;
      if (splitDiv) splitDiv.style.left = `${v * 100}%`;
    });

    const gainSlider = document.getElementById('gi-gain-slider') as HTMLInputElement;
    const gainVal = document.getElementById('gi-gain-val') as HTMLElement;
    this.giUniforms.giIntensity.value = Number(gainSlider?.value ?? 1.15);
    gainSlider?.addEventListener('input', () => {
      const v = Number(gainSlider.value);
      this.giUniforms.giIntensity.value = v;
      if (gainVal) gainVal.textContent = `${v.toFixed(2)}×`;
    });

    document.getElementById('chk-patches')?.addEventListener('change', (e) => {
      this.setOverlays({ patches: (e.target as HTMLInputElement).checked });
    });
    document.getElementById('chk-sh-probes')?.addEventListener('change', (e) => {
      this.setOverlays({ shProbes: (e.target as HTMLInputElement).checked });
    });

    document.getElementById('btn-gizmo-t')?.addEventListener('click', () => {
      this.transform.setMode('translate');
      document.getElementById('btn-gizmo-t')?.classList.add('active');
      document.getElementById('btn-gizmo-r')?.classList.remove('active');
    });
    document.getElementById('btn-gizmo-r')?.addEventListener('click', () => {
      this.transform.setMode('rotate');
      document.getElementById('btn-gizmo-r')?.classList.add('active');
      document.getElementById('btn-gizmo-t')?.classList.remove('active');
    });
    document.getElementById('btn-deselect')?.addEventListener('click', () => this._deselect());

    document.querySelectorAll('#color-swatches .swatch').forEach((btn) => {
      btn.addEventListener('click', () => {
        if (!this.selectedSpec) {
          this._showToast('Click a wall or object first to tint its bounce color');
          return;
        }
        const hex = (btn as HTMLElement).dataset.color!;
        this.setObjectColorByName(this.selectedSpec.name, hex);
      });
    });

    // Click picking on viewport
    let downX = 0, downY = 0;
    this.canvas.addEventListener('pointerdown', (e) => {
      downX = e.clientX; downY = e.clientY;
    });
    this.canvas.addEventListener('pointerup', (e) => {
      if (this.isDraggingGizmo) return;
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 6) return;
      const rect = this.canvas.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      this.raycaster.setFromCamera(ndc, this.camera);
      const meshes = this.roomObjects.filter((o) => o.draggable && o.mesh.visible).map((o) => o.mesh);
      const hits = this.raycaster.intersectObjects(meshes, false);
      if (hits.length > 0) {
        const hitObj = hits[0].object;
        const spec = this.roomObjects.find((o) => o.mesh === hitObj) ?? null;
        this._selectSpec(spec);
      } else {
        this._deselect();
      }
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'g' || e.key === 'G') this.transform.setMode('translate');
      if (e.key === 'r' || e.key === 'R') this.transform.setMode('rotate');
      if (e.key === 'Escape') this._deselect();
    });

    // Drag & drop .glb support
    const dropZone = document.getElementById('drop-zone')!;
    const fileInput = document.getElementById('glb-file-input') as HTMLInputElement;
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async () => {
      const f = fileInput.files?.[0];
      if (f) await this.dropGLBBuffer(await f.arrayBuffer(), f.name);
    });
    for (const ev of ['dragenter', 'dragover']) {
      window.addEventListener(ev, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
      });
    }
    for (const ev of ['dragleave', 'drop']) {
      window.addEventListener(ev, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
      });
    }
    window.addEventListener('drop', async (e: DragEvent) => {
      const f = e.dataTransfer?.files?.[0];
      if (f && (f.name.endsWith('.glb') || f.name.endsWith('.gltf'))) {
        await this.dropGLBBuffer(await f.arrayBuffer(), f.name);
      }
    });
  }

  private _updateRoomButtons(activeIdx: number): void {
    document.querySelectorAll('#room-buttons button').forEach((btn) => {
      btn.classList.toggle('active', Number((btn as HTMLElement).dataset.room) === activeIdx);
    });
  }

  private _updateMetricsHUD(computeMs: number, nTokens: number, statusText: string): void {
    const elMs = document.getElementById('m-gi-ms');
    const elTok = document.getElementById('m-tokens');
    const elSt = document.getElementById('m-state');
    if (elMs) elMs.textContent = `${computeMs.toFixed(1)} ms`;
    if (elTok) elTok.textContent = `${nTokens}`;
    if (elSt) elSt.textContent = statusText;
  }

  private _showToast(msg: string): void {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.remove('hidden');
    window.setTimeout(() => el.classList.add('hidden'), 2600);
  }

  private _animate = (): void => {
    requestAnimationFrame(this._animate);

    const now = performance.now();
    const dt = now - this.lastFrameTs;
    this.lastFrameTs = now;
    this.frameCount++;
    this.fpsAccumTime += dt;
    if (this.fpsAccumTime >= 400) {
      this.currentFps = (this.frameCount / this.fpsAccumTime) * 1000;
      const elFps = document.getElementById('m-fps');
      if (elFps) elFps.textContent = `${Math.round(this.currentFps)}`;
      if (!this.giDirty && !this.computeInFlight && this.engine?.isWeightsLoaded) {
        const elMs = document.getElementById('m-gi-ms');
        const elSt = document.getElementById('m-state');
        if (elMs) elMs.textContent = '0.0 ms';
        if (elSt) elSt.textContent = 'STATIC';
      }
      this.frameCount = 0;
      this.fpsAccumTime = 0;
    }

    // Non-blocking dirty-flag WebGPU GI recompute (0 ms when static!)
    if (this.giDirty && !this.computeInFlight) {
      this.giDirty = false;
      this.computeInFlight = true;
      this.recomputeGI().finally(() => {
        this.computeInFlight = false;
      });
    }

    this.orbit.update();
    // Synchronous Three.js 0.187.0 WebGPU render!
    this.renderer.render(this.scene, this.camera);
  };
}

const app = new RadiantShowcaseApp();
(window as any).__RADIANT_APP__ = app;
