/**
 * 4 Unseen Playable Rooms + Held-Out 1024-spp Benchmark Room Builder for THREE-RADIANT-GI.
 *
 * Every mesh is built from crisp watertight Three.js BufferGeometries with
 * MeshStandardNodeMaterial / MeshPhysicalNodeMaterial and real Three.js direct shadow-casting lights
 * paired with emissive area-light panels (so direct + zero-bake 3-bounce RADIANT-GI compose cleanly!).
 */

import * as THREE from 'three/webgpu';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export interface RoomObjectSpec {
  name: string;
  mesh: THREE.Mesh;
  draggable: boolean;
  isEmitter: boolean;
  linkedLight?: THREE.Light;
  lightOffset?: THREE.Vector3;
}

export interface RoomBuildResult {
  title: string;
  cameraPos: THREE.Vector3;
  cameraTarget: THREE.Vector3;
  objects: RoomObjectSpec[];
  lights: THREE.Light[];
  isBenchmarkProbe?: boolean;
  benchmarkTokens?: Float32Array;
  benchmarkGtSH?: Float32Array;
}

function makeMat(
  colorHex: number,
  opts?: {
    roughness?: number;
    metalness?: number;
    emissiveHex?: number;
    emissiveIntensity?: number;
    physical?: boolean;
  },
): THREE.MeshStandardNodeMaterial | THREE.MeshPhysicalNodeMaterial {
  const params: any = {
    color: colorHex,
    roughness: opts?.roughness ?? 0.85,
    metalness: opts?.metalness ?? 0.0,
  };
  if (opts?.emissiveHex !== undefined) {
    params.emissive = new THREE.Color(opts.emissiveHex);
    params.emissiveIntensity = opts.emissiveIntensity ?? 1.0;
  }
  return opts?.physical
    ? new THREE.MeshPhysicalNodeMaterial(params)
    : new THREE.MeshStandardNodeMaterial(params);
}

function addBox(
  objects: RoomObjectSpec[],
  name: string,
  size: [number, number, number],
  pos: [number, number, number],
  mat: THREE.MeshStandardNodeMaterial | THREE.MeshPhysicalNodeMaterial,
  opts?: { draggable?: boolean; isEmitter?: boolean; rotY?: number; segs?: number },
): THREE.Mesh {
  const s = opts?.segs ?? 4;
  const geom = new THREE.BoxGeometry(size[0], size[1], size[2], s, s, s);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.name = name;
  mesh.position.set(pos[0], pos[1], pos[2]);
  if (opts?.rotY) mesh.rotation.y = opts.rotY;
  mesh.castShadow = !opts?.isEmitter;
  mesh.receiveShadow = !opts?.isEmitter;
  objects.push({
    name,
    mesh,
    draggable: opts?.draggable ?? true,
    isEmitter: opts?.isEmitter ?? false,
  });
  return mesh;
}

function createVaultGeometry(width: number, height: number, depth: number): THREE.BufferGeometry {
  // Smooth inward-facing vaulted arch ceiling (all 32 patches cover the visible interior arch cleanly!)
  const rx = width * 0.52;
  const ry = height * 0.36;
  const segsU = 32;
  const segsZ = 10;
  const positions: number[] = [];
  const indices: number[] = [];
  for (let j = 0; j <= segsZ; j++) {
    const z = -depth * 0.5 + depth * (j / segsZ);
    for (let i = 0; i <= segsU; i++) {
      const a = Math.PI * (i / segsU);
      const x = Math.cos(a) * rx;
      const y = Math.sin(a) * ry;
      positions.push(x, y, z);
    }
  }
  const stride = segsU + 1;
  for (let j = 0; j < segsZ; j++) {
    for (let i = 0; i < segsU; i++) {
      const a = j * stride + i;
      const b = a + 1;
      const c = a + stride;
      const d = c + 1;
      // Winding facing inward (-radial) into the gallery
      indices.push(a, d, b);
      indices.push(a, c, d);
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

function createVaseGeometry(): THREE.BufferGeometry {
  const pts: THREE.Vector2[] = [];
  for (let i = 0; i <= 22; i++) {
    const t = i / 22;
    const r = Math.max(0.06, 0.24 + 0.14 * Math.sin(t * Math.PI * 1.6) - 0.06 * Math.cos(t * Math.PI * 3.0));
    pts.push(new THREE.Vector2(r, (t - 0.5) * 1.15));
  }
  pts.unshift(new THREE.Vector2(0.001, -0.575));
  pts.push(new THREE.Vector2(0.001, 0.575));
  const raw = new THREE.LatheGeometry(pts, 36);
  // Weld vertical Lathe seam vertices so vertex normals are 100% seamless!
  raw.deleteAttribute('uv');
  const welded = mergeVertices(raw, 1e-4);
  welded.computeVertexNormals();
  raw.dispose();
  return welded;
}

function makeDirectLight(
  pos: [number, number, number],
  colorHex: number,
  intensity: number,
  distance = 16,
): THREE.PointLight {
  const light = new THREE.PointLight(colorHex, intensity, distance, 1.65);
  light.position.set(pos[0], pos[1], pos[2]);
  light.castShadow = true;
  light.shadow.mapSize.set(1024, 1024);
  light.shadow.bias = -0.0012;
  light.shadow.radius = 4.0;
  return light;
}

/** Add a soft 2-light area softbox cluster (primary shadow caster + soft fill offset) for softer penumbras. */
function addSoftboxLights(
  lights: THREE.Light[],
  center: [number, number, number],
  colorHex: number,
  totalIntensity: number,
  spanX = 0.55,
  spanZ = 0.55,
  distance = 16,
): THREE.PointLight {
  const main = makeDirectLight(center, colorHex, totalIntensity * 0.68, distance);
  const fill = new THREE.PointLight(colorHex, totalIntensity * 0.32, distance, 1.5);
  fill.position.set(center[0] + spanX * 0.45, center[1] - 0.12, center[2] + spanZ * 0.45);
  fill.castShadow = false;
  lights.push(main, fill);
  return main;
}

/**
 * Add an inward-facing (-Z) rear bounce wall at the front of the room (z = +D/2).
 * Because it uses FrontSide facing -Z (into the room), the camera outside looking in
 * sees right through it, while its 32 inward-facing patches bounce realistic indirect light
 * back onto the front of columns, doorway piers, and sculptures (eliminating black backs)!
 */
function addVirtualFrontBounceWall(
  objects: RoomObjectSpec[],
  W: number,
  H: number,
  zFront: number,
  colorHex = 0xc9c5bd,
): void {
  const geom = new THREE.PlaneGeometry(W, H, 6, 6);
  // Rotate so normal faces -Z (into the room); FrontSide makes it invisible from camera at z > zFront!
  geom.rotateY(Math.PI);
  const mat = makeMat(colorHex, { roughness: 0.9 });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.name = 'Front Bounce Wall';
  mesh.position.set(0, H * 0.5, zFront);
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  objects.push({ name: 'Front Bounce Wall', mesh, draggable: false, isEmitter: false });
}

// ---------------------------------------------------------------------------
// Room 1: Classic Cornell Color-Bleed Box
// ---------------------------------------------------------------------------
export function buildCornellRoom(): RoomBuildResult {
  const objects: RoomObjectSpec[] = [];
  const lights: THREE.Light[] = [];
  const W = 5.2, H = 4.0, D = 5.2, t = 0.2;

  // Floor, Ceiling, Back wall, Virtual Front bounce wall (neutral matte plaster)
  addBox(objects, 'Floor', [W + 2 * t, t, D + 2 * t], [0, -t * 0.5, 0], makeMat(0xcac6be, { roughness: 0.88 }), { draggable: false });
  addBox(objects, 'Ceiling', [W + 2 * t, t, D + 2 * t], [0, H + t * 0.5, 0], makeMat(0xd4d0c8, { roughness: 0.92 }), { draggable: false });
  addBox(objects, 'Back Wall', [W, H, t], [0, H * 0.5, -D * 0.5 - t * 0.5], makeMat(0xcfcac2, { roughness: 0.9 }), { draggable: true });
  addVirtualFrontBounceWall(objects, W, H, D * 0.5, 0xcac6be);

  // Vivid Cornell Crimson Left Wall & Emerald Right Wall (draggable!)
  addBox(objects, 'Crimson Left Wall', [t, H, D + 2 * t], [-W * 0.5 - t * 0.5, H * 0.5, 0], makeMat(0xd61c17, { roughness: 0.88 }), { draggable: true });
  addBox(objects, 'Emerald Right Wall', [t, H, D + 2 * t], [W * 0.5 + t * 0.5, H * 0.5, 0], makeMat(0x1cc729, { roughness: 0.88 }), { draggable: true });

  // Movable Ceiling Area Softbox Lamp
  const lampPos: [number, number, number] = [0, H - 0.1, 0];
  const lampMesh = addBox(
    objects,
    'Ceiling Softbox Lamp',
    [1.6, 0.1, 1.6],
    lampPos,
    makeMat(0xffffff, { emissiveHex: 0xfff3de, emissiveIntensity: 14.0, roughness: 0.9 }),
    { draggable: true, isEmitter: true },
  );
  const direct = addSoftboxLights(lights, [0, H - 0.35, 0], 0xfff3de, 18.0, 0.7, 0.7, 15);
  objects[objects.length - 1].linkedLight = direct;
  objects[objects.length - 1].lightOffset = new THREE.Vector3(0, -0.25, 0);
  void lampMesh;

  // Classic tall & short white gallery plinths + sculptural TorusKnot & Matte Sphere (great color-bleed receivers!)
  addBox(objects, 'Tall White Plinth', [1.15, 2.0, 1.15], [-0.95, 1.0, -0.85], makeMat(0xdad6ce, { roughness: 0.85 }), { rotY: 0.32 });
  addBox(objects, 'Short White Plinth', [1.1, 1.05, 1.1], [1.0, 0.525, 0.65], makeMat(0xdad6ce, { roughness: 0.85 }), { rotY: -0.28 });

  const knotMesh = new THREE.Mesh(
    new THREE.TorusKnotGeometry(0.42, 0.14, 96, 20),
    makeMat(0xe6e4df, { roughness: 0.55, metalness: 0.1, physical: true }),
  );
  knotMesh.name = 'Sculptural Knot';
  knotMesh.position.set(1.0, 1.56, 0.65);
  knotMesh.castShadow = true;
  knotMesh.receiveShadow = true;
  objects.push({ name: 'Sculptural Knot', mesh: knotMesh, draggable: true, isEmitter: false });

  const sphereMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.46, 36, 24),
    makeMat(0xe5e3de, { roughness: 0.35, metalness: 0.72, physical: true }),
  );
  sphereMesh.name = 'Brushed Metal Sphere';
  sphereMesh.position.set(-0.95, 2.46, -0.85);
  sphereMesh.castShadow = true;
  sphereMesh.receiveShadow = true;
  objects.push({ name: 'Brushed Metal Sphere', mesh: sphereMesh, draggable: true, isEmitter: false });

  return {
    title: 'Classic Cornell Color-Bleed Box',
    cameraPos: new THREE.Vector3(0, 2.15, 6.2),
    cameraTarget: new THREE.Vector3(0, 1.85, -0.4),
    objects,
    lights,
  };
}

// ---------------------------------------------------------------------------
// Room 2: Vaulted Brutalist Gallery with Skylight
// ---------------------------------------------------------------------------
export function buildVaultedGalleryRoom(): RoomBuildResult {
  const objects: RoomObjectSpec[] = [];
  const lights: THREE.Light[] = [];
  const W = 6.4, H = 4.4, D = 7.4, t = 0.22;

  addBox(objects, 'Concrete Floor', [W + 2 * t, t, D + 2 * t], [0, -t * 0.5, 0], makeMat(0x9a958d, { roughness: 0.85 }), { draggable: false });
  addBox(objects, 'Left Concrete Wall', [t, H * 0.66, D], [-W * 0.5 - t * 0.5, H * 0.33, 0], makeMat(0xa8a49c, { roughness: 0.9 }), { draggable: false });
  addBox(objects, 'Right Concrete Wall', [t, H * 0.66, D], [W * 0.5 + t * 0.5, H * 0.33, 0], makeMat(0xa8a49c, { roughness: 0.9 }), { draggable: false });
  addBox(objects, 'Back Gallery Wall', [W, H, t], [0, H * 0.5, -D * 0.5 - t * 0.5], makeMat(0xbdb8af, { roughness: 0.9 }), { draggable: true });
  addVirtualFrontBounceWall(objects, W, H, D * 0.5, 0xc4bfb6);

  // Curved Concrete Barrel Vault Roof
  const vaultMesh = new THREE.Mesh(
    createVaultGeometry(W, H, D),
    makeMat(0xb6b2aa, { roughness: 0.9 }),
  );
  vaultMesh.name = 'Barrel Vault Ceiling';
  vaultMesh.position.set(0, H * 0.64, 0);
  vaultMesh.castShadow = true;
  vaultMesh.receiveShadow = true;
  objects.push({ name: 'Barrel Vault Ceiling', mesh: vaultMesh, draggable: false, isEmitter: false });

  // Skylight Emitter Panel under vault apex
  addBox(
    objects,
    'Skylight Panel',
    [1.8, 0.1, 3.6],
    [0, H * 0.96, -0.3],
    makeMat(0xffffff, { emissiveHex: 0xdcecff, emissiveIntensity: 13.0, roughness: 0.9 }),
    { draggable: true, isEmitter: true },
  );
  const skyLight = addSoftboxLights(lights, [0, H * 0.85, -0.15], 0xdcecff, 26.0, 0.9, 1.4, 18);
  objects[objects.length - 1].linkedLight = skyLight;
  objects[objects.length - 1].lightOffset = new THREE.Vector3(0, -0.35, 0);

  // Terracotta & Cobalt Sculptural Bounce Baffles (draggable!)
  addBox(objects, 'Terracotta Bounce Baffle', [0.22, 2.7, 2.4], [-2.05, 1.35, -0.6], makeMat(0xd9481a, { roughness: 0.82 }), { rotY: 0.2 });
  addBox(objects, 'Cobalt Bounce Baffle', [0.22, 2.7, 2.4], [2.05, 1.35, 0.2], makeMat(0x1a47db, { roughness: 0.82 }), { rotY: -0.22 });

  // Central Travertine Plinth + Ceramic Vase
  addBox(objects, 'Travertine Plinth', [1.1, 1.0, 1.1], [0, 0.5, -0.6], makeMat(0xd8d0c2, { roughness: 0.78 }));
  const vase = new THREE.Mesh(createVaseGeometry(), makeMat(0xe8e4dc, { roughness: 0.45, metalness: 0.05, physical: true }));
  vase.name = 'Ceramic Gallery Vessel';
  vase.position.set(0, 1.58, -0.6);
  vase.castShadow = true;
  vase.receiveShadow = true;
  objects.push({ name: 'Ceramic Gallery Vessel', mesh: vase, draggable: true, isEmitter: false });

  // Two Flanking Travertine Columns (placed cleanly inside gallery lighting envelope!)
  for (const sx of [-1, 1]) {
    const col = new THREE.Mesh(
      new THREE.CylinderGeometry(0.26, 0.28, H * 0.66, 28, 6),
      makeMat(0xd4cfc5, { roughness: 0.78 }),
    );
    col.name = sx < 0 ? 'West Column' : 'East Column';
    col.position.set(sx * 1.95, H * 0.33, 1.35);
    col.castShadow = true;
    col.receiveShadow = true;
    objects.push({ name: col.name, mesh: col, draggable: true, isEmitter: false });
  }

  return {
    title: 'Vaulted Brutalist Gallery with Skylight',
    cameraPos: new THREE.Vector3(0.4, 2.2, 6.8),
    cameraTarget: new THREE.Vector3(0, 1.7, -0.8),
    objects,
    lights,
  };
}

// ---------------------------------------------------------------------------
// Room 3: Multi-Room L-Corridor with Doorway Occlusion
// ---------------------------------------------------------------------------
export function buildLCorridorRoom(): RoomBuildResult {
  const objects: RoomObjectSpec[] = [];
  const lights: THREE.Light[] = [];
  const W = 7.0, H = 3.8, D = 6.0, t = 0.2;

  addBox(objects, 'Corridor Floor', [W + 2 * t, t, D + 2 * t], [0, -t * 0.5, 0], makeMat(0xc5bfb6, { roughness: 0.85 }), { draggable: false });
  addBox(objects, 'Corridor Ceiling', [W + 2 * t, t, D + 2 * t], [0, H + t * 0.5, 0], makeMat(0xd5d2cb, { roughness: 0.92 }), { draggable: false });
  addBox(objects, 'Back Wall', [W, H, t], [0, H * 0.5, -D * 0.5 - t * 0.5], makeMat(0xcbc7bf, { roughness: 0.9 }), { draggable: false });
  addBox(objects, 'Amber West Wall', [t, H, D], [-W * 0.5 - t * 0.5, H * 0.5, 0], makeMat(0xe08a12, { roughness: 0.85 }), { draggable: true });
  addBox(objects, 'East Gallery Wall', [t, H, D], [W * 0.5 + t * 0.5, H * 0.5, 0], makeMat(0xcac7c2, { roughness: 0.88 }), { draggable: true });
  addVirtualFrontBounceWall(objects, W, H, D * 0.5 + 0.12, 0xd0cbc2);

  // Partition with Doorway Opening + Draggable Sliding Occlusion Door!
  addBox(objects, 'Doorway Pier Back', [0.24, H, 1.8], [0.0, H * 0.5, -2.1], makeMat(0xc2beb6, { roughness: 0.85 }), { draggable: false });
  addBox(objects, 'Doorway Pier Front', [0.24, H, 1.8], [0.0, H * 0.5, 2.1], makeMat(0xc2beb6, { roughness: 0.85 }), { draggable: false });
  addBox(objects, 'Doorway Lintel', [0.24, 1.1, 2.4], [0.0, H - 0.55, 0.0], makeMat(0xc2beb6, { roughness: 0.85 }), { draggable: false });

  // Draggable Sliding Occluder Panel (starts open; slide to Z=0 to block doorway bounce!)
  addBox(objects, 'Sliding Occluder Door', [0.20, 2.65, 2.35], [0.26, 1.325, -1.65], makeMat(0x474e5c, { roughness: 0.82 }), { draggable: true });

  // Warm Lamp in West Chamber
  addBox(
    objects,
    'West Chamber Lamp',
    [1.4, 0.1, 1.8],
    [-1.9, H - 0.12, 0.0],
    makeMat(0xffffff, { emissiveHex: 0xffebd1, emissiveIntensity: 18.0, roughness: 0.9 }),
    { draggable: true, isEmitter: true },
  );
  const westLight = makeDirectLight([-1.9, H - 0.4, 0.0], 0xffebd1, 22.0, 15);
  lights.push(westLight);
  objects[objects.length - 1].linkedLight = westLight;
  objects[objects.length - 1].lightOffset = new THREE.Vector3(0, -0.28, 0);

  // Target White Sculpture in East Chamber (receives bounce light strictly through the doorway!)
  const eastSculpt = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.68, 2),
    makeMat(0xe5e3de, { roughness: 0.72 }),
  );
  eastSculpt.name = 'East Chamber Sculpture';
  eastSculpt.position.set(1.95, 1.15, 0.0);
  eastSculpt.castShadow = true;
  eastSculpt.receiveShadow = true;
  objects.push({ name: 'East Chamber Sculpture', mesh: eastSculpt, draggable: true, isEmitter: false });

  addBox(objects, 'East Pedestal', [1.0, 0.55, 1.0], [1.95, 0.275, 0.0], makeMat(0xd5d2cb, { roughness: 0.85 }));

  return {
    title: 'Multi-Room L-Corridor with Doorway Occlusion',
    cameraPos: new THREE.Vector3(0.6, 2.6, 7.6),
    cameraTarget: new THREE.Vector3(0.1, 1.6, 0.0),
    objects,
    lights,
  };
}

// ---------------------------------------------------------------------------
// Room 4: Studio Showcase with Sculptural Props & Drop-In .GLB Stage
// ---------------------------------------------------------------------------
export function buildStudioShowcaseRoom(): RoomBuildResult {
  const objects: RoomObjectSpec[] = [];
  const lights: THREE.Light[] = [];
  const W = 6.8, H = 4.2, D = 6.2, t = 0.2;

  addBox(objects, 'Studio Floor', [W + 2 * t, t, D + 2 * t], [0, -t * 0.5, 0], makeMat(0xc4c4c8, { roughness: 0.72 }), { draggable: false });
  addBox(objects, 'Studio Ceiling', [W + 2 * t, t, D + 2 * t], [0, H + t * 0.5, 0], makeMat(0xceced2, { roughness: 0.9 }), { draggable: false });
  addBox(objects, 'Studio Cyclorama Wall', [W, H, t], [0, H * 0.5, -D * 0.5 - t * 0.5], makeMat(0xd8d8dc, { roughness: 0.88 }), { draggable: false });
  addVirtualFrontBounceWall(objects, W, H, D * 0.5, 0xd0d0d4);

  // Movable Colored Bounce Cards (Teal Left, Magenta/Warm Right!)
  addBox(objects, 'Teal Bounce Card', [0.14, 2.8, 2.6], [-2.5, 1.4, -0.2], makeMat(0x14b8c2, { roughness: 0.8 }), { rotY: 0.26 });
  addBox(objects, 'Crimson Bounce Card', [0.14, 2.8, 2.6], [2.5, 1.4, -0.2], makeMat(0xd92662, { roughness: 0.8 }), { rotY: -0.26 });

  // Two Studio Area Softboxes
  addBox(
    objects,
    'Key Studio Softbox',
    [1.8, 0.1, 1.4],
    [-0.9, H - 0.14, 0.4],
    makeMat(0xffffff, { emissiveHex: 0xfff8ee, emissiveIntensity: 14.0, roughness: 0.9 }),
    { draggable: true, isEmitter: true },
  );
  const keyL = makeDirectLight([-0.9, H - 0.4, 0.4], 0xfff8ee, 20.0, 15);
  lights.push(keyL);
  objects[objects.length - 1].linkedLight = keyL;
  objects[objects.length - 1].lightOffset = new THREE.Vector3(0, -0.26, 0);

  // Central Stage Plinth + Hero Sculptures (perfect target for dropped-in .glb models!)
  addBox(objects, 'Hero Stage Plinth', [1.8, 0.5, 1.6], [0, 0.25, -0.4], makeMat(0xdcdad6, { roughness: 0.75 }));

  const heroKnot = new THREE.Mesh(
    new THREE.TorusKnotGeometry(0.55, 0.17, 128, 24, 2, 3),
    makeMat(0xeae8e4, { roughness: 0.38, metalness: 0.18, physical: true }),
  );
  heroKnot.name = 'Hero Studio Sculpture';
  heroKnot.position.set(0, 1.32, -0.4);
  heroKnot.castShadow = true;
  heroKnot.receiveShadow = true;
  objects.push({ name: 'Hero Studio Sculpture', mesh: heroKnot, draggable: true, isEmitter: false });

  const brassOrb = new THREE.Mesh(
    new THREE.SphereGeometry(0.42, 32, 24),
    makeMat(0xd4b26a, { roughness: 0.28, metalness: 0.82, physical: true }),
  );
  brassOrb.name = 'Brass Orb';
  brassOrb.position.set(-1.35, 0.42, 0.85);
  brassOrb.castShadow = true;
  brassOrb.receiveShadow = true;
  objects.push({ name: 'Brass Orb', mesh: brassOrb, draggable: true, isEmitter: false });

  const ceramicVase = new THREE.Mesh(
    createVaseGeometry(),
    makeMat(0xe4e2de, { roughness: 0.62, metalness: 0.0 }),
  );
  ceramicVase.name = 'Studio Vessel';
  ceramicVase.position.set(1.35, 0.58, 0.75);
  ceramicVase.castShadow = true;
  ceramicVase.receiveShadow = true;
  objects.push({ name: 'Studio Vessel', mesh: ceramicVase, draggable: true, isEmitter: false });

  return {
    title: 'Studio Showcase (Drop .GLB Ready)',
    cameraPos: new THREE.Vector3(0, 2.2, 6.4),
    cameraTarget: new THREE.Vector3(0, 1.3, -0.3),
    objects,
    lights,
  };
}

// ---------------------------------------------------------------------------
// Room 5: Held-Out 1024-spp Benchmark Probe Room (live WGSL vs 1024-spp GT!)
// ---------------------------------------------------------------------------
export async function buildHeldoutBenchmarkRoom(probesUrl = '/data/parity_probes.json'): Promise<RoomBuildResult> {
  const resp = await fetch(probesUrl);
  if (!resp.ok) {
    throw new Error(`Failed to fetch held-out benchmark probe from ${probesUrl}`);
  }
  const data = await resp.json();
  const probe = data.probes[0];
  const nTokens: number = probe.n_tokens;
  const tokens = new Float32Array(nTokens * 21);
  const gtSH = new Float32Array(nTokens * 12);

  for (let i = 0; i < nTokens; i++) {
    for (let k = 0; k < 21; k++) tokens[i * 21 + k] = probe.tokens[i][k];
    for (let k = 0; k < 12; k++) gtSH[i * 12 + k] = probe.gt_sh[i][k];
  }

  // Reconstruct clickable patch proxy discs/boxes per object group so user can inspect
  // live WGSL prediction vs 1024-spp Reference Path Trace & Per-Patch Error Heatmap!
  const objects: RoomObjectSpec[] = [];
  const lights: THREE.Light[] = [];

  // Group patches by objectId (32 patches per object) to synthesize clean watertight bounding hulls
  // plus render exact patch surfel discs!
  const nObjs = Math.floor(nTokens / 32);
  let emitterLightAdded = false;

  for (let oid = 0; oid < nObjs; oid++) {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let ar = 0, ag = 0, ab = 0, er = 0, eg = 0, eb = 0, rough = 0.8, metal = 0;

    for (let k = 0; k < 32; k++) {
      const idx = (oid * 32 + k) * 21;
      const px = tokens[idx + 0], py = tokens[idx + 1], pz = tokens[idx + 2];
      if (px < minX) minX = px; if (py < minY) minY = py; if (pz < minZ) minZ = pz;
      if (px > maxX) maxX = px; if (py > maxY) maxY = py; if (pz > maxZ) maxZ = pz;
      ar = tokens[idx + 13]; ag = tokens[idx + 14]; ab = tokens[idx + 15];
      rough = tokens[idx + 16]; metal = tokens[idx + 17];
      er = tokens[idx + 18]; eg = tokens[idx + 19]; eb = tokens[idx + 20];
    }

    const sx = Math.max(maxX - minX, 0.16);
    const sy = Math.max(maxY - minY, 0.16);
    const sz = Math.max(maxZ - minZ, 0.16);
    const cx = 0.5 * (minX + maxX);
    const cy = 0.5 * (minY + maxY);
    const cz = 0.5 * (minZ + maxZ);
    const isEm = Math.max(er, eg, eb) > 0.05;

    const col = new THREE.Color(ar, ag, ab);
    const mat = new THREE.MeshStandardNodeMaterial({
      color: col,
      roughness: rough,
      metalness: metal,
      emissive: isEm ? new THREE.Color(er, eg, eb).multiplyScalar(0.15) : new THREE.Color(0),
      emissiveIntensity: isEm ? 6.0 : 0.0,
    });

    // Hybrid Benchmark Geometry:
    // - Architectural slabs/emitters (oid <= 6 or isEm) -> crisp watertight BoxGeometry (zero gaps!)
    // - Sculptural interior props (oid > 6) -> oriented surfel discs showing true organic patch geometry!
    const isFrontWall = oid <= 5 && cz > 1.8 && sz < 0.45;
    let mesh: THREE.Mesh;
    if (oid <= 6 || isEm) {
      const geom = new THREE.BoxGeometry(sx, sy, sz, 4, 4, 4);
      mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(cx, cy, cz);
    } else {
      const discSegs = 12;
      const posArr: number[] = [];
      const nrmArr: number[] = [];
      const idxArr: number[] = [];
      let vBase = 0;
      for (let k = 0; k < 32; k++) {
        const idx = (oid * 32 + k) * 21;
        const px = tokens[idx + 0], py = tokens[idx + 1], pz = tokens[idx + 2];
        const tx = tokens[idx + 3], ty = tokens[idx + 4], tz = tokens[idx + 5];
        const bx = tokens[idx + 6], by = tokens[idx + 7], bz = tokens[idx + 8];
        const nx = tokens[idx + 9], ny = tokens[idx + 10], nz = tokens[idx + 11];
        const area = Math.max(tokens[idx + 12], 0.01);
        const rad = Math.min(0.65, Math.sqrt(area / Math.PI) * 1.45);

        posArr.push(px, py, pz);
        nrmArr.push(nx, ny, nz);
        for (let s = 0; s < discSegs; s++) {
          const ang = (2 * Math.PI * s) / discSegs;
          const ca = Math.cos(ang) * rad;
          const sa = Math.sin(ang) * rad;
          posArr.push(px + tx * ca + bx * sa, py + ty * ca + by * sa, pz + tz * ca + bz * sa);
          nrmArr.push(nx, ny, nz);
        }
        for (let s = 0; s < discSegs; s++) {
          const nxt = (s + 1) % discSegs;
          idxArr.push(vBase, vBase + 1 + s, vBase + 1 + nxt);
        }
        vBase += 1 + discSegs;
      }
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(posArr, 3));
      geom.setAttribute('normal', new THREE.Float32BufferAttribute(nrmArr, 3));
      geom.setIndex(idxArr);
      mesh = new THREE.Mesh(geom, mat);
    }
    mesh.name = isEm ? `Benchmark Emitter ${oid}` : `Benchmark Object ${oid}`;
    mesh.visible = !isFrontWall;
    mesh.castShadow = !isEm;
    mesh.receiveShadow = !isEm;
    objects.push({ name: mesh.name, mesh, draggable: false, isEmitter: isEm });

    if (isEm && !emitterLightAdded) {
      const l = makeDirectLight([cx, cy - 0.35, cz], 0xfff4e0, 18.0, 14);
      lights.push(l);
      emitterLightAdded = true;
    }
  }

  return {
    title: `Held-Out Unseen Benchmark (${nTokens} tokens, 1024-spp GT)`,
    cameraPos: new THREE.Vector3(0.2, 2.0, 5.2),
    cameraTarget: new THREE.Vector3(0, 1.6, -0.5),
    objects,
    lights,
    isBenchmarkProbe: true,
    benchmarkTokens: tokens,
    benchmarkGtSH: gtSH,
  };
}
