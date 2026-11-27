/**
 * Pure WebGPU WGSL Compute Shader Engine for THREE-RADIANT-GI.
 *
 * Executes the exported FP16 RADIANT SE(3) Form-Factor Transformer (`RDGI` binary, < 20 MB)
 * natively on Apple M1 Metal 3 / WebGPU with zero Python server and zero O(N^2) bias matrices.
 *
 * Also maintains GPU-ready world-space Order-1 SH patch buffers (`StorageBufferAttribute`)
 * directly consumable by `radiantGINode.ts` (`three/tsl`).
 */

import * as THREE from 'three/webgpu';
import {
  PATCHES_PER_OBJECT,
  POS_SCALE_METERS,
  SH_STRIDE,
  TOKEN_STRIDE,
} from './shMath.ts';
import { combineBounceSH, runRadiantCPU, unpackRDG8ToCanonicalF32, unpackRDGIWeightsF32 } from './cpuFallback.ts';
import {
  WGSL_BUILD_SE3_QKV_FUSED,
  WGSL_LINEAR_F32,
  WGSL_PACK_WORLD_SH,
  WGSL_RMS_NORM_F32,
  buildAttentionTurboWGSL,
  buildLinearQ8WGSL,
} from './wgslTurboQ8.ts';

export interface RDGIHeader {
  version: number;
  dModel: number;
  nHeads: number;
  nLayers: number;
  dFF: number;
  nTensors: number;
  flags: number;
  posScale: number;
  hasAreaPrior: boolean;
  isMultiBounce: boolean;
}

export interface ObjectPatchRange {
  objectId: number;
  patchStart: number;
  patchCount: number;
}

interface LayerOffsets {
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

interface WeightLayout {
  stem1W: number; stem1B: number;
  stem2W: number; stem2B: number;
  layers: LayerOffsets[];
  finalNormW: number;
  head1W: number; head1B: number;
  head2W: number; head2B: number;
}

interface Q8LayerOffsets {
  norm1W: number;
  qkvW: number; qkvS: number; qkvB: number;
  wOW: number; wOS: number; wOB: number;
  norm2W: number;
  fc1W: number; fc1S: number; fc1B: number;
  fc2W: number; fc2S: number; fc2B: number;
}

interface Q8WeightLayout {
  stem1W: number; stem1B: number;
  stem2W: number; stem2S: number; stem2B: number;
  layers: Q8LayerOffsets[];
  finalNormW: number;
  head1W: number; head1S: number; head1B: number;
  head2W: number; head2B: number;
}

// ---------------------------------------------------------------------------
// WGSL Compute Shaders (exact PyTorch FP32 math + packed FP16 weight storage)
// ---------------------------------------------------------------------------

const WGSL_CENTROID = /* wgsl */ `
struct Params {
  n_tokens: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tokens: array<f32>;
@group(0) @binding(2) var<storage, read_write> centroid: array<f32>;

var<workgroup> sx: array<f32, 256>;
var<workgroup> sy: array<f32, 256>;
var<workgroup> sz: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let n = params.n_tokens;
  var ax = 0.0;
  var ay = 0.0;
  var az = 0.0;
  var i = tid;
  while (i < n) {
    let base = i * 21u;
    ax += tokens[base + 0u];
    ay += tokens[base + 1u];
    az += tokens[base + 2u];
    i += 256u;
  }
  sx[tid] = ax;
  sy[tid] = ay;
  sz[tid] = az;
  workgroupBarrier();

  var stride = 128u;
  while (stride > 0u) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
      sz[tid] += sz[tid + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }

  if (tid == 0u) {
    let inv = 1.0 / max(f32(n), 1.0);
    centroid[0] = sx[0] * inv;
    centroid[1] = sy[0] * inv;
    centroid[2] = sz[0] * inv;
    centroid[3] = f32(n);
  }
}
`;

const WGSL_PREP_GEO_INTR = /* wgsl */ `
struct Params {
  n_tokens: u32,
  pos_scale: f32,
  area_prior: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tokens: array<f32>;
@group(0) @binding(2) var<storage, read> centroid: array<f32>;
@group(0) @binding(3) var<storage, read_write> geo: array<f32>;   // [N, 16]
@group(0) @binding(4) var<storage, read_write> intr: array<f32>;  // [N, 13]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n_tokens) { return; }

  let tb = i * 21u;
  let inv_s = 1.0 / params.pos_scale;
  let px = (tokens[tb + 0u] - centroid[0]) * inv_s;
  let py = (tokens[tb + 1u] - centroid[1]) * inv_s;
  let pz = (tokens[tb + 2u] - centroid[2]) * inv_s;
  let pn2 = px * px + py * py + pz * pz;
  let area = max(tokens[tb + 12u], 0.0);
  let log_area_prior = select(0.0, log(max(area, 1e-4)), params.area_prior != 0u);

  let gb = i * 16u;
  geo[gb + 0u] = px;
  geo[gb + 1u] = py;
  geo[gb + 2u] = pz;
  geo[gb + 3u] = pn2;
  // T + Riemann Area-Prior log(A_j) in w
  geo[gb + 4u] = tokens[tb + 3u];
  geo[gb + 5u] = tokens[tb + 4u];
  geo[gb + 6u] = tokens[tb + 5u];
  geo[gb + 7u] = log_area_prior;
  // B
  geo[gb + 8u] = tokens[tb + 6u];
  geo[gb + 9u] = tokens[tb + 7u];
  geo[gb + 10u] = tokens[tb + 8u];
  geo[gb + 11u] = 0.0;
  // N
  geo[gb + 12u] = tokens[tb + 9u];
  geo[gb + 13u] = tokens[tb + 10u];
  geo[gb + 14u] = tokens[tb + 11u];
  geo[gb + 15u] = 0.0;

  let ar = clamp(tokens[tb + 13u], 0.0, 1.0);
  let ag = clamp(tokens[tb + 14u], 0.0, 1.0);
  let ab = clamp(tokens[tb + 15u], 0.0, 1.0);
  let rough = clamp(tokens[tb + 16u], 0.0, 1.0);
  let metal = clamp(tokens[tb + 17u], 0.0, 1.0);
  let er = max(tokens[tb + 18u], 0.0);
  let eg = max(tokens[tb + 19u], 0.0);
  let eb = max(tokens[tb + 20u], 0.0);
  let is_em = select(0.0, 1.0, max(er, max(eg, eb)) > 1e-3);

  let ib = i * 13u;
  intr[ib + 0u] = log(1.0 + area);
  intr[ib + 1u] = ar;
  intr[ib + 2u] = ag;
  intr[ib + 3u] = ab;
  intr[ib + 4u] = rough;
  intr[ib + 5u] = metal;
  intr[ib + 6u] = log(1.0 + er);
  intr[ib + 7u] = log(1.0 + eg);
  intr[ib + 8u] = log(1.0 + eb);
  intr[ib + 9u] = is_em;
  intr[ib + 10u] = log(1.0 + er * area);
  intr[ib + 11u] = log(1.0 + eg * area);
  intr[ib + 12u] = log(1.0 + eb * area);
}
`;

const WGSL_RMS_NORM = /* wgsl */ `
struct Params {
  n_tokens: u32,
  dim: u32,
  w_off_u32: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_in: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> x_out: array<f32>;

var<workgroup> sq_sum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let tok = wg.x;
  if (tok >= params.n_tokens) { return; }
  let tid = lid.x;
  let dim = params.dim;
  let base = tok * dim;

  var s = 0.0;
  var d = tid;
  while (d < dim) {
    let v = x_in[base + d];
    s += v * v;
    d += 64u;
  }
  sq_sum[tid] = s;
  workgroupBarrier();

  var stride = 32u;
  while (stride > 0u) {
    if (tid < stride) {
      sq_sum[tid] += sq_sum[tid + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }

  let inv_rms = inverseSqrt(sq_sum[0] / f32(dim) + 1e-5);
  d = tid;
  while (d < dim) {
    let w_word = weights[params.w_off_u32 + (d >> 1u)];
    let w_pair = unpack2x16float(w_word);
    let w = select(w_pair.x, w_pair.y, (d & 1u) != 0u);
    x_out[base + d] = x_in[base + d] * inv_rms * w;
    d += 64u;
  }
}
`;

// Tiled FP16-weight GEMM (TILE_N = 16 tokens per workgroup, K-tile = 32):
// Cuts DRAM weight reads by 16x on Apple M1 unified memory (~180ms -> ~10ms)!
const WGSL_LINEAR = /* wgsl */ `
struct Params {
  n_tokens: u32,
  in_dim: u32,
  out_dim: u32,
  w_off_u32: u32,
  b_off_u32: u32,
  act_mode: u32,   // 0 = none, 1 = gelu_tanh
  add_res: u32,    // 0 = overwrite, 1 = y += res
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_in: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read> res_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> y_out: array<f32>;

// Shared cache: 16 tokens x 32 K-slice = 512 floats (2 KB)
var<workgroup> x_tile: array<f32, 512>;

fn gelu_tanh(x: f32) -> f32 {
  let c = 0.7978845608028654; // sqrt(2 / pi)
  // Clamp arg to [-15, 15] so Apple Metal tanh() never overflows to NaN!
  let arg = clamp(c * (x + 0.044715 * x * x * x), -15.0, 15.0);
  return 0.5 * x * (1.0 + tanh(arg));
}

fn read_f16(base_u32: u32, elem_idx: u32) -> f32 {
  let word = weights[base_u32 + (elem_idx >> 1u)];
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (elem_idx & 1u) != 0u);
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let tid = lid.x;
  let tok_base = wg.y * 16u;
  let in_d = params.in_dim;
  let out_d = params.out_dim;
  let out_idx = wg.x * 64u + tid;
  let valid_out = out_idx < out_d;

  var bias = 0.0;
  if (valid_out) {
    bias = read_f16(params.b_off_u32, out_idx);
  }

  var acc: array<f32, 16>;
  for (var t: u32 = 0u; t < 16u; t++) {
    acc[t] = bias;
  }

  let row_elem = out_idx * in_d;
  let even_in = (in_d & 1u) == 0u;

  var k_start: u32 = 0u;
  while (k_start < in_d) {
    let k_len = min(32u, in_d - k_start);
    // Cooperatively load [16 tokens x k_len] into x_tile (512 floats / 64 threads = 8 floats/thread)
    var idx: u32 = tid;
    while (idx < 512u) {
      let t_loc = idx >> 5u;
      let k_loc = idx & 31u;
      let tok = tok_base + t_loc;
      if (tok < params.n_tokens && k_loc < k_len) {
        x_tile[idx] = x_in[tok * in_d + k_start + k_loc];
      } else {
        x_tile[idx] = 0.0;
      }
      idx += 64u;
    }
    workgroupBarrier();

    if (valid_out) {
      if (even_in && (k_len & 3u) == 0u) {
        let row_u32 = params.w_off_u32 + ((row_elem + k_start) >> 1u);
        let quads = k_len >> 2u;
        for (var q: u32 = 0u; q < quads; q++) {
          let p2 = q << 1u;
          let w01 = unpack2x16float(weights[row_u32 + p2]);
          let w23 = unpack2x16float(weights[row_u32 + p2 + 1u]);
          let wv = vec4<f32>(w01.x, w01.y, w23.x, w23.y);
          let k4 = q << 2u;
          for (var t: u32 = 0u; t < 16u; t++) {
            let xb = (t << 5u) + k4;
            let xv = vec4<f32>(x_tile[xb], x_tile[xb + 1u], x_tile[xb + 2u], x_tile[xb + 3u]);
            acc[t] += dot(xv, wv);
          }
        }
      } else if (even_in) {
        let row_u32 = params.w_off_u32 + ((row_elem + k_start) >> 1u);
        let pairs = k_len >> 1u;
        for (var p: u32 = 0u; p < pairs; p++) {
          let w2 = unpack2x16float(weights[row_u32 + p]);
          let k2 = p << 1u;
          for (var t: u32 = 0u; t < 16u; t++) {
            let xb = (t << 5u) + k2;
            acc[t] += x_tile[xb] * w2.x + x_tile[xb + 1u] * w2.y;
          }
        }
      } else {
        for (var j: u32 = 0u; j < k_len; j++) {
          let w = read_f16(params.w_off_u32, row_elem + k_start + j);
          for (var t: u32 = 0u; t < 16u; t++) {
            acc[t] += x_tile[(t << 5u) + j] * w;
          }
        }
      }
    }
    workgroupBarrier();
    k_start += 32u;
  }

  if (!valid_out) { return; }

  for (var t: u32 = 0u; t < 16u; t++) {
    let tok = tok_base + t;
    if (tok >= params.n_tokens) { break; }
    var v = acc[t];
    if (params.act_mode == 1u) {
      v = gelu_tanh(v);
    }
    let y_idx = tok * out_d + out_idx;
    if (params.add_res == 1u) {
      v += res_in[y_idx];
    }
    y_out[y_idx] = v;
  }
}
`;

const WGSL_BUILD_SE3_QK = /* wgsl */ `
struct Params {
  n_tokens: u32,
  n_heads: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> geo: array<f32>;     // [N, 16]
@group(0) @binding(2) var<storage, read> q_sem: array<f32>;   // [N, 152]
@group(0) @binding(3) var<storage, read> k_sem: array<f32>;   // [N, 152]
@group(0) @binding(4) var<storage, read> geo_q: array<f32>;   // [N, 56]
@group(0) @binding(5) var<storage, read> geo_k: array<f32>;   // [N, 56]
@group(0) @binding(6) var<storage, read_write> Q_out: array<f32>; // [N, 8, 32]
@group(0) @binding(7) var<storage, read_write> K_out: array<f32>; // [N, 8, 32]

fn softplus_clamped(x: f32) -> f32 {
  var sp = 0.0;
  if (x > 20.0) {
    sp = x;
  } else {
    sp = log(1.0 + exp(x));
  }
  return clamp(sp, 0.0, 20.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_tokens * params.n_heads;
  if (idx >= total) { return; }

  let i = idx / params.n_heads;
  let h = idx - i * params.n_heads;

  let gb = i * 16u;
  let p = vec3<f32>(geo[gb + 0u], geo[gb + 1u], geo[gb + 2u]);
  let pn2 = geo[gb + 3u];
  let T = vec3<f32>(geo[gb + 4u], geo[gb + 5u], geo[gb + 6u]);
  let B = vec3<f32>(geo[gb + 8u], geo[gb + 9u], geo[gb + 10u]);
  let N = vec3<f32>(geo[gb + 12u], geo[gb + 13u], geo[gb + 14u]);

  let out_base = idx * 32u;
  let sem_base = (i * params.n_heads + h) * 19u;
  let sem_scale = 0.22941573387056177; // 1 / sqrt(19)

  for (var d: u32 = 0u; d < 19u; d++) {
    Q_out[out_base + d] = q_sem[sem_base + d] * sem_scale;
    K_out[out_base + d] = k_sem[sem_base + d];
  }

  let gqb = (i * params.n_heads + h) * 7u;
  let alpha = softplus_clamped(geo_q[gqb + 0u]);
  let u_w = geo_q[gqb + 1u] * T + geo_q[gqb + 2u] * B + geo_q[gqb + 3u] * N;
  let m_w = geo_q[gqb + 4u] * T + geo_q[gqb + 5u] * B + geo_q[gqb + 6u] * N;

  let q0 = 2.0 * alpha * p + u_w;
  Q_out[out_base + 19u] = q0.x;
  Q_out[out_base + 20u] = q0.y;
  Q_out[out_base + 21u] = q0.z;
  Q_out[out_base + 22u] = -alpha * pn2 - dot(u_w, p);
  Q_out[out_base + 23u] = -alpha;
  Q_out[out_base + 24u] = p.x;
  Q_out[out_base + 25u] = p.y;
  Q_out[out_base + 26u] = p.z;
  Q_out[out_base + 27u] = 1.0;
  Q_out[out_base + 28u] = pn2;
  Q_out[out_base + 29u] = m_w.x;
  Q_out[out_base + 30u] = m_w.y;
  Q_out[out_base + 31u] = m_w.z;

  let gkb = (i * params.n_heads + h) * 7u;
  let beta = softplus_clamped(geo_k[gkb + 0u]);
  let v_w = geo_k[gkb + 1u] * T + geo_k[gkb + 2u] * B + geo_k[gkb + 3u] * N;
  let r_w = geo_k[gkb + 4u] * T + geo_k[gkb + 5u] * B + geo_k[gkb + 6u] * N;

  let k3 = 2.0 * beta * p + v_w;
  K_out[out_base + 19u] = p.x;
  K_out[out_base + 20u] = p.y;
  K_out[out_base + 21u] = p.z;
  K_out[out_base + 22u] = 1.0;
  K_out[out_base + 23u] = pn2;
  K_out[out_base + 24u] = k3.x;
  K_out[out_base + 25u] = k3.y;
  K_out[out_base + 26u] = k3.z;
  K_out[out_base + 27u] = -beta * pn2 - dot(v_w, p) + geo[gb + 7u];
  K_out[out_base + 28u] = -beta;
  K_out[out_base + 29u] = r_w.x;
  K_out[out_base + 30u] = r_w.y;
  K_out[out_base + 31u] = r_w.z;
}
`;

const WGSL_BUILD_SE3_V = /* wgsl */ `
struct Params {
  n_tokens: u32,
  n_heads: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> geo: array<f32>;     // [N, 16]
@group(0) @binding(2) var<storage, read> v_inv: array<f32>;   // [N, 208]
@group(0) @binding(3) var<storage, read_write> V_out: array<f32>; // [N, 8, 32]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_tokens * params.n_heads;
  if (idx >= total) { return; }

  let i = idx / params.n_heads;
  let h = idx - i * params.n_heads;
  let gb = i * 16u;
  let out_base = idx * 32u;
  let vb = (i * params.n_heads + h) * 26u;

  for (var d: u32 = 0u; d < 26u; d++) {
    V_out[out_base + d] = v_inv[vb + d];
  }
  V_out[out_base + 26u] = geo[gb + 0u];
  V_out[out_base + 27u] = geo[gb + 1u];
  V_out[out_base + 28u] = geo[gb + 2u];
  V_out[out_base + 29u] = geo[gb + 12u];
  V_out[out_base + 30u] = geo[gb + 13u];
  V_out[out_base + 31u] = geo[gb + 14u];
}
`;

// Block-Tiled FlashAttention (64 query tokens/WG x 16 K/V keys/tile, vec4 SIMD FMAs):
// Reduces workgroupBarrier() count by 80x and K/V VRAM traffic by 64x on Apple M1!
const WGSL_SE3_ATTENTION = /* wgsl */ `
struct Params {
  n_tokens: u32,
  n_heads: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> geo: array<f32>;   // [N, 16]
@group(0) @binding(2) var<storage, read> Q_in: array<f32>;  // [N, 8, 32]
@group(0) @binding(3) var<storage, read> K_in: array<f32>;  // [N, 8, 32]
@group(0) @binding(4) var<storage, read> V_in: array<f32>;  // [N, 8, 32]
@group(0) @binding(5) var<storage, read_write> out_merged: array<f32>; // [N, 256]

// 16 keys x 8 vec4s = 128 vec4s (2 KB each)
var<workgroup> K_tile: array<vec4<f32>, 128>;
var<workgroup> V_tile: array<vec4<f32>, 128>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let h = wg.x;
  let i = wg.y * 64u + lid.x;
  let tid = lid.x;
  let N_tok = params.n_tokens;
  let H = params.n_heads;
  let valid_i = i < N_tok && h < H;

  // Load query Q[i, h, 0..31] into 8 vec4 registers once
  var q0 = vec4<f32>(0.0); var q1 = vec4<f32>(0.0); var q2 = vec4<f32>(0.0); var q3 = vec4<f32>(0.0);
  var q4 = vec4<f32>(0.0); var q5 = vec4<f32>(0.0); var q6 = vec4<f32>(0.0); var q7 = vec4<f32>(0.0);
  if (valid_i) {
    let qb = (i * H + h) * 32u;
    q0 = vec4<f32>(Q_in[qb + 0u],  Q_in[qb + 1u],  Q_in[qb + 2u],  Q_in[qb + 3u]);
    q1 = vec4<f32>(Q_in[qb + 4u],  Q_in[qb + 5u],  Q_in[qb + 6u],  Q_in[qb + 7u]);
    q2 = vec4<f32>(Q_in[qb + 8u],  Q_in[qb + 9u],  Q_in[qb + 10u], Q_in[qb + 11u]);
    q3 = vec4<f32>(Q_in[qb + 12u], Q_in[qb + 13u], Q_in[qb + 14u], Q_in[qb + 15u]);
    q4 = vec4<f32>(Q_in[qb + 16u], Q_in[qb + 17u], Q_in[qb + 18u], Q_in[qb + 19u]);
    q5 = vec4<f32>(Q_in[qb + 20u], Q_in[qb + 21u], Q_in[qb + 22u], Q_in[qb + 23u]);
    q6 = vec4<f32>(Q_in[qb + 24u], Q_in[qb + 25u], Q_in[qb + 26u], Q_in[qb + 27u]);
    q7 = vec4<f32>(Q_in[qb + 28u], Q_in[qb + 29u], Q_in[qb + 30u], Q_in[qb + 31u]);
  }

  var m_max = -1e30;
  var l_sum = 0.0;
  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
  var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0); var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);

  var j_base: u32 = 0u;
  while (j_base < N_tok) {
    let tile_len = min(16u, N_tok - j_base);
    // 64 threads load 128 vec4s (2 vec4s per thread!) into K_tile and V_tile
    for (var s: u32 = 0u; s < 2u; s++) {
      let slot = tid + s * 64u; // 0..127
      let kj_loc = slot >> 3u;  // 0..15
      let v_idx = slot & 7u;    // 0..7
      let j = j_base + kj_loc;
      if (j < N_tok) {
        let kb = (j * H + h) * 32u + (v_idx << 2u);
        K_tile[slot] = vec4<f32>(K_in[kb], K_in[kb + 1u], K_in[kb + 2u], K_in[kb + 3u]);
        V_tile[slot] = vec4<f32>(V_in[kb], V_in[kb + 1u], V_in[kb + 2u], V_in[kb + 3u]);
      }
    }
    workgroupBarrier();

    if (valid_i) {
      for (var kj: u32 = 0u; kj < tile_len; kj++) {
        let tb = kj << 3u;
        let score =
          dot(q0, K_tile[tb + 0u]) + dot(q1, K_tile[tb + 1u]) +
          dot(q2, K_tile[tb + 2u]) + dot(q3, K_tile[tb + 3u]) +
          dot(q4, K_tile[tb + 4u]) + dot(q5, K_tile[tb + 5u]) +
          dot(q6, K_tile[tb + 6u]) + dot(q7, K_tile[tb + 7u]);

        if (j_base == 0u && kj == 0u) {
          m_max = score;
          l_sum = 1.0;
          a0 = V_tile[tb + 0u]; a1 = V_tile[tb + 1u]; a2 = V_tile[tb + 2u]; a3 = V_tile[tb + 3u];
          a4 = V_tile[tb + 4u]; a5 = V_tile[tb + 5u]; a6 = V_tile[tb + 6u]; a7 = V_tile[tb + 7u];
        } else if (score <= m_max) {
          // Fast path (>95% of keys): single exp(), ZERO rescale multiplies on a0..a7!
          let w = exp(max(score - m_max, -80.0));
          l_sum += w;
          a0 += V_tile[tb + 0u] * w;
          a1 += V_tile[tb + 1u] * w;
          a2 += V_tile[tb + 2u] * w;
          a3 += V_tile[tb + 3u] * w;
          a4 += V_tile[tb + 4u] * w;
          a5 += V_tile[tb + 5u] * w;
          a6 += V_tile[tb + 6u] * w;
          a7 += V_tile[tb + 7u] * w;
        } else {
          // New max (<5% of keys): single exp(), w == 1.0!
          let rescale = exp(max(m_max - score, -80.0));
          l_sum = l_sum * rescale + 1.0;
          a0 = a0 * rescale + V_tile[tb + 0u];
          a1 = a1 * rescale + V_tile[tb + 1u];
          a2 = a2 * rescale + V_tile[tb + 2u];
          a3 = a3 * rescale + V_tile[tb + 3u];
          a4 = a4 * rescale + V_tile[tb + 4u];
          a5 = a5 * rescale + V_tile[tb + 5u];
          a6 = a6 * rescale + V_tile[tb + 6u];
          a7 = a7 * rescale + V_tile[tb + 7u];
          m_max = score;
        }
      }
    }
    workgroupBarrier();
    j_base += 16u;
  }

  if (!valid_i) { return; }

  let inv_l = 1.0 / max(l_sum, 1e-12);
  a0 *= inv_l; a1 *= inv_l; a2 *= inv_l; a3 *= inv_l;
  a4 *= inv_l; a5 *= inv_l; a6 *= inv_l; a7 *= inv_l;

  let ob = i * (H * 32u) + h * 32u;
  out_merged[ob + 0u]  = a0.x; out_merged[ob + 1u]  = a0.y; out_merged[ob + 2u]  = a0.z; out_merged[ob + 3u]  = a0.w;
  out_merged[ob + 4u]  = a1.x; out_merged[ob + 5u]  = a1.y; out_merged[ob + 6u]  = a1.z; out_merged[ob + 7u]  = a1.w;
  out_merged[ob + 8u]  = a2.x; out_merged[ob + 9u]  = a2.y; out_merged[ob + 10u] = a2.z; out_merged[ob + 11u] = a2.w;
  out_merged[ob + 12u] = a3.x; out_merged[ob + 13u] = a3.y; out_merged[ob + 14u] = a3.z; out_merged[ob + 15u] = a3.w;
  out_merged[ob + 16u] = a4.x; out_merged[ob + 17u] = a4.y; out_merged[ob + 18u] = a4.z; out_merged[ob + 19u] = a4.w;
  out_merged[ob + 20u] = a5.x; out_merged[ob + 21u] = a5.y; out_merged[ob + 22u] = a5.z; out_merged[ob + 23u] = a5.w;
  out_merged[ob + 24u] = a6.x; out_merged[ob + 25u] = a6.y;

  // Exact SE(3)-invariant local-frame projection of (pos_bar - p_hat_i) and nrm_bar
  let gb = i * 16u;
  let p_i = vec3<f32>(geo[gb + 0u], geo[gb + 1u], geo[gb + 2u]);
  let T = vec3<f32>(geo[gb + 4u], geo[gb + 5u], geo[gb + 6u]);
  let B = vec3<f32>(geo[gb + 8u], geo[gb + 9u], geo[gb + 10u]);
  let N = vec3<f32>(geo[gb + 12u], geo[gb + 13u], geo[gb + 14u]);
  let dp = vec3<f32>(a6.z, a6.w, a7.x) - p_i;
  let nb = vec3<f32>(a7.y, a7.z, a7.w);
  out_merged[ob + 26u] = dot(dp, T);
  out_merged[ob + 27u] = dot(dp, B);
  out_merged[ob + 28u] = dot(dp, N);
  out_merged[ob + 29u] = dot(nb, T);
  out_merged[ob + 30u] = dot(nb, B);
  out_merged[ob + 31u] = dot(nb, N);
}
`;

const WGSL_DECODE_SH = /* wgsl */ `
struct Params {
  n_tokens: u32,
  multi_bounce: u32,
  bounce_count: f32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> raw_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> sh_out: array<f32>;

fn softplus_i0(x: f32) -> f32 {
  var sp = 0.0;
  if (x > 20.0) {
    sp = x;
  } else {
    sp = log(1.0 + exp(x));
  }
  return min(sp, 64.0);
}

fn decode_sh_elem(base: u32, k: u32, i0r: f32, i0g: f32, i0b: f32) -> f32 {
  if (k == 0u) { return i0r; }
  if (k == 1u) { return i0g; }
  if (k == 2u) { return i0b; }
  if (k == 3u) { return i0r * tanh(clamp(raw_in[base + 3u], -15.0, 15.0)); }
  if (k == 4u) { return i0g * tanh(clamp(raw_in[base + 4u], -15.0, 15.0)); }
  if (k == 5u) { return i0b * tanh(clamp(raw_in[base + 5u], -15.0, 15.0)); }
  if (k == 6u) { return i0r * tanh(clamp(raw_in[base + 6u], -15.0, 15.0)); }
  if (k == 7u) { return i0g * tanh(clamp(raw_in[base + 7u], -15.0, 15.0)); }
  if (k == 8u) { return i0b * tanh(clamp(raw_in[base + 8u], -15.0, 15.0)); }
  if (k == 9u) { return i0r / (1.0 + exp(-clamp(raw_in[base + 9u], -20.0, 20.0))); }
  if (k == 10u) { return i0g / (1.0 + exp(-clamp(raw_in[base + 10u], -20.0, 20.0))); }
  return i0b / (1.0 + exp(-clamp(raw_in[base + 11u], -20.0, 20.0)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n_tokens) { return; }
  let dst = i * 12u;

  if (params.multi_bounce == 0u) {
    let i0r = softplus_i0(raw_in[dst + 0u]);
    let i0g = softplus_i0(raw_in[dst + 1u]);
    let i0b = softplus_i0(raw_in[dst + 2u]);
    for (var k: u32 = 0u; k < 12u; k++) {
      sh_out[dst + k] = decode_sh_elem(dst, k, i0r, i0g, i0b);
    }
    return;
  }

  let b1 = i * 36u;
  let b2 = b1 + 12u;
  let b3 = b1 + 24u;
  let w2 = clamp(params.bounce_count - 1.0, 0.0, 1.0);
  let w3 = clamp(params.bounce_count - 2.0, 0.0, 1.0);

  let r1 = softplus_i0(raw_in[b1 + 0u]); let g1 = softplus_i0(raw_in[b1 + 1u]); let bl1 = softplus_i0(raw_in[b1 + 2u]);
  let r2 = softplus_i0(raw_in[b2 + 0u]); let g2 = softplus_i0(raw_in[b2 + 1u]); let bl2 = softplus_i0(raw_in[b2 + 2u]);
  let r3 = softplus_i0(raw_in[b3 + 0u]); let g3 = softplus_i0(raw_in[b3 + 1u]); let bl3 = softplus_i0(raw_in[b3 + 2u]);

  for (var k: u32 = 0u; k < 12u; k++) {
    let v1 = decode_sh_elem(b1, k, r1, g1, bl1);
    let v2 = decode_sh_elem(b2, k, r2, g2, bl2);
    let v3 = decode_sh_elem(b3, k, r3, g3, bl3);
    sh_out[dst + k] = v1 + w2 * v2 + w3 * v3;
  }
}
`;

// ---------------------------------------------------------------------------
// RadiantComputeEngine Class
// ---------------------------------------------------------------------------

/**
 * World-space per-patch packed GPU layout for `radiantGINode.ts` (6 vec4s = 24 floats/patch):
 *   vec4[0]: pos.xyz, patchRadius
 *   vec4[1]: N.xyz,   area
 *   vec4[2]: I0.rgb,  I1_N.r
 *   vec4[3]: I1_world_R.xyz, I1_N.g
 *   vec4[4]: I1_world_G.xyz, I1_N.b
 *   vec4[5]: I1_world_B.xyz, gtErrorScalar (optional diagnostic)
 */
export const WORLD_PATCH_VEC4S = 6;
export const MAX_PATCHES_PER_OBJECT = 64;
export const MAX_RENDER_OBJECTS = 48;
export const MAX_RENDER_PATCHES = 2048;

export class RadiantComputeEngine {
  public device: GPUDevice | null;
  public isCPUFallback: boolean;
  /** True whenever the Three.js renderer is using WebGLBackend (must read DataTexture, not storage()). */
  public readonly useTextureLookup: boolean;
  public isQ8Turbo = false;
  public hasF16 = false;
  private rendererBackend: any = null;
  public header!: RDGIHeader;
  private layout!: WeightLayout;
  private q8Layout: Q8WeightLayout | null = null;
  private weightsBuf: GPUBuffer | null = null;
  private f32AuxBuf: GPUBuffer | null = null;
  private cpuWeightsF32: Float32Array | null = null;
  private cpuWorker: Worker | null = null;
  private cpuWorkerReady: Promise<void> | null = null;
  private cpuWorkerSeq = 0;
  private rawWeightsBuffer: ArrayBuffer | null = null;

  // Compute pipelines (RDGI FP16 + RDG8 Turbo INT8 + On-GPU World SH Packer)
  private pipeCentroid!: GPUComputePipeline;
  private pipePrep!: GPUComputePipeline;
  private pipeRMS!: GPUComputePipeline;
  private pipeLinear!: GPUComputePipeline;
  private pipeBuildQK!: GPUComputePipeline;
  private pipeBuildV!: GPUComputePipeline;
  private pipeAttn!: GPUComputePipeline;
  private pipeDecode!: GPUComputePipeline;
  private pipeRMSF32!: GPUComputePipeline;
  private pipeLinearF32!: GPUComputePipeline;
  private pipeLinearQ8!: GPUComputePipeline;
  private pipeBuildQKVFused!: GPUComputePipeline;
  private pipePackWorldSH!: GPUComputePipeline;

  // Dynamic capacity workspace buffers
  private maxTokens = 0;
  private tokensBuf!: GPUBuffer;
  private centroidBuf!: GPUBuffer;
  private geoBuf!: GPUBuffer;
  private intrBuf!: GPUBuffer;
  private xABuf!: GPUBuffer;
  private xBBuf!: GPUBuffer;
  private xNormBuf!: GPUBuffer;
  private ffnBuf!: GPUBuffer;
  private qkvGeoBuf!: GPUBuffer;
  private qSemBuf!: GPUBuffer;
  private kSemBuf!: GPUBuffer;
  private vInvBuf!: GPUBuffer;
  private geoQBuf!: GPUBuffer;
  private geoKBuf!: GPUBuffer;
  private qBuf!: GPUBuffer;
  private kBuf!: GPUBuffer;
  private vBuf!: GPUBuffer;
  private attnMergedBuf!: GPUBuffer;
  private rawHeadBuf!: GPUBuffer;
  private shOutBuf!: GPUBuffer;
  private worldPatchGpuBuf!: GPUBuffer;
  private readbackBuf!: GPUBuffer;
  private paramsRingBuf!: GPUBuffer;

  // Uniform slot stride (WebGPU spec minUniformBufferOffsetAlignment <= 256)
  private readonly PARAM_STRIDE_BYTES = 256;
  private readonly MAX_PARAM_SLOTS = 256;
  private paramsCPU = new Uint32Array((256 * 256) / 4);
  private paramsCPUF32 = new Float32Array(this.paramsCPU.buffer);

  // Shared Three.js TSL StorageBufferAttribute (WebGPU) + RGBA32F DataTexture (WebGL2 fallback)
  public readonly worldPatchArray = new Float32Array(MAX_RENDER_PATCHES * WORLD_PATCH_VEC4S * 4);
  public readonly worldPatchTexArray = new Float32Array(
    MAX_RENDER_OBJECTS * MAX_PATCHES_PER_OBJECT * WORLD_PATCH_VEC4S * 4,
  );
  public readonly worldPatchStorage: THREE.StorageBufferAttribute;
  public readonly worldPatchTexture: THREE.DataTexture;
  public activeTokenCount = 0;
  public bounceCount = 3.0;
  public lastComputeMs = 0;
  public lastLocalSH: Float32Array = new Float32Array(0);
  public lastRawBounces: Float32Array = new Float32Array(0);
  public lastWorldTokens: Float32Array = new Float32Array(0);
  private lastObjectRanges: ObjectPatchRange[] | null = null;

  // Cached GPUBindGroups (rebuilt only when capacity or weights change — zero per-frame allocation!)
  private cachedBGs: {
    centroid: GPUBindGroup;
    prep: GPUBindGroup;
    stem1: GPUBindGroup;
    stem2: GPUBindGroup;
    rmsX: GPUBindGroup;
    qSem: GPUBindGroup;
    kSem: GPUBindGroup;
    vInv: GPUBindGroup;
    geoQ: GPUBindGroup;
    geoK: GPUBindGroup;
    buildQK: GPUBindGroup;
    buildV: GPUBindGroup;
    attn: GPUBindGroup;
    wO: GPUBindGroup;
    rmsXB: GPUBindGroup;
    fc1: GPUBindGroup;
    fc2: GPUBindGroup;
    head1: GPUBindGroup;
    head2: GPUBindGroup;
    decode: GPUBindGroup;
    packWorldSH: GPUBindGroup;
    // RDG8 Turbo Q8 bind groups
    q8Stem1?: GPUBindGroup;
    q8Stem2?: GPUBindGroup;
    q8RmsX?: GPUBindGroup;
    q8QkvGeo?: GPUBindGroup;
    q8BuildQKV?: GPUBindGroup;
    q8WO?: GPUBindGroup;
    q8RmsXB?: GPUBindGroup;
    q8Fc1?: GPUBindGroup;
    q8Fc2?: GPUBindGroup;
    q8Head1?: GPUBindGroup;
    q8Head2?: GPUBindGroup;
  } | null = null;

  private isBusy = false;
  private inferQueue: Promise<unknown> = Promise.resolve();

  constructor(device: GPUDevice | null, useTextureLookup = false, rendererBackend: any = null) {
    this.device = device;
    this.isCPUFallback = device === null;
    this.useTextureLookup = useTextureLookup || device === null;
    this.rendererBackend = rendererBackend;
    this.hasF16 = Boolean(device && device.features && device.features.has('shader-f16'));
    this.worldPatchStorage = new THREE.StorageBufferAttribute(this.worldPatchArray, 4);
    this.worldPatchTexture = new THREE.DataTexture(
      this.worldPatchTexArray,
      MAX_PATCHES_PER_OBJECT * WORLD_PATCH_VEC4S, // 384 vec4s per object row (supports up to 64 patches/object!)
      MAX_RENDER_OBJECTS, // 48 objects max
      THREE.RGBAFormat,
      THREE.FloatType,
    );
    this.worldPatchTexture.minFilter = THREE.NearestFilter;
    this.worldPatchTexture.magFilter = THREE.NearestFilter;
    this.worldPatchTexture.generateMipmaps = false;
    this.worldPatchTexture.needsUpdate = true;
    if (this.device) {
      try {
        this._createPipelines();
      } catch (err) {
        console.warn('[RadiantComputeEngine] WebGPU compute pipeline init failed, falling back to CPU:', err);
        this.device = null;
        this.isCPUFallback = true;
      }
    }
  }

  public static async createUnloadedFromRenderer(
    renderer: THREE.WebGPURenderer,
  ): Promise<RadiantComputeEngine> {
    const backend = (renderer as any).backend;
    const isWebGLRenderer = Boolean(backend?.isWebGLBackend) || !backend?.device;
    let device: GPUDevice | null = backend?.device ?? null;
    if (!device && typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          const reqFeatures: GPUFeatureName[] = [];
          if (adapter.features.has('shader-f16')) reqFeatures.push('shader-f16');
          device = await adapter.requestDevice({ requiredFeatures: reqFeatures });
        }
      } catch {
        device = null;
      }
    }
    return new RadiantComputeEngine(device, isWebGLRenderer, backend);
  }

  public static async createFromRenderer(
    renderer: THREE.WebGPURenderer,
    weightsUrl: string,
    onProgress?: (loadedBytes: number, totalBytes: number) => void,
  ): Promise<RadiantComputeEngine> {
    const engine = await RadiantComputeEngine.createUnloadedFromRenderer(renderer);
    await engine.loadWeightsFromUrl(weightsUrl, onProgress);
    return engine;
  }

  private _initCpuWorker(buffer: ArrayBuffer, isQ8: boolean, q8U32Len = 0, f32Len = 0): void {
    if (typeof Worker === 'undefined') return;
    try {
      if (this.cpuWorker) {
        this.cpuWorker.terminate();
        this.cpuWorker = null;
      }
      const w = new Worker(new URL('./cpuWorker.ts', import.meta.url), { type: 'module' });
      const id = ++this.cpuWorkerSeq;
      this.cpuWorker = w;
      const copyBuf = buffer.slice(0);
      this.cpuWorkerReady = new Promise<void>((resolve, reject) => {
        const onMsg = (ev: MessageEvent) => {
          if (ev.data?.id !== id) return;
          w.removeEventListener('message', onMsg);
          if (ev.data?.type === 'init_done') resolve();
          else reject(new Error(ev.data?.error || 'CPU worker init failed'));
        };
        w.addEventListener('message', onMsg);
        w.postMessage(
          { type: 'init', id, buffer: copyBuf, header: this.header, isQ8, q8U32Len, f32Len, layout: isQ8 ? null : this.layout },
          [copyBuf],
        );
      }).catch(() => {
        this.cpuWorker = null;
        this.cpuWorkerReady = null;
      });
    } catch {
      this.cpuWorker = null;
      this.cpuWorkerReady = null;
    }
  }

  private _runCpuWorker(tokens: Float32Array, nTokens: number): Promise<{ result: Float32Array; computeMs: number } | null> {
    if (!this.cpuWorker || !this.cpuWorkerReady) return Promise.resolve(null);
    const w = this.cpuWorker;
    const ready = this.cpuWorkerReady;
    return ready.then(() => {
      if (this.cpuWorker !== w) return null;
      return new Promise<{ result: Float32Array; computeMs: number } | null>((resolve) => {
        const id = ++this.cpuWorkerSeq;
        const onMsg = (ev: MessageEvent) => {
          if (ev.data?.id !== id) return;
          w.removeEventListener('message', onMsg);
          if (ev.data?.type === 'infer_done') {
            resolve({ result: new Float32Array(ev.data.result), computeMs: ev.data.computeMs });
          } else {
            resolve(null);
          }
        };
        w.addEventListener('message', onMsg);
        const tokCopy = new Float32Array(tokens.subarray(0, nTokens * TOKEN_STRIDE));
        w.postMessage({ type: 'infer', id, tokens: tokCopy.buffer, nTokens }, [tokCopy.buffer]);
      });
    });
  }

  public get isWeightsLoaded(): boolean {
    return this.isCPUFallback ? (this.cpuWeightsF32 !== null || this.cpuWorker !== null) : this.weightsBuf !== null;
  }

  public static async createStandalone(weightsUrlOrBuffer: string | ArrayBuffer): Promise<RadiantComputeEngine> {
    if (!navigator.gpu) throw new Error('WebGPU navigator.gpu unavailable');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU GPUAdapter available');
    const device = await adapter.requestDevice();
    const engine = new RadiantComputeEngine(device);
    if (typeof weightsUrlOrBuffer === 'string') {
      await engine.loadWeightsFromUrl(weightsUrlOrBuffer);
    } else {
      engine.loadWeightsFromBuffer(weightsUrlOrBuffer);
    }
    return engine;
  }

  private _createPipelines(): void {
    const device = this.device!;
    const mk = (code: string, dynamicUniform: boolean) => {
      const module = device.createShaderModule({ code });
      if (!dynamicUniform) {
        return device.createComputePipeline({
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
      }
      return module;
    };

    // Explicit BindGroupLayouts with hasDynamicOffset=true on binding 0 for single-encoder ring dispatch!
    const mkDynamicPipe = (code: string, storageEntries: { binding: number; type: GPUBufferBindingType }[]) => {
      const bgl = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 32 },
          },
          ...storageEntries.map((e) => ({
            binding: e.binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: e.type },
          })),
        ],
      });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const module = device.createShaderModule({ code });
      return device.createComputePipeline({
        layout,
        compute: { module, entryPoint: 'main' },
      });
    };

    this.pipeCentroid = mkDynamicPipe(WGSL_CENTROID, [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'storage' },
    ]);
    this.pipePrep = mkDynamicPipe(WGSL_PREP_GEO_INTR, [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'storage' },
      { binding: 4, type: 'storage' },
    ]);
    this.pipeRMS = mkDynamicPipe(WGSL_RMS_NORM, [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'storage' },
    ]);
    this.pipeLinear = mkDynamicPipe(WGSL_LINEAR, [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'read-only-storage' },
      { binding: 4, type: 'storage' },
    ]);
    this.pipeBuildQK = mkDynamicPipe(WGSL_BUILD_SE3_QK, [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'read-only-storage' },
      { binding: 4, type: 'read-only-storage' },
      { binding: 5, type: 'read-only-storage' },
      { binding: 6, type: 'storage' },
      { binding: 7, type: 'storage' },
    ]);
    this.pipeBuildV = mkDynamicPipe(WGSL_BUILD_SE3_V, [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'storage' },
    ]);
    this.pipeAttn = mkDynamicPipe(buildAttentionTurboWGSL(this.hasF16), [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'read-only-storage' },
      { binding: 4, type: 'read-only-storage' },
      { binding: 5, type: 'storage' },
    ]);
    this.pipeDecode = mkDynamicPipe(WGSL_DECODE_SH, [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'storage' },
    ]);
    this.pipeRMSF32 = mkDynamicPipe(WGSL_RMS_NORM_F32, [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'storage' },
    ]);
    this.pipeLinearF32 = mkDynamicPipe(WGSL_LINEAR_F32, [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'read-only-storage' },
      { binding: 4, type: 'storage' },
    ]);
    this.pipeLinearQ8 = mkDynamicPipe(buildLinearQ8WGSL(this.hasF16), [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'read-only-storage' },
      { binding: 4, type: 'read-only-storage' },
      { binding: 5, type: 'storage' },
    ]);
    this.pipeBuildQKVFused = mkDynamicPipe(WGSL_BUILD_SE3_QKV_FUSED, [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'storage' },
      { binding: 4, type: 'storage' },
      { binding: 5, type: 'storage' },
    ]);
    this.pipePackWorldSH = mkDynamicPipe(WGSL_PACK_WORLD_SH, [
      { binding: 1, type: 'read-only-storage' },
      { binding: 2, type: 'read-only-storage' },
      { binding: 3, type: 'storage' },
    ]);

    void mk;
    this.paramsRingBuf = device.createBuffer({
      size: this.PARAM_STRIDE_BYTES * this.MAX_PARAM_SLOTS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  public async loadWeightsFromUrl(
    url: string,
    onProgress?: (loadedBytes: number, totalBytes: number) => void,
  ): Promise<void> {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to fetch RDGI weights from ${url}: HTTP ${resp.status}`);
    }
    const total = Number(resp.headers.get('content-length') || '0');
    if (resp.body && onProgress) {
      const reader = resp.body.getReader();
      const chunks: Uint8Array[] = [];
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          loaded += value.byteLength;
          onProgress(loaded, total);
        }
      }
      const merged = new Uint8Array(loaded);
      let off = 0;
      for (const c of chunks) {
        merged.set(c, off);
        off += c.byteLength;
      }
      this.loadWeightsFromBuffer(merged.buffer);
    } else {
      const buf = await resp.arrayBuffer();
      if (onProgress) onProgress(buf.byteLength, buf.byteLength);
      this.loadWeightsFromBuffer(buf);
    }
  }

  public loadWeightsFromBuffer(buffer: ArrayBuffer): void {
    const dv = new DataView(buffer);
    const magic =
      String.fromCharCode(dv.getUint8(0)) +
      String.fromCharCode(dv.getUint8(1)) +
      String.fromCharCode(dv.getUint8(2)) +
      String.fromCharCode(dv.getUint8(3));

    if (magic === 'RDG8') {
      this._loadRDG8Buffer(buffer, dv);
      return;
    }
    if (magic !== 'RDGI') {
      throw new Error(`Invalid RDGI/RDG8 weight binary magic: "${magic}"`);
    }
    this.isQ8Turbo = false;
    this.q8Layout = null;

    const flags = dv.getUint32(28, true);
    this.header = {
      version: dv.getUint32(4, true),
      dModel: dv.getUint32(8, true),
      nHeads: dv.getUint32(12, true),
      nLayers: dv.getUint32(16, true),
      dFF: dv.getUint32(20, true),
      nTensors: dv.getUint32(24, true),
      flags,
      posScale: dv.getFloat32(32, true),
      hasAreaPrior: (flags & 2) !== 0,
      isMultiBounce: (flags & 4) !== 0,
    };

    const { dModel, nHeads, nLayers, dFF } = this.header;
    let elemOff = 0;
    const alloc = (count: number): number => {
      if ((elemOff & 1) !== 0 || (count & 1) !== 0) {
        throw new Error(`Unaligned RDGI FP16 tensor offset=${elemOff}, count=${count}`);
      }
      const u32Off = elemOff >> 1;
      elemOff += count;
      return u32Off;
    };

    const stem1W = alloc(dModel * 13);
    const stem1B = alloc(dModel);
    const stem2W = alloc(dModel * dModel);
    const stem2B = alloc(dModel);

    const layers: LayerOffsets[] = [];
    for (let l = 0; l < nLayers; l++) {
      layers.push({
        norm1W: alloc(dModel),
        qSemW: alloc(nHeads * 19 * dModel),
        qSemB: alloc(nHeads * 19),
        kSemW: alloc(nHeads * 19 * dModel),
        kSemB: alloc(nHeads * 19),
        vInvW: alloc(nHeads * 26 * dModel),
        vInvB: alloc(nHeads * 26),
        geoQW: alloc(nHeads * 7 * dModel),
        geoQB: alloc(nHeads * 7),
        geoKW: alloc(nHeads * 7 * dModel),
        geoKB: alloc(nHeads * 7),
        wOW: alloc(dModel * dModel),
        wOB: alloc(dModel),
        norm2W: alloc(dModel),
        fc1W: alloc(dFF * dModel),
        fc1B: alloc(dFF),
        fc2W: alloc(dModel * dFF),
        fc2B: alloc(dModel),
      });
    }

    const head2OutDim = this.header.isMultiBounce ? 36 : 12;
    const finalNormW = alloc(dModel);
    const head1W = alloc(dModel * dModel);
    const head1B = alloc(dModel);
    const head2W = alloc(head2OutDim * dModel);
    const head2B = alloc(head2OutDim);

    const expectedBytes = 64 + elemOff * 2;
    if (buffer.byteLength !== expectedBytes) {
      throw new Error(`RDGI size mismatch: got ${buffer.byteLength}, expected ${expectedBytes}`);
    }

    this.layout = {
      stem1W, stem1B, stem2W, stem2B, layers,
      finalNormW, head1W, head1B, head2W, head2B,
    };

    this.rawWeightsBuffer = buffer;
    if (this.isCPUFallback || !this.device) {
      this.cpuWeightsF32 = unpackRDGIWeightsF32(buffer);
      this._initCpuWorker(buffer, false);
      return;
    }

    const payloadBytes = new Uint8Array(buffer, 64);
    if (this.weightsBuf) this.weightsBuf.destroy();
    this.weightsBuf = this.device.createBuffer({
      size: payloadBytes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.weightsBuf, 0, payloadBytes);
    this.cachedBGs = null;
    this._ensureCapacity(MAX_RENDER_PATCHES);
    this._rebuildBindGroups();
  }

  private _loadRDG8Buffer(buffer: ArrayBuffer, dv: DataView): void {
    this.isQ8Turbo = true;
    const q8U32Len = dv.getUint32(24, true);
    const f32Len = dv.getUint32(28, true);
    const featFlags = dv.getUint32(36, true);
    this.header = {
      version: dv.getUint32(4, true),
      dModel: dv.getUint32(8, true),
      nHeads: dv.getUint32(12, true),
      nLayers: dv.getUint32(16, true),
      dFF: dv.getUint32(20, true),
      nTensors: q8U32Len,
      flags: 8 | (featFlags << 8),
      posScale: dv.getFloat32(32, true),
      hasAreaPrior: (featFlags & 1) !== 0,
      isMultiBounce: (featFlags & 2) !== 0,
    };

    const expectedBytes = 64 + q8U32Len * 4 + f32Len * 4;
    if (buffer.byteLength !== expectedBytes) {
      throw new Error(`RDG8 size mismatch: got ${buffer.byteLength}, expected ${expectedBytes}`);
    }

    const { dModel, nHeads, nLayers, dFF } = this.header;
    const qkvRows = nHeads * 78;
    let q8Off = 0;
    let auxOff = 0;
    const allocQ8 = (rows: number, cols: number): number => {
      const off = q8Off;
      q8Off += (rows * cols) >> 2;
      return off;
    };
    const allocAux = (count: number): number => {
      const off = auxOff;
      auxOff += count;
      return off;
    };

    const stem1W = allocAux(dModel * 13);
    const stem1B = allocAux(dModel);
    const stem2W = allocQ8(dModel, dModel);
    const stem2S = allocAux(dModel);
    const stem2B = allocAux(dModel);

    const layers: Q8LayerOffsets[] = [];
    for (let l = 0; l < nLayers; l++) {
      const norm1W = allocAux(dModel);
      const qkvW = allocQ8(qkvRows, dModel);
      const qkvS = allocAux(qkvRows);
      const qkvB = allocAux(qkvRows);
      const wOW = allocQ8(dModel, dModel);
      const wOS = allocAux(dModel);
      const wOB = allocAux(dModel);
      const norm2W = allocAux(dModel);
      const fc1W = allocQ8(dFF, dModel);
      const fc1S = allocAux(dFF);
      const fc1B = allocAux(dFF);
      const fc2W = allocQ8(dModel, dFF);
      const fc2S = allocAux(dModel);
      const fc2B = allocAux(dModel);
      layers.push({ norm1W, qkvW, qkvS, qkvB, wOW, wOS, wOB, norm2W, fc1W, fc1S, fc1B, fc2W, fc2S, fc2B });
    }

    const head2OutDim = this.header.isMultiBounce ? 36 : 12;
    const finalNormW = allocAux(dModel);
    const head1W = allocQ8(dModel, dModel);
    const head1S = allocAux(dModel);
    const head1B = allocAux(dModel);
    const head2W = allocAux(head2OutDim * dModel);
    const head2B = allocAux(head2OutDim);

    if (q8Off !== q8U32Len || auxOff !== f32Len) {
      throw new Error(`RDG8 offset mismatch: q8=${q8Off}/${q8U32Len}, aux=${auxOff}/${f32Len}`);
    }

    this.q8Layout = {
      stem1W, stem1B, stem2W, stem2S, stem2B, layers,
      finalNormW, head1W, head1S, head1B, head2W, head2B,
    };

    this.rawWeightsBuffer = buffer;
    if (this.isCPUFallback || !this.device) {
      const unpacked = unpackRDG8ToCanonicalF32(buffer, this.header, q8U32Len, f32Len);
      this.cpuWeightsF32 = unpacked.weights;
      this.layout = unpacked.layout;
      this._initCpuWorker(buffer, true, q8U32Len, f32Len);
      return;
    }

    const q8Bytes = new Uint8Array(buffer, 64, q8U32Len * 4);
    const auxBytes = new Uint8Array(buffer, 64 + q8U32Len * 4, f32Len * 4);
    if (this.weightsBuf) this.weightsBuf.destroy();
    if (this.f32AuxBuf) this.f32AuxBuf.destroy();
    this.weightsBuf = this.device.createBuffer({
      size: q8Bytes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.f32AuxBuf = this.device.createBuffer({
      size: auxBytes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.weightsBuf, 0, q8Bytes);
    this.device.queue.writeBuffer(this.f32AuxBuf, 0, auxBytes);
    this.cachedBGs = null;
    this._ensureCapacity(MAX_RENDER_PATCHES);
    this._rebuildBindGroups();
  }

  private _rebuildBindGroups(): void {
    if (!this.device || !this.weightsBuf || this.maxTokens === 0) return;
    const device = this.device;
    const bg = (pipe: GPUComputePipeline, entries: GPUBuffer[]) =>
      device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.paramsRingBuf, offset: 0, size: 32 } },
          ...entries.map((buffer, idx) => ({ binding: idx + 1, resource: { buffer } })),
        ],
      });
    this.cachedBGs = {
      centroid: bg(this.pipeCentroid, [this.tokensBuf, this.centroidBuf]),
      prep: bg(this.pipePrep, [this.tokensBuf, this.centroidBuf, this.geoBuf, this.intrBuf]),
      stem1: bg(this.pipeLinear, [this.intrBuf, this.weightsBuf, this.centroidBuf, this.xBBuf]),
      stem2: bg(this.pipeLinear, [this.xBBuf, this.weightsBuf, this.centroidBuf, this.xABuf]),
      rmsX: bg(this.pipeRMS, [this.xABuf, this.weightsBuf, this.xNormBuf]),
      qSem: bg(this.pipeLinear, [this.xNormBuf, this.weightsBuf, this.centroidBuf, this.qSemBuf]),
      kSem: bg(this.pipeLinear, [this.xNormBuf, this.weightsBuf, this.centroidBuf, this.kSemBuf]),
      vInv: bg(this.pipeLinear, [this.xNormBuf, this.weightsBuf, this.centroidBuf, this.vInvBuf]),
      geoQ: bg(this.pipeLinear, [this.xNormBuf, this.weightsBuf, this.centroidBuf, this.geoQBuf]),
      geoK: bg(this.pipeLinear, [this.xNormBuf, this.weightsBuf, this.centroidBuf, this.geoKBuf]),
      buildQK: bg(this.pipeBuildQK, [
        this.geoBuf, this.qSemBuf, this.kSemBuf, this.geoQBuf, this.geoKBuf,
        this.qBuf, this.kBuf,
      ]),
      buildV: bg(this.pipeBuildV, [this.geoBuf, this.vInvBuf, this.vBuf]),
      attn: bg(this.pipeAttn, [this.geoBuf, this.qBuf, this.kBuf, this.vBuf, this.attnMergedBuf]),
      wO: bg(this.pipeLinear, [this.attnMergedBuf, this.weightsBuf, this.xABuf, this.xBBuf]),
      rmsXB: bg(this.pipeRMS, [this.xBBuf, this.weightsBuf, this.xNormBuf]),
      fc1: bg(this.pipeLinear, [this.xNormBuf, this.weightsBuf, this.centroidBuf, this.ffnBuf]),
      fc2: bg(this.pipeLinear, [this.ffnBuf, this.weightsBuf, this.xBBuf, this.xABuf]),
      head1: bg(this.pipeLinear, [this.xNormBuf, this.weightsBuf, this.centroidBuf, this.xBBuf]),
      head2: bg(this.pipeLinear, [this.xBBuf, this.weightsBuf, this.centroidBuf, this.rawHeadBuf]),
      decode: bg(this.pipeDecode, [this.rawHeadBuf, this.shOutBuf]),
      packWorldSH: bg(this.pipePackWorldSH, [this.tokensBuf, this.shOutBuf, this.worldPatchGpuBuf]),
    };
    if (this.isQ8Turbo && this.f32AuxBuf) {
      const aux = this.f32AuxBuf;
      this.cachedBGs.q8Stem1 = bg(this.pipeLinearF32, [this.intrBuf, aux, this.centroidBuf, this.xBBuf]);
      this.cachedBGs.q8Stem2 = bg(this.pipeLinearQ8, [this.xBBuf, this.weightsBuf, aux, this.centroidBuf, this.xABuf]);
      this.cachedBGs.q8RmsX = bg(this.pipeRMSF32, [this.xABuf, aux, this.xNormBuf]);
      this.cachedBGs.q8QkvGeo = bg(this.pipeLinearQ8, [this.xNormBuf, this.weightsBuf, aux, this.centroidBuf, this.qkvGeoBuf]);
      this.cachedBGs.q8BuildQKV = bg(this.pipeBuildQKVFused, [this.geoBuf, this.qkvGeoBuf, this.qBuf, this.kBuf, this.vBuf]);
      this.cachedBGs.q8WO = bg(this.pipeLinearQ8, [this.attnMergedBuf, this.weightsBuf, aux, this.xABuf, this.xBBuf]);
      this.cachedBGs.q8RmsXB = bg(this.pipeRMSF32, [this.xBBuf, aux, this.xNormBuf]);
      this.cachedBGs.q8Fc1 = bg(this.pipeLinearQ8, [this.xNormBuf, this.weightsBuf, aux, this.centroidBuf, this.ffnBuf]);
      this.cachedBGs.q8Fc2 = bg(this.pipeLinearQ8, [this.ffnBuf, this.weightsBuf, aux, this.xBBuf, this.xABuf]);
      this.cachedBGs.q8Head1 = bg(this.pipeLinearQ8, [this.xNormBuf, this.weightsBuf, aux, this.centroidBuf, this.xBBuf]);
      this.cachedBGs.q8Head2 = bg(this.pipeLinearF32, [this.xBBuf, aux, this.centroidBuf, this.rawHeadBuf]);
    }
  }

  private _ensureCapacity(nTokens: number): void {
    if (!this.device || nTokens <= this.maxTokens) return;
    const device = this.device;
    const cap = Math.max(nTokens, 1024);
    const { dModel, nHeads, dFF } = this.header;

    const mkBuf = (floats: number, extraUsage = 0) =>
      device.createBuffer({
        size: Math.max(16, floats * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
      });

    for (const b of [
      this.tokensBuf, this.centroidBuf, this.geoBuf, this.intrBuf,
      this.xABuf, this.xBBuf, this.xNormBuf, this.ffnBuf, this.qkvGeoBuf,
      this.qSemBuf, this.kSemBuf, this.vInvBuf, this.geoQBuf, this.geoKBuf,
      this.qBuf, this.kBuf, this.vBuf, this.attnMergedBuf,
      this.rawHeadBuf, this.shOutBuf, this.worldPatchGpuBuf, this.readbackBuf,
    ]) {
      if (b) b.destroy();
    }

    this.maxTokens = cap;
    this.tokensBuf = mkBuf(cap * TOKEN_STRIDE);
    this.centroidBuf = mkBuf(4);
    this.geoBuf = mkBuf(cap * 16);
    this.intrBuf = mkBuf(cap * 13);
    this.xABuf = mkBuf(cap * dModel);
    this.xBBuf = mkBuf(cap * dModel);
    this.xNormBuf = mkBuf(cap * dModel);
    this.ffnBuf = mkBuf(cap * dFF);
    this.qkvGeoBuf = mkBuf(cap * nHeads * 78);
    this.qSemBuf = mkBuf(cap * nHeads * 19);
    this.kSemBuf = mkBuf(cap * nHeads * 19);
    this.vInvBuf = mkBuf(cap * nHeads * 26);
    this.geoQBuf = mkBuf(cap * nHeads * 7);
    this.geoKBuf = mkBuf(cap * nHeads * 7);
    this.qBuf = mkBuf(cap * nHeads * 32);
    this.kBuf = mkBuf(cap * nHeads * 32);
    this.vBuf = mkBuf(cap * nHeads * 32);
    this.attnMergedBuf = mkBuf(cap * dModel);
    this.rawHeadBuf = mkBuf(cap * 36);
    this.shOutBuf = mkBuf(cap * SH_STRIDE);
    this.worldPatchGpuBuf = mkBuf(cap * WORLD_PATCH_VEC4S * 4);
    this.readbackBuf = device.createBuffer({
      size: cap * SH_STRIDE * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this._rebuildBindGroups();
  }

  /**
   * Run full 9-layer RADIANT SE(3) Form-Factor Transformer on GPU (RDG8 Turbo INT8 or RDGI FP16).
   * When `readback = false` and Three.js's WebGPU StorageBufferAttribute GPUBuffer is active,
   * packs world-space SH directly on-GPU with ZERO `mapAsync` CPU readback stall!
   */
  public inferTokens(
    tokens: Float32Array,
    updateWorldStorage = true,
    readback = true,
    objectRanges?: ObjectPatchRange[] | null,
    bounceOverride?: number,
  ): Promise<Float32Array> {
    const tokensSnap = new Float32Array(tokens);
    const rangesSnap = objectRanges ? objectRanges.map((r) => ({ ...r })) : null;
    const task = this.inferQueue.then(() =>
      this._runInferTokens(tokensSnap, updateWorldStorage, readback, rangesSnap, bounceOverride),
    );
    this.inferQueue = task.then(
      () => undefined,
      () => undefined,
    );
    return task;
  }

  /**
   * Instant 0-ms bounce-depth switching (1 Bounce / 2 Bounce / 3 Bounce) without re-running the
   * 9-layer Transformer! Re-decodes cached `rawHeadBuf` on GPU (or `lastRawBounces` on CPU) in < 0.1 ms.
   */
  public async setBounceCount(
    bounces: number,
    updateWorldStorage = true,
    objectRanges?: ObjectPatchRange[] | null,
    readback = false,
  ): Promise<Float32Array> {
    const clamped = Math.max(1.0, Math.min(3.0, bounces));
    this.bounceCount = clamped;
    const nTokens = this.activeTokenCount;
    if (nTokens <= 0 || !this.isWeightsLoaded) return this.lastLocalSH;
    const ranges = objectRanges ?? this.lastObjectRanges;

    if (this.isCPUFallback || !this.device) {
      if (this.lastRawBounces.length > 0) {
        const combined = combineBounceSH(this.lastRawBounces, nTokens, clamped, Boolean(this.header?.isMultiBounce));
        this.lastLocalSH = combined;
        if (updateWorldStorage && this.lastWorldTokens.length >= nTokens * TOKEN_STRIDE) {
          this.uploadWorldSHToTSL(this.lastWorldTokens, combined, nTokens, null, ranges);
        }
        return combined;
      }
      return this.lastLocalSH;
    }

    if (!this.cachedBGs) this._rebuildBindGroups();
    const bgs = this.cachedBGs!;
    const threeDstBuf: GPUBuffer | undefined =
      updateWorldStorage && !this.useTextureLookup
        ? this.rendererBackend?.get?.(this.worldPatchStorage)?.buffer
        : undefined;

    const base0 = 0;
    this.paramsCPU[base0 + 0] = nTokens >>> 0;
    this.paramsCPU[base0 + 1] = this.header.isMultiBounce ? 1 : 0;
    this.paramsCPUF32[base0 + 2] = clamped;
    this.paramsCPU[base0 + 3] = 0;

    const base1 = this.PARAM_STRIDE_BYTES >> 2;
    this.paramsCPU[base1 + 0] = Math.min(nTokens, MAX_RENDER_PATCHES) >>> 0;
    this.paramsCPU[base1 + 1] = 0;
    this.paramsCPU[base1 + 2] = 0;
    this.paramsCPU[base1 + 3] = 0;

    this.device.queue.writeBuffer(this.paramsRingBuf, 0, this.paramsCPU.buffer, 0, this.PARAM_STRIDE_BYTES * 2);

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeDecode);
    pass.setBindGroup(0, bgs.decode, [0]);
    pass.dispatchWorkgroups(Math.ceil(nTokens / 64), 1, 1);
    if (threeDstBuf) {
      pass.setPipeline(this.pipePackWorldSH);
      pass.setBindGroup(0, bgs.packWorldSH, [this.PARAM_STRIDE_BYTES]);
      pass.dispatchWorkgroups(Math.ceil(Math.min(nTokens, MAX_RENDER_PATCHES) / 64), 1, 1);
    }
    pass.end();

    const outBytes = nTokens * SH_STRIDE * 4;
    if (threeDstBuf) {
      const packBytes = Math.min(nTokens, MAX_RENDER_PATCHES) * WORLD_PATCH_VEC4S * 4 * 4;
      enc.copyBufferToBuffer(this.worldPatchGpuBuf, 0, threeDstBuf, 0, packBytes);
      if (!readback) {
        this.device.queue.submit([enc.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        return this.lastLocalSH;
      }
    }

    enc.copyBufferToBuffer(this.shOutBuf, 0, this.readbackBuf, 0, outBytes);
    this.device.queue.submit([enc.finish()]);
    await this.readbackBuf.mapAsync(GPUMapMode.READ, 0, outBytes);
    const mapped = new Float32Array(this.readbackBuf.getMappedRange(0, outBytes));
    const result = new Float32Array(mapped);
    this.readbackBuf.unmap();
    this.lastLocalSH = result;
    if (!threeDstBuf && updateWorldStorage && this.lastWorldTokens.length >= nTokens * TOKEN_STRIDE) {
      this.uploadWorldSHToTSL(this.lastWorldTokens, result, nTokens, null, ranges);
    }
    return result;
  }

  private async _runInferTokens(
    tokens: Float32Array,
    updateWorldStorage: boolean,
    readback: boolean,
    objectRanges: ObjectPatchRange[] | null,
    bounceOverride?: number,
  ): Promise<Float32Array> {
    this.isBusy = true;
    try {
      const nTokens = Math.floor(tokens.length / TOKEN_STRIDE);
      if (nTokens <= 0) return new Float32Array(0);
      const effBounces = bounceOverride !== undefined ? bounceOverride : this.bounceCount;
      if (objectRanges) this.lastObjectRanges = objectRanges;
      if (this.isCPUFallback || !this.device) {
        if (!this.cpuWeightsF32 && !this.cpuWorker) return new Float32Array(nTokens * SH_STRIDE);
        const t0 = performance.now();
        const workerOut = await this._runCpuWorker(tokens, nTokens);
        const rawDecoded = workerOut
          ? workerOut.result
          : await runRadiantCPU(tokens, nTokens, this.header, this.cpuWeightsF32!, this.layout);
        const result = combineBounceSH(rawDecoded, nTokens, effBounces, Boolean(this.header?.isMultiBounce));
        this.lastComputeMs = workerOut ? workerOut.computeMs : performance.now() - t0;
        this.activeTokenCount = nTokens;
        this.lastRawBounces = rawDecoded;
        this.lastLocalSH = result;
        this.lastWorldTokens = new Float32Array(tokens.subarray(0, nTokens * TOKEN_STRIDE));
        if (updateWorldStorage) {
          this.uploadWorldSHToTSL(tokens, result, nTokens, null, objectRanges);
        }
        return result;
      }
      if (!this.weightsBuf) return new Float32Array(nTokens * SH_STRIDE);
      this._ensureCapacity(nTokens);

      const t0 = performance.now();
      const { dModel, nHeads, nLayers, dFF, posScale, hasAreaPrior, isMultiBounce } = this.header;
      const head2OutDim = isMultiBounce ? 36 : 12;

      // Upload input tokens
      this.device.queue.writeBuffer(this.tokensBuf, 0, tokens.buffer, tokens.byteOffset, nTokens * TOKEN_STRIDE * 4);

      // Build uniform ring buffer for ALL dispatches in this forward pass
      let slot = 0;
      const pushU32 = (...vals: number[]): number => {
        const s = slot++;
        const base = (s * this.PARAM_STRIDE_BYTES) >> 2;
        for (let k = 0; k < vals.length; k++) this.paramsCPU[base + k] = vals[k] >>> 0;
        return s * this.PARAM_STRIDE_BYTES;
      };
      const pushPrep = (n: number, scale: number, areaPrior: boolean): number => {
        const s = slot++;
        const base = (s * this.PARAM_STRIDE_BYTES) >> 2;
        this.paramsCPU[base + 0] = n >>> 0;
        this.paramsCPUF32[base + 1] = scale;
        this.paramsCPU[base + 2] = areaPrior ? 1 : 0;
        this.paramsCPU[base + 3] = 0;
        return s * this.PARAM_STRIDE_BYTES;
      };
      const pushDecode = (n: number, multiB: boolean, bCount: number): number => {
        const s = slot++;
        const base = (s * this.PARAM_STRIDE_BYTES) >> 2;
        this.paramsCPU[base + 0] = n >>> 0;
        this.paramsCPU[base + 1] = multiB ? 1 : 0;
        this.paramsCPUF32[base + 2] = bCount;
        this.paramsCPU[base + 3] = 0;
        return s * this.PARAM_STRIDE_BYTES;
      };

      if (!this.cachedBGs) this._rebuildBindGroups();
      const bgs = this.cachedBGs!;

      // Check whether we can do zero-readback GPU-to-GPU World SH copy directly into Three.js StorageBufferAttribute!
      const threeDstBuf: GPUBuffer | undefined =
        updateWorldStorage && !readback && !this.useTextureLookup
          ? this.rendererBackend?.get?.(this.worldPatchStorage)?.buffer
          : undefined;
      const useZeroReadback = Boolean(threeDstBuf);

      // Single command encoder for the ENTIRE 9-layer Transformer!
      const enc = this.device.createCommandEncoder();
      const wgTok32 = Math.ceil(nTokens / 32);
      const wgTok64 = Math.ceil(nTokens / 64);
      const wgQKV = Math.ceil((nTokens * nHeads) / 64);

      if (this.isQ8Turbo && this.q8Layout) {
        const QL = this.q8Layout;
        const offCentroid = pushU32(nTokens, 0, 0, 0);
        const offPrep = pushPrep(nTokens, posScale, hasAreaPrior);
        const offStem1 = pushU32(nTokens, 13, dModel, QL.stem1W, QL.stem1B, 1, 0, 0);
        const offStem2 = pushU32(nTokens, dModel, dModel, QL.stem2W, QL.stem2S, QL.stem2B, 0, 0);

        const qkvOutDim = nHeads * 78;
        const layerSlots = QL.layers.map((L) => ({
          rms1: pushU32(nTokens, dModel, L.norm1W, 0),
          qkvGeo: pushU32(nTokens, dModel, qkvOutDim, L.qkvW, L.qkvS, L.qkvB, 0, 0),
          qkvBuild: pushU32(nTokens, nHeads, 0, 0),
          attn: pushU32(nTokens, nHeads, 0, 0),
          wO: pushU32(nTokens, dModel, dModel, L.wOW, L.wOS, L.wOB, 0, 1),
          rms2: pushU32(nTokens, dModel, L.norm2W, 0),
          fc1: pushU32(nTokens, dModel, dFF, L.fc1W, L.fc1S, L.fc1B, 1, 0),
          fc2: pushU32(nTokens, dFF, dModel, L.fc2W, L.fc2S, L.fc2B, 0, 1),
        }));

        const offFinalNorm = pushU32(nTokens, dModel, QL.finalNormW, 0);
        const offHead1 = pushU32(nTokens, dModel, dModel, QL.head1W, QL.head1S, QL.head1B, 1, 0);
        const offHead2 = pushU32(nTokens, dModel, head2OutDim, QL.head2W, QL.head2B, 0, 0, 0);
        const offDecode = pushDecode(nTokens, isMultiBounce, effBounces);
        const offPack = useZeroReadback ? pushU32(Math.min(nTokens, MAX_RENDER_PATCHES), 0, 0, 0) : 0;

        this.device.queue.writeBuffer(this.paramsRingBuf, 0, this.paramsCPU.buffer, 0, slot * this.PARAM_STRIDE_BYTES);

        const pass = enc.beginComputePass();
        const dispatchQ8 = (bindGroup: GPUBindGroup, dynOff: number, outDim: number) => {
          pass.setPipeline(this.pipeLinearQ8);
          pass.setBindGroup(0, bindGroup, [dynOff]);
          pass.dispatchWorkgroups(Math.ceil(outDim / 32), wgTok32, 1);
        };

        pass.setPipeline(this.pipeCentroid);
        pass.setBindGroup(0, bgs.centroid, [offCentroid]);
        pass.dispatchWorkgroups(1, 1, 1);

        pass.setPipeline(this.pipePrep);
        pass.setBindGroup(0, bgs.prep, [offPrep]);
        pass.dispatchWorkgroups(wgTok64, 1, 1);

        pass.setPipeline(this.pipeLinearF32);
        pass.setBindGroup(0, bgs.q8Stem1!, [offStem1]);
        pass.dispatchWorkgroups(Math.ceil(dModel / 64), nTokens, 1);

        dispatchQ8(bgs.q8Stem2!, offStem2, dModel);

        for (let l = 0; l < nLayers; l++) {
          const s = layerSlots[l];
          pass.setPipeline(this.pipeRMSF32);
          pass.setBindGroup(0, bgs.q8RmsX!, [s.rms1]);
          pass.dispatchWorkgroups(nTokens, 1, 1);

          // Single fused QKV+Geo projection (5 dispatches -> 1!)
          dispatchQ8(bgs.q8QkvGeo!, s.qkvGeo, qkvOutDim);

          // Single fused Q+K+V builder (2 dispatches -> 1!)
          pass.setPipeline(this.pipeBuildQKVFused);
          pass.setBindGroup(0, bgs.q8BuildQKV!, [s.qkvBuild]);
          pass.dispatchWorkgroups(wgQKV, 1, 1);

          pass.setPipeline(this.pipeAttn);
          pass.setBindGroup(0, bgs.attn, [s.attn]);
          pass.dispatchWorkgroups(nHeads, wgTok32, 1);

          dispatchQ8(bgs.q8WO!, s.wO, dModel);

          pass.setPipeline(this.pipeRMSF32);
          pass.setBindGroup(0, bgs.q8RmsXB!, [s.rms2]);
          pass.dispatchWorkgroups(nTokens, 1, 1);

          dispatchQ8(bgs.q8Fc1!, s.fc1, dFF);
          dispatchQ8(bgs.q8Fc2!, s.fc2, dModel);
        }

        pass.setPipeline(this.pipeRMSF32);
        pass.setBindGroup(0, bgs.q8RmsX!, [offFinalNorm]);
        pass.dispatchWorkgroups(nTokens, 1, 1);

        dispatchQ8(bgs.q8Head1!, offHead1, dModel);

        pass.setPipeline(this.pipeLinearF32);
        pass.setBindGroup(0, bgs.q8Head2!, [offHead2]);
        pass.dispatchWorkgroups(1, nTokens, 1);

        pass.setPipeline(this.pipeDecode);
        pass.setBindGroup(0, bgs.decode, [offDecode]);
        pass.dispatchWorkgroups(wgTok64, 1, 1);

        if (useZeroReadback) {
          pass.setPipeline(this.pipePackWorldSH);
          pass.setBindGroup(0, bgs.packWorldSH, [offPack]);
          pass.dispatchWorkgroups(Math.ceil(Math.min(nTokens, MAX_RENDER_PATCHES) / 64), 1, 1);
        }
        pass.end();
      } else {
        // Legacy RDGI v1 FP16 path (100% preserved + Area-Prior & Multi-Bounce support)
        const offCentroid = pushU32(nTokens, 0, 0, 0);
        const offPrep = pushPrep(nTokens, posScale, hasAreaPrior);
        const offStem1 = pushU32(nTokens, 13, dModel, this.layout.stem1W, this.layout.stem1B, 1, 0, 0);
        const offStem2 = pushU32(nTokens, dModel, dModel, this.layout.stem2W, this.layout.stem2B, 0, 0, 0);

        const layerSlots = this.layout.layers.map((L) => ({
          rms1: pushU32(nTokens, dModel, L.norm1W, 0),
          qSem: pushU32(nTokens, dModel, nHeads * 19, L.qSemW, L.qSemB, 0, 0, 0),
          kSem: pushU32(nTokens, dModel, nHeads * 19, L.kSemW, L.kSemB, 0, 0, 0),
          vInv: pushU32(nTokens, dModel, nHeads * 26, L.vInvW, L.vInvB, 0, 0, 0),
          geoQ: pushU32(nTokens, dModel, nHeads * 7, L.geoQW, L.geoQB, 0, 0, 0),
          geoK: pushU32(nTokens, dModel, nHeads * 7, L.geoKW, L.geoKB, 0, 0, 0),
          qkv: pushU32(nTokens, nHeads, 0, 0),
          attn: pushU32(nTokens, nHeads, 0, 0),
          wO: pushU32(nTokens, dModel, dModel, L.wOW, L.wOB, 0, 1, 0),
          rms2: pushU32(nTokens, dModel, L.norm2W, 0),
          fc1: pushU32(nTokens, dModel, dFF, L.fc1W, L.fc1B, 1, 0, 0),
          fc2: pushU32(nTokens, dFF, dModel, L.fc2W, L.fc2B, 0, 1, 0),
        }));

        const offFinalNorm = pushU32(nTokens, dModel, this.layout.finalNormW, 0);
        const offHead1 = pushU32(nTokens, dModel, dModel, this.layout.head1W, this.layout.head1B, 1, 0, 0);
        const offHead2 = pushU32(nTokens, dModel, head2OutDim, this.layout.head2W, this.layout.head2B, 0, 0, 0);
        const offDecode = pushDecode(nTokens, isMultiBounce, effBounces);
        const offPack = useZeroReadback ? pushU32(Math.min(nTokens, MAX_RENDER_PATCHES), 0, 0, 0) : 0;

        this.device.queue.writeBuffer(this.paramsRingBuf, 0, this.paramsCPU.buffer, 0, slot * this.PARAM_STRIDE_BYTES);

        const pass = enc.beginComputePass();
        const wgTok16 = Math.ceil(nTokens / 16);
        const dispatchLinear = (bindGroup: GPUBindGroup, dynOff: number, outDim: number) => {
          pass.setPipeline(this.pipeLinear);
          pass.setBindGroup(0, bindGroup, [dynOff]);
          pass.dispatchWorkgroups(Math.ceil(outDim / 64), wgTok16, 1);
        };

        pass.setPipeline(this.pipeCentroid);
        pass.setBindGroup(0, bgs.centroid, [offCentroid]);
        pass.dispatchWorkgroups(1, 1, 1);

        pass.setPipeline(this.pipePrep);
        pass.setBindGroup(0, bgs.prep, [offPrep]);
        pass.dispatchWorkgroups(wgTok64, 1, 1);

        dispatchLinear(bgs.stem1, offStem1, dModel);
        dispatchLinear(bgs.stem2, offStem2, dModel);

        for (let l = 0; l < nLayers; l++) {
          const s = layerSlots[l];
          pass.setPipeline(this.pipeRMS);
          pass.setBindGroup(0, bgs.rmsX, [s.rms1]);
          pass.dispatchWorkgroups(nTokens, 1, 1);

          dispatchLinear(bgs.qSem, s.qSem, nHeads * 19);
          dispatchLinear(bgs.kSem, s.kSem, nHeads * 19);
          dispatchLinear(bgs.vInv, s.vInv, nHeads * 26);
          dispatchLinear(bgs.geoQ, s.geoQ, nHeads * 7);
          dispatchLinear(bgs.geoK, s.geoK, nHeads * 7);

          pass.setPipeline(this.pipeBuildQK);
          pass.setBindGroup(0, bgs.buildQK, [s.qkv]);
          pass.dispatchWorkgroups(wgQKV, 1, 1);

          pass.setPipeline(this.pipeBuildV);
          pass.setBindGroup(0, bgs.buildV, [s.qkv]);
          pass.dispatchWorkgroups(wgQKV, 1, 1);

          pass.setPipeline(this.pipeAttn);
          pass.setBindGroup(0, bgs.attn, [s.attn]);
          pass.dispatchWorkgroups(nHeads, wgTok32, 1);

          dispatchLinear(bgs.wO, s.wO, dModel);

          pass.setPipeline(this.pipeRMS);
          pass.setBindGroup(0, bgs.rmsXB, [s.rms2]);
          pass.dispatchWorkgroups(nTokens, 1, 1);

          dispatchLinear(bgs.fc1, s.fc1, dFF);
          dispatchLinear(bgs.fc2, s.fc2, dModel);
        }

        pass.setPipeline(this.pipeRMS);
        pass.setBindGroup(0, bgs.rmsX, [offFinalNorm]);
        pass.dispatchWorkgroups(nTokens, 1, 1);

        dispatchLinear(bgs.head1, offHead1, dModel);
        dispatchLinear(bgs.head2, offHead2, head2OutDim);

        pass.setPipeline(this.pipeDecode);
        pass.setBindGroup(0, bgs.decode, [offDecode]);
        pass.dispatchWorkgroups(wgTok64, 1, 1);

        if (useZeroReadback) {
          pass.setPipeline(this.pipePackWorldSH);
          pass.setBindGroup(0, bgs.packWorldSH, [offPack]);
          pass.dispatchWorkgroups(Math.ceil(Math.min(nTokens, MAX_RENDER_PATCHES) / 64), 1, 1);
        }
        pass.end();
      }

      const outBytes = nTokens * SH_STRIDE * 4;
      if (useZeroReadback && threeDstBuf) {
        const packBytes = Math.min(nTokens, MAX_RENDER_PATCHES) * WORLD_PATCH_VEC4S * 4 * 4;
        enc.copyBufferToBuffer(this.worldPatchGpuBuf, 0, threeDstBuf, 0, packBytes);
      } else {
        enc.copyBufferToBuffer(this.shOutBuf, 0, this.readbackBuf, 0, outBytes);
      }

      let gpuValidationErr: GPUError | null = null;
      if (this.useTextureLookup && typeof this.device.pushErrorScope === 'function') {
        this.device.pushErrorScope('validation');
      }
      this.device.queue.submit([enc.finish()]);
      if (this.useTextureLookup && typeof this.device.popErrorScope === 'function') {
        gpuValidationErr = await this.device.popErrorScope();
      }
      if (gpuValidationErr) {
        console.warn('[RadiantComputeEngine] WebGPU compute validation error, switching to CPU fallback:', gpuValidationErr.message);
        this.device = null;
        this.isCPUFallback = true;
        if (!this.cpuWeightsF32 && this.rawWeightsBuffer) {
          if (this.isQ8Turbo) {
            const dv = new DataView(this.rawWeightsBuffer);
            const unpacked = unpackRDG8ToCanonicalF32(
              this.rawWeightsBuffer,
              this.header,
              dv.getUint32(24, true),
              dv.getUint32(28, true),
            );
            this.cpuWeightsF32 = unpacked.weights;
            this.layout = unpacked.layout;
          } else {
            this.cpuWeightsF32 = unpackRDGIWeightsF32(this.rawWeightsBuffer);
          }
        }
        return this._runInferTokens(tokens, updateWorldStorage, readback, objectRanges, bounceOverride);
      }

      if (useZeroReadback) {
        await this.device.queue.onSubmittedWorkDone();
        this.lastComputeMs = performance.now() - t0;
        this.activeTokenCount = nTokens;
        this.lastWorldTokens = new Float32Array(tokens.subarray(0, nTokens * TOKEN_STRIDE));
        return this.lastLocalSH;
      }

      await this.readbackBuf.mapAsync(GPUMapMode.READ, 0, outBytes);
      const mapped = new Float32Array(this.readbackBuf.getMappedRange(0, outBytes));
      const result = new Float32Array(mapped);
      this.readbackBuf.unmap();

      this.lastComputeMs = performance.now() - t0;
      this.activeTokenCount = nTokens;
      this.lastLocalSH = result;
      this.lastWorldTokens = new Float32Array(tokens.subarray(0, nTokens * TOKEN_STRIDE));

      if (updateWorldStorage) {
        this.uploadWorldSHToTSL(tokens, result, nTokens, null, objectRanges);
      }
      return result;
    } finally {
      this.isBusy = false;
    }
  }

  /**
   * Convert local Order-1 SH [N, 12] to world-space Order-1 SH and pack into `worldPatchStorage`
   * (WebGPU StorageBufferAttribute) AND `worldPatchTexture` (WebGL2 RGBA32F DataTexture with
   * per-object variable LOD rows up to 64 patches/object!).
   */
  public uploadWorldSHToTSL(
    tokens: Float32Array,
    shLocal: Float32Array,
    nTokens: number,
    gtLocalSH?: Float32Array | null,
    objectRanges?: ObjectPatchRange[] | null,
  ): void {
    const count = Math.min(
      nTokens,
      MAX_RENDER_PATCHES,
      Math.floor(tokens.length / TOKEN_STRIDE),
      Math.floor(shLocal.length / SH_STRIDE),
    );
    const dst = this.worldPatchArray;

    for (let i = 0; i < count; i++) {
      const tb = i * TOKEN_STRIDE;
      const sb = i * SH_STRIDE;
      const db = i * WORLD_PATCH_VEC4S * 4;

      const px = tokens[tb + 0], py = tokens[tb + 1], pz = tokens[tb + 2];
      const tx = tokens[tb + 3], ty = tokens[tb + 4], tz = tokens[tb + 5];
      const bx = tokens[tb + 6], by = tokens[tb + 7], bz = tokens[tb + 8];
      const nx = tokens[tb + 9], ny = tokens[tb + 10], nz = tokens[tb + 11];
      const area = Math.max(tokens[tb + 12], 1e-4);
      const radius = Math.sqrt(area / Math.PI);

      const i0r = shLocal[sb + 0], i0g = shLocal[sb + 1], i0b = shLocal[sb + 2];
      const itr = shLocal[sb + 3], itg = shLocal[sb + 4], itb = shLocal[sb + 5];
      const ibr = shLocal[sb + 6], ibg = shLocal[sb + 7], ibb = shLocal[sb + 8];
      const inr = shLocal[sb + 9], ing = shLocal[sb + 10], inb = shLocal[sb + 11];

      const wrx = itr * tx + ibr * bx + inr * nx;
      const wry = itr * ty + ibr * by + inr * ny;
      const wrz = itr * tz + ibr * bz + inr * nz;

      const wgx = itg * tx + ibg * bx + ing * nx;
      const wgy = itg * ty + ibg * by + ing * ny;
      const wgz = itg * tz + ibg * bz + ing * nz;

      const wbx = itb * tx + ibb * bx + inb * nx;
      const wby = itb * ty + ibb * by + inb * ny;
      const wbz = itb * tz + ibb * bz + inb * nz;

      let errVal = 0.0;
      if (gtLocalSH && gtLocalSH.length >= sb + 3) {
        const dr = i0r - gtLocalSH[sb + 0];
        const dg = i0g - gtLocalSH[sb + 1];
        const dbErr = i0b - gtLocalSH[sb + 2];
        errVal = Math.sqrt((dr * dr + dg * dg + dbErr * dbErr) / 3.0);
      }

      dst[db + 0] = px; dst[db + 1] = py; dst[db + 2] = pz; dst[db + 3] = radius;
      dst[db + 4] = nx; dst[db + 5] = ny; dst[db + 6] = nz; dst[db + 7] = area;
      dst[db + 8] = i0r; dst[db + 9] = i0g; dst[db + 10] = i0b; dst[db + 11] = inr;
      dst[db + 12] = wrx; dst[db + 13] = wry; dst[db + 14] = wrz; dst[db + 15] = ing;
      dst[db + 16] = wgx; dst[db + 17] = wgy; dst[db + 18] = wgz; dst[db + 19] = inb;
      dst[db + 20] = wbx; dst[db + 21] = wby; dst[db + 22] = wbz; dst[db + 23] = errVal;
    }

    // Mirror into per-object rows of WebGL2 RGBA32F DataTexture (supports mixed 4/8/16/32/64 LODs!)
    if (this.useTextureLookup) {
      const texDst = this.worldPatchTexArray;
      const rowStrideFloats = MAX_PATCHES_PER_OBJECT * WORLD_PATCH_VEC4S * 4;
      const patchFloats = WORLD_PATCH_VEC4S * 4;
      const ranges = objectRanges ?? this.lastObjectRanges;
      if (ranges && ranges.length > 0) {
        for (const r of ranges) {
          if (r.objectId < 0 || r.objectId >= MAX_RENDER_OBJECTS) continue;
          const kCount = Math.min(r.patchCount, MAX_PATCHES_PER_OBJECT, Math.max(0, count - r.patchStart));
          if (kCount <= 0) continue;
          const srcOff = r.patchStart * patchFloats;
          const dstOff = r.objectId * rowStrideFloats;
          texDst.set(dst.subarray(srcOff, srcOff + kCount * patchFloats), dstOff);
        }
      } else {
        // Legacy uniform-32 fallback layout
        const nObjs = Math.min(MAX_RENDER_OBJECTS, Math.ceil(count / PATCHES_PER_OBJECT));
        for (let o = 0; o < nObjs; o++) {
          const kCount = Math.min(PATCHES_PER_OBJECT, count - o * PATCHES_PER_OBJECT);
          const srcOff = o * PATCHES_PER_OBJECT * patchFloats;
          const dstOff = o * rowStrideFloats;
          texDst.set(dst.subarray(srcOff, srcOff + kCount * patchFloats), dstOff);
        }
      }
      this.worldPatchTexture.needsUpdate = true;
    }

    this.worldPatchStorage.needsUpdate = true;
  }

  public getPatchOffsetForObject(objectId: number): number {
    return objectId * PATCHES_PER_OBJECT;
  }
}
