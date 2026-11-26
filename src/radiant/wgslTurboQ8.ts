/**
 * Turbo WGSL Compute Shaders for `RDG8` v2 (Fused-QKV Per-Row Symmetric INT8 + Zero-Readback GPU Pack).
 *
 * Key Apple M1 Metal 3 optimizations:
 *   1. `WGSL_LINEAR_Q8`: TILE_N = 32 tokens x 64 outputs/WG, 4 signed int8 weights unpacked
 *      in ONE hardware instruction via `unpack4x8snorm(u32)` -> 32 FLOPs/byte (4x less DRAM
 *      weight bandwidth than FP16 TILE_N=16!), with optional `enable f16;` shared `x_tile`
 *      halving threadgroup memory to 2 KB for higher GPU core occupancy.
 *   2. `WGSL_BUILD_SE3_QKV_FUSED`: Builds Q, K, and V in ONE dispatch from the single
 *      fused [N, 624] QKV+Geo projection (cutting 7 dispatches/layer down to 2!).
 *   3. `WGSL_PACK_WORLD_SH`: Converts local Order-1 SH -> world-space 6x vec4 patch buffer
 *      directly on GPU so live drag frames never stall on `readbackBuf.mapAsync()`.
 */

export const WGSL_RMS_NORM_F32 = /* wgsl */ `
struct Params {
  n_tokens: u32,
  dim: u32,
  w_off: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_in: array<f32>;
@group(0) @binding(2) var<storage, read> f32_aux: array<f32>;
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
    x_out[base + d] = x_in[base + d] * inv_rms * f32_aux[params.w_off + d];
    d += 64u;
  }
}
`;

export const WGSL_LINEAR_F32 = /* wgsl */ `
struct Params {
  n_tokens: u32,
  in_dim: u32,
  out_dim: u32,
  w_off: u32,
  b_off: u32,
  act_mode: u32,
  add_res: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_in: array<f32>;
@group(0) @binding(2) var<storage, read> f32_aux: array<f32>;
@group(0) @binding(3) var<storage, read> res_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> y_out: array<f32>;

fn gelu_tanh(x: f32) -> f32 {
  let c = 0.7978845608028654;
  let arg = clamp(c * (x + 0.044715 * x * x * x), -15.0, 15.0);
  return 0.5 * x * (1.0 + tanh(arg));
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let out_idx = wg.x * 64u + lid.x;
  let tok = wg.y;
  if (out_idx >= params.out_dim || tok >= params.n_tokens) { return; }

  let in_d = params.in_dim;
  let xb = tok * in_d;
  let wb = params.w_off + out_idx * in_d;
  var acc = f32_aux[params.b_off + out_idx];
  for (var k: u32 = 0u; k < in_d; k++) {
    acc += x_in[xb + k] * f32_aux[wb + k];
  }
  if (params.act_mode == 1u) {
    acc = gelu_tanh(acc);
  }
  let y_idx = tok * params.out_dim + out_idx;
  if (params.add_res == 1u) {
    acc += res_in[y_idx];
  }
  y_out[y_idx] = acc;
}
`;

/**
 * Build Coalesced 2D Register-Blocked TILE_M=32 x TILE_N=64 INT8 GEMM shader (@workgroup_size(16, 8)).
 *
 * Each thread owns a 4-token x 4-output micro-tile in 4 explicit vec4<f32> registers (ZERO local
 * arrays, ZERO stack spills!), while all 128 threads cooperatively load both `x_tile` (32x32)
 * and `w_tile` (64x32, unpacked via hardware `unpack4x8snorm`) 100% coalesced into threadgroup
 * memory (only 6 KB total in f16 -> high Apple M1 Metal 3 occupancy!).
 */
export function buildLinearQ8WGSL(useF16Tile: boolean): string {
  const enableHeader = useF16Tile ? 'enable f16;\n' : '';
  const tileType = useF16Tile ? 'vec4<f16>' : 'vec4<f32>';
  const tileZero = useF16Tile ? 'vec4<f16>(0.0h)' : 'vec4<f32>(0.0)';
  const loadXExpr = useF16Tile
    ? 'vec4<f16>(f16(x_in[xb]), f16(x_in[xb + 1u]), f16(x_in[xb + 2u]), f16(x_in[xb + 3u]))'
    : 'vec4<f32>(x_in[xb], x_in[xb + 1u], x_in[xb + 2u], x_in[xb + 3u])';
  const loadWExpr = useF16Tile
    ? 'vec4<f16>(unpack4x8snorm(w_q8[params.w_off_u32 + out_i * in_u32 + k_u32 + q_loc]))'
    : 'unpack4x8snorm(w_q8[params.w_off_u32 + out_i * in_u32 + k_u32 + q_loc])';
  const dotCast = (x: string, w: string) => (useF16Tile ? `f32(dot(${x}, ${w}))` : `dot(${x}, ${w})`);

  return /* wgsl */ `${enableHeader}
struct Params {
  n_tokens: u32,
  in_dim: u32,
  out_dim: u32,
  w_off_u32: u32,
  s_off: u32,
  b_off: u32,
  act_mode: u32,   // 0 = none, 1 = gelu_tanh
  add_res: u32,    // 0 = overwrite, 1 = y += res
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_in: array<f32>;
@group(0) @binding(2) var<storage, read> w_q8: array<u32>;
@group(0) @binding(3) var<storage, read> f32_aux: array<f32>;
@group(0) @binding(4) var<storage, read> res_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> y_out: array<f32>;

// 32 tokens x 8 vec4s (2 KB f16) + 64 outputs x 8 vec4s (4 KB f16) = 6 KB shared!
var<workgroup> x_tile: array<${tileType}, 256>;
var<workgroup> w_tile: array<${tileType}, 512>;

fn gelu_tanh(x: f32) -> f32 {
  let c = 0.7978845608028654;
  let arg = clamp(c * (x + 0.044715 * x * x * x), -15.0, 15.0);
  return 0.5 * x * (1.0 + tanh(arg));
}

@compute @workgroup_size(16, 8)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let lx = lid.x; // 0..15 -> 4 outputs each = 64 outputs
  let ly = lid.y; // 0..7  -> 4 tokens each  = 32 tokens
  let tid = ly * 16u + lx; // 0..127
  let tok_base = wg.y * 32u;
  let out_base = wg.x * 64u;
  let in_d = params.in_dim;
  let out_d = params.out_dim;
  let in_u32 = in_d >> 2u;

  let t0 = ly << 2u;
  let o0 = lx << 2u;

  var a0 = vec4<f32>(0.0);
  var a1 = vec4<f32>(0.0);
  var a2 = vec4<f32>(0.0);
  var a3 = vec4<f32>(0.0);

  var k_start: u32 = 0u;
  while (k_start < in_d) {
    let k_u32 = k_start >> 2u;
    // 1. Cooperatively load x_tile [32 x 8 vec4s] (2 loads/thread, coalesced!)
    for (var s: u32 = 0u; s < 2u; s++) {
      let slot = tid + s * 128u;
      let t_loc = slot >> 3u;
      let q_loc = slot & 7u;
      let tok = tok_base + t_loc;
      let k_off = k_start + (q_loc << 2u);
      if (tok < params.n_tokens && k_off < in_d) {
        let xb = tok * in_d + k_off;
        x_tile[slot] = ${loadXExpr};
      } else {
        x_tile[slot] = ${tileZero};
      }
    }
    // 2. Cooperatively load w_tile [64 x 8 vec4s] (4 loads/thread, 100% coalesced!)
    for (var s: u32 = 0u; s < 4u; s++) {
      let slot = tid + s * 128u;
      let o_loc = slot >> 3u;
      let q_loc = slot & 7u;
      let out_i = out_base + o_loc;
      if (out_i < out_d && (k_u32 + q_loc) < in_u32) {
        w_tile[slot] = ${loadWExpr};
      } else {
        w_tile[slot] = ${tileZero};
      }
    }
    workgroupBarrier();

    // 3. 4x4 register-blocked outer-product micro-kernel (ZERO local array spills!)
    let xb0 = (t0 + 0u) << 3u;
    let xb1 = (t0 + 1u) << 3u;
    let xb2 = (t0 + 2u) << 3u;
    let xb3 = (t0 + 3u) << 3u;
    let wb0 = (o0 + 0u) << 3u;
    let wb1 = (o0 + 1u) << 3u;
    let wb2 = (o0 + 2u) << 3u;
    let wb3 = (o0 + 3u) << 3u;

    for (var q: u32 = 0u; q < 8u; q++) {
      let x0 = x_tile[xb0 + q];
      let x1 = x_tile[xb1 + q];
      let x2 = x_tile[xb2 + q];
      let x3 = x_tile[xb3 + q];
      let w0 = w_tile[wb0 + q];
      let w1 = w_tile[wb1 + q];
      let w2 = w_tile[wb2 + q];
      let w3 = w_tile[wb3 + q];
      a0 += vec4<f32>(${dotCast('x0', 'w0')}, ${dotCast('x0', 'w1')}, ${dotCast('x0', 'w2')}, ${dotCast('x0', 'w3')});
      a1 += vec4<f32>(${dotCast('x1', 'w0')}, ${dotCast('x1', 'w1')}, ${dotCast('x1', 'w2')}, ${dotCast('x1', 'w3')});
      a2 += vec4<f32>(${dotCast('x2', 'w0')}, ${dotCast('x2', 'w1')}, ${dotCast('x2', 'w2')}, ${dotCast('x2', 'w3')});
      a3 += vec4<f32>(${dotCast('x3', 'w0')}, ${dotCast('x3', 'w1')}, ${dotCast('x3', 'w2')}, ${dotCast('x3', 'w3')});
    }
    workgroupBarrier();
    k_start += 32u;
  }

  let out_g0 = out_base + o0;
  if (out_g0 >= out_d) { return; }

  // Load 4 scales and 4 biases once for all 4 tokens
  let sc = vec4<f32>(
    f32_aux[params.s_off + min(out_g0 + 0u, out_d - 1u)],
    f32_aux[params.s_off + min(out_g0 + 1u, out_d - 1u)],
    f32_aux[params.s_off + min(out_g0 + 2u, out_d - 1u)],
    f32_aux[params.s_off + min(out_g0 + 3u, out_d - 1u)]
  );
  let bi = vec4<f32>(
    f32_aux[params.b_off + min(out_g0 + 0u, out_d - 1u)],
    f32_aux[params.b_off + min(out_g0 + 1u, out_d - 1u)],
    f32_aux[params.b_off + min(out_g0 + 2u, out_d - 1u)],
    f32_aux[params.b_off + min(out_g0 + 3u, out_d - 1u)]
  );

  for (var dt: u32 = 0u; dt < 4u; dt++) {
    let tok = tok_base + t0 + dt;
    if (tok >= params.n_tokens) { break; }
    var v4 = a0;
    if (dt == 1u) { v4 = a1; }
    else if (dt == 2u) { v4 = a2; }
    else if (dt == 3u) { v4 = a3; }
    v4 = v4 * sc + bi;
    let row_y = tok * out_d + out_g0;
    for (var d_o: u32 = 0u; d_o < 4u; d_o++) {
      let oi = out_g0 + d_o;
      if (oi >= out_d) { break; }
      var val = v4[d_o];
      if (params.act_mode == 1u) {
        val = gelu_tanh(val);
      }
      if (params.add_res == 1u) {
        val += res_in[row_y + d_o];
      }
      y_out[row_y + d_o] = val;
    }
  }
}
`;
}

/**
 * Single-dispatch Fused Q, K, V builder from the packed [N, 624] QKV+Geo linear output.
 * Uses only 5 storage bindings (well under Apple M1's 8-storage-buffer limit).
 */
export const WGSL_BUILD_SE3_QKV_FUSED = /* wgsl */ `
struct Params {
  n_tokens: u32,
  n_heads: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> geo: array<f32>;       // [N, 16]
@group(0) @binding(2) var<storage, read> qkv_geo: array<f32>;   // [N, 624]
@group(0) @binding(3) var<storage, read_write> Q_out: array<f32>; // [N, 8, 32]
@group(0) @binding(4) var<storage, read_write> K_out: array<f32>; // [N, 8, 32]
@group(0) @binding(5) var<storage, read_write> V_out: array<f32>; // [N, 8, 32]

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

  let H = params.n_heads;
  let i = idx / H;
  let h = idx - i * H;

  let gb = i * 16u;
  let p = vec3<f32>(geo[gb + 0u], geo[gb + 1u], geo[gb + 2u]);
  let pn2 = geo[gb + 3u];
  let T = vec3<f32>(geo[gb + 4u], geo[gb + 5u], geo[gb + 6u]);
  let B = vec3<f32>(geo[gb + 8u], geo[gb + 9u], geo[gb + 10u]);
  let N = vec3<f32>(geo[gb + 12u], geo[gb + 13u], geo[gb + 14u]);

  // Row layout inside qkv_geo[i * 624 + ...]:
  //   qSem: 0..151 (H*19), kSem: 152..303 (H*19), vInv: 304..511 (H*26),
  //   geoQ: 512..567 (H*7), geoK: 568..623 (H*7)
  let row = i * (H * 78u);
  let q_sem_b = row + h * 19u;
  let k_sem_b = row + H * 19u + h * 19u;
  let v_inv_b = row + H * 38u + h * 26u;
  let gq_b    = row + H * 64u + h * 7u;
  let gk_b    = row + H * 71u + h * 7u;

  let out_base = idx * 32u;
  let sem_scale = 0.22941573387056177;

  for (var d: u32 = 0u; d < 19u; d++) {
    Q_out[out_base + d] = qkv_geo[q_sem_b + d] * sem_scale;
    K_out[out_base + d] = qkv_geo[k_sem_b + d];
  }

  let alpha = softplus_clamped(qkv_geo[gq_b + 0u]);
  let u_w = qkv_geo[gq_b + 1u] * T + qkv_geo[gq_b + 2u] * B + qkv_geo[gq_b + 3u] * N;
  let m_w = qkv_geo[gq_b + 4u] * T + qkv_geo[gq_b + 5u] * B + qkv_geo[gq_b + 6u] * N;

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

  let beta = softplus_clamped(qkv_geo[gk_b + 0u]);
  let v_w = qkv_geo[gk_b + 1u] * T + qkv_geo[gk_b + 2u] * B + qkv_geo[gk_b + 3u] * N;
  let r_w = qkv_geo[gk_b + 4u] * T + qkv_geo[gk_b + 5u] * B + qkv_geo[gk_b + 6u] * N;

  let k3 = 2.0 * beta * p + v_w;
  K_out[out_base + 19u] = p.x;
  K_out[out_base + 20u] = p.y;
  K_out[out_base + 21u] = p.z;
  K_out[out_base + 22u] = 1.0;
  K_out[out_base + 23u] = pn2;
  K_out[out_base + 24u] = k3.x;
  K_out[out_base + 25u] = k3.y;
  K_out[out_base + 26u] = k3.z;
  K_out[out_base + 27u] = -beta * pn2 - dot(v_w, p);
  K_out[out_base + 28u] = -beta;
  K_out[out_base + 29u] = r_w.x;
  K_out[out_base + 30u] = r_w.y;
  K_out[out_base + 31u] = r_w.z;

  for (var d: u32 = 0u; d < 26u; d++) {
    V_out[out_base + d] = qkv_geo[v_inv_b + d];
  }
  V_out[out_base + 26u] = p.x;
  V_out[out_base + 27u] = p.y;
  V_out[out_base + 28u] = p.z;
  V_out[out_base + 29u] = N.x;
  V_out[out_base + 30u] = N.y;
  V_out[out_base + 31u] = N.z;
}
`;

/**
 * Zero-readback GPU World SH Packer: writes the exact 6-vec4 (24-float) per-patch layout
 * consumed by `radiantGINode.ts` directly on GPU (eliminates mapAsync + CPU re-upload on drag!).
 */
export const WGSL_PACK_WORLD_SH = /* wgsl */ `
struct Params {
  n_tokens: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tokens: array<f32>;     // [N, 21]
@group(0) @binding(2) var<storage, read> sh_local: array<f32>;   // [N, 12]
@group(0) @binding(3) var<storage, read_write> world_out: array<f32>; // [N, 24]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n_tokens) { return; }

  let tb = i * 21u;
  let sb = i * 12u;
  let db = i * 24u;

  let px = tokens[tb + 0u]; let py = tokens[tb + 1u]; let pz = tokens[tb + 2u];
  let T = vec3<f32>(tokens[tb + 3u], tokens[tb + 4u], tokens[tb + 5u]);
  let B = vec3<f32>(tokens[tb + 6u], tokens[tb + 7u], tokens[tb + 8u]);
  let N = vec3<f32>(tokens[tb + 9u], tokens[tb + 10u], tokens[tb + 11u]);
  let area = max(tokens[tb + 12u], 1e-4);
  let radius = sqrt(area * 0.3183098861837907);

  let i0r = sh_local[sb + 0u]; let i0g = sh_local[sb + 1u]; let i0b = sh_local[sb + 2u];
  let itr = sh_local[sb + 3u]; let itg = sh_local[sb + 4u]; let itb = sh_local[sb + 5u];
  let ibr = sh_local[sb + 6u]; let ibg = sh_local[sb + 7u]; let ibb = sh_local[sb + 8u];
  let inr = sh_local[sb + 9u]; let ing = sh_local[sb + 10u]; let inb = sh_local[sb + 11u];

  let wr = itr * T + ibr * B + inr * N;
  let wg = itg * T + ibg * B + ing * N;
  let wb = itb * T + ibb * B + inb * N;

  world_out[db + 0u] = px; world_out[db + 1u] = py; world_out[db + 2u] = pz; world_out[db + 3u] = radius;
  world_out[db + 4u] = N.x; world_out[db + 5u] = N.y; world_out[db + 6u] = N.z; world_out[db + 7u] = area;
  world_out[db + 8u] = i0r; world_out[db + 9u] = i0g; world_out[db + 10u] = i0b; world_out[db + 11u] = inr;
  world_out[db + 12u] = wr.x; world_out[db + 13u] = wr.y; world_out[db + 14u] = wr.z; world_out[db + 15u] = ing;
  world_out[db + 16u] = wg.x; world_out[db + 17u] = wg.y; world_out[db + 18u] = wg.z; world_out[db + 19u] = inb;
  world_out[db + 20u] = wb.x; world_out[db + 21u] = wb.y; world_out[db + 22u] = wb.z; world_out[db + 23u] = 0.0;
}
`;
