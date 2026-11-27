/**
 * Turbo WGSL Compute Shaders for `RDG8` v2.6 (Apple M1 Metal 3 Peak-Throughput Edition).
 *
 * Profiler-driven optimizations:
 *   1. `buildLinearQ8WGSL`: Coalesced 2D `TILE_M=32 x TILE_N=32` (@workgroup_size(8, 8) = 64 threads,
 *      only 4 KB shared memory -> 8 WGs/core!), 128-bit `array<vec4<f32>>` global loads & stores,
 *      hardware `unpack4x8snorm(u32)`, K-tile `vec4<f16>` inner accumulation with FP32 outer sum.
 *   2. `buildAttentionTurboWGSL`: `@workgroup_size(32)` (2x more WGs saturating M1's 8 GPU cores)
 *      x `TILE_K=32` keys (2x fewer barriers!), `array<vec4<f32>>` 128-bit loads/stores, `vec4<f16>`
 *      K/V tiles (4 KB shared) with FP32 online softmax + single-exp fast path.
 *   3. `WGSL_BUILD_SE3_QKV_FUSED`: 128-bit `array<vec4<f32>>` output stores (24 vec4 stores instead
 *      of 96 scalar stores -> ~5x faster QKV packing).
 *   4. `WGSL_PACK_WORLD_SH`: Zero-readback on-GPU World SH packing directly into Three.js StorageBuffer.
 */

export const WGSL_RMS_NORM_F32 = /* wgsl */ `
struct Params {
  n_tokens: u32,
  dim: u32,
  w_off: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> f32_aux: array<f32>;
@group(0) @binding(3) var<storage, read_write> x_out: array<vec4<f32>>;

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
  let dim4 = dim >> 2u;
  let base4 = tok * dim4;

  var s = 0.0;
  var d4 = tid;
  while (d4 < dim4) {
    let v = x_in[base4 + d4];
    s += dot(v, v);
    d4 += 64u;
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
  d4 = tid;
  while (d4 < dim4) {
    let d = d4 << 2u;
    let w = vec4<f32>(
      f32_aux[params.w_off + d + 0u],
      f32_aux[params.w_off + d + 1u],
      f32_aux[params.w_off + d + 2u],
      f32_aux[params.w_off + d + 3u]
    );
    x_out[base4 + d4] = x_in[base4 + d4] * (inv_rms * w);
    d4 += 64u;
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
 * Coalesced 2D Register-Blocked TILE_M=32 x TILE_N=32 INT8 GEMM (@workgroup_size(8, 8) = 64 threads).
 * Uses 128-bit `array<vec4<f32>>` global loads/stores, 4 KB shared memory (8 WGs/core on M1!),
 * and K-tile `vec4<f16>` inner accumulation with exact FP32 outer accumulation.
 * Note: requires in_dim % 4 == 0 and out_dim % 4 == 0 (all RDG8 Q8 layers are 256, 624, or 1024!).
 */
export function buildLinearQ8WGSL(useF16Tile: boolean): string {
  const enableHeader = useF16Tile ? 'enable f16;\n' : '';
  const tileType = useF16Tile ? 'vec4<f16>' : 'vec4<f32>';
  const tileZero = useF16Tile ? 'vec4<f16>(0.0h)' : 'vec4<f32>(0.0)';
  const loadXExpr = useF16Tile
    ? 'vec4<f16>(x_in[tok * in_v4 + k_v4 + q_loc])'
    : 'x_in[tok * in_v4 + k_v4 + q_loc]';
  const loadWExpr = useF16Tile
    ? 'vec4<f16>(unpack4x8snorm(w_q8[params.w_off_u32 + out_i * in_v4 + k_v4 + q_loc]))'
    : 'unpack4x8snorm(w_q8[params.w_off_u32 + out_i * in_v4 + k_v4 + q_loc])';

  const innerCompute = useF16Tile
    ? `
    var t0_acc = vec4<f16>(0.0h);
    var t1_acc = vec4<f16>(0.0h);
    var t2_acc = vec4<f16>(0.0h);
    var t3_acc = vec4<f16>(0.0h);
    for (var q: u32 = 0u; q < 8u; q++) {
      let x0 = x_tile[xb0 + q];
      let x1 = x_tile[xb1 + q];
      let x2 = x_tile[xb2 + q];
      let x3 = x_tile[xb3 + q];
      let w0 = w_tile[wb0 + q];
      let w1 = w_tile[wb1 + q];
      let w2 = w_tile[wb2 + q];
      let w3 = w_tile[wb3 + q];
      t0_acc += vec4<f16>(dot(x0, w0), dot(x0, w1), dot(x0, w2), dot(x0, w3));
      t1_acc += vec4<f16>(dot(x1, w0), dot(x1, w1), dot(x1, w2), dot(x1, w3));
      t2_acc += vec4<f16>(dot(x2, w0), dot(x2, w1), dot(x2, w2), dot(x2, w3));
      t3_acc += vec4<f16>(dot(x3, w0), dot(x3, w1), dot(x3, w2), dot(x3, w3));
    }
    a0 += vec4<f32>(t0_acc);
    a1 += vec4<f32>(t1_acc);
    a2 += vec4<f32>(t2_acc);
    a3 += vec4<f32>(t3_acc);`
    : `
    for (var q: u32 = 0u; q < 8u; q++) {
      let x0 = x_tile[xb0 + q];
      let x1 = x_tile[xb1 + q];
      let x2 = x_tile[xb2 + q];
      let x3 = x_tile[xb3 + q];
      let w0 = w_tile[wb0 + q];
      let w1 = w_tile[wb1 + q];
      let w2 = w_tile[wb2 + q];
      let w3 = w_tile[wb3 + q];
      a0 += vec4<f32>(dot(x0, w0), dot(x0, w1), dot(x0, w2), dot(x0, w3));
      a1 += vec4<f32>(dot(x1, w0), dot(x1, w1), dot(x1, w2), dot(x1, w3));
      a2 += vec4<f32>(dot(x2, w0), dot(x2, w1), dot(x2, w2), dot(x2, w3));
      a3 += vec4<f32>(dot(x3, w0), dot(x3, w1), dot(x3, w2), dot(x3, w3));
    }`;

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
@group(0) @binding(1) var<storage, read> x_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w_q8: array<u32>;
@group(0) @binding(3) var<storage, read> f32_aux: array<f32>;
@group(0) @binding(4) var<storage, read> res_in: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> y_out: array<vec4<f32>>;

// K_TILE = 32 with padded stride = 9 vec4s (32 x 9 = 288 -> 2.25 KB each = 4.5 KB total, ZERO bank conflicts!)
var<workgroup> x_tile: array<${tileType}, 288>;
var<workgroup> w_tile: array<${tileType}, 288>;

fn gelu_tanh_v4(x: vec4<f32>) -> vec4<f32> {
  let c = 0.7978845608028654;
  let arg = clamp(c * (x + 0.044715 * x * x * x), vec4<f32>(-15.0), vec4<f32>(15.0));
  return 0.5 * x * (vec4<f32>(1.0) + tanh(arg));
}

@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let lx = lid.x; // 0..7 -> 4 outputs each = 32 outputs
  let ly = lid.y; // 0..7 -> 4 tokens each  = 32 tokens
  let tid = ly * 8u + lx; // 0..63
  let tok_base = wg.y * 32u;
  let out_base = wg.x * 32u;
  let in_d = params.in_dim;
  let out_d = params.out_dim;
  let in_v4 = in_d >> 2u;
  let out_v4 = out_d >> 2u;

  let t0 = ly << 2u;
  let o0 = lx << 2u;

  var a0 = vec4<f32>(0.0);
  var a1 = vec4<f32>(0.0);
  var a2 = vec4<f32>(0.0);
  var a3 = vec4<f32>(0.0);

  let xb0 = (t0 + 0u) * 9u;
  let xb1 = (t0 + 1u) * 9u;
  let xb2 = (t0 + 2u) * 9u;
  let xb3 = (t0 + 3u) * 9u;
  let wb0 = (o0 + 0u) * 9u;
  let wb1 = (o0 + 1u) * 9u;
  let wb2 = (o0 + 2u) * 9u;
  let wb3 = (o0 + 3u) * 9u;

  var k_v4: u32 = 0u;
  while (k_v4 < in_v4) {
    // 64 threads load 256 vec4s into padded stride-9 shared tiles (4 coalesced loads each!)
    for (var s: u32 = 0u; s < 4u; s++) {
      let idx = tid + (s << 6u);
      let r_loc = idx >> 3u; // 0..31
      let q_loc = idx & 7u;  // 0..7
      let slot = r_loc * 9u + q_loc;
      let tok = tok_base + r_loc;
      if (tok < params.n_tokens && (k_v4 + q_loc) < in_v4) {
        x_tile[slot] = ${loadXExpr};
      } else {
        x_tile[slot] = ${tileZero};
      }
      let out_i = out_base + r_loc;
      if (out_i < out_d && (k_v4 + q_loc) < in_v4) {
        w_tile[slot] = ${loadWExpr};
      } else {
        w_tile[slot] = ${tileZero};
      }
    }
    workgroupBarrier();
    ${innerCompute}
    workgroupBarrier();
    k_v4 += 8u;
  }

  let out_g0 = out_base + o0;
  if (out_g0 >= out_d) { return; }
  let out_g4 = out_g0 >> 2u;

  let sc = vec4<f32>(
    f32_aux[params.s_off + out_g0 + 0u],
    f32_aux[params.s_off + out_g0 + 1u],
    f32_aux[params.s_off + out_g0 + 2u],
    f32_aux[params.s_off + out_g0 + 3u]
  );
  let bi = vec4<f32>(
    f32_aux[params.b_off + out_g0 + 0u],
    f32_aux[params.b_off + out_g0 + 1u],
    f32_aux[params.b_off + out_g0 + 2u],
    f32_aux[params.b_off + out_g0 + 3u]
  );

  // Coalesced 128-bit vec4<f32> stores!
  var v0 = a0 * sc + bi;
  var v1 = a1 * sc + bi;
  var v2 = a2 * sc + bi;
  var v3 = a3 * sc + bi;
  if (params.act_mode == 1u) {
    v0 = gelu_tanh_v4(v0);
    v1 = gelu_tanh_v4(v1);
    v2 = gelu_tanh_v4(v2);
    v3 = gelu_tanh_v4(v3);
  }

  let tok0 = tok_base + t0;
  if (tok0 + 0u < params.n_tokens) {
    let idx = (tok0 + 0u) * out_v4 + out_g4;
    if (params.add_res == 1u) { v0 += res_in[idx]; }
    y_out[idx] = v0;
  }
  if (tok0 + 1u < params.n_tokens) {
    let idx = (tok0 + 1u) * out_v4 + out_g4;
    if (params.add_res == 1u) { v1 += res_in[idx]; }
    y_out[idx] = v1;
  }
  if (tok0 + 2u < params.n_tokens) {
    let idx = (tok0 + 2u) * out_v4 + out_g4;
    if (params.add_res == 1u) { v2 += res_in[idx]; }
    y_out[idx] = v2;
  }
  if (tok0 + 3u < params.n_tokens) {
    let idx = (tok0 + 3u) * out_v4 + out_g4;
    if (params.add_res == 1u) { v3 += res_in[idx]; }
    y_out[idx] = v3;
  }
}
`;
}

/**
 * High-Occupancy Block-Tiled FlashAttention (@workgroup_size(32) x TILE_K=32 keys, 128-bit vec4 loads,
 * optional f16 K/V tiles + exact FP32 online softmax with single-exp fast path!).
 * Doubles GPU workgroup count (96 WGs vs 48) and halves barriers (11 vs 22) on Apple M1!
 */
export function buildAttentionTurboWGSL(useF16Tile: boolean): string {
  const enableHeader = useF16Tile ? 'enable f16;\n' : '';
  const kvType = useF16Tile ? 'vec4<f16>' : 'vec4<f32>';
  const kvZero = useF16Tile ? 'vec4<f16>(0.0h)' : 'vec4<f32>(0.0)';
  const castKV = (expr: string) => (useF16Tile ? `vec4<f16>(${expr})` : expr);
  const castF32 = (expr: string) => (useF16Tile ? `vec4<f32>(${expr})` : expr);
  const scoreExpr = useF16Tile
    ? `f32(
          dot(q0, K_tile[tb + 0u]) + dot(q1, K_tile[tb + 1u]) +
          dot(q2, K_tile[tb + 2u]) + dot(q3, K_tile[tb + 3u]) +
          dot(q4, K_tile[tb + 4u]) + dot(q5, K_tile[tb + 5u]) +
          dot(q6, K_tile[tb + 6u]) + dot(q7, K_tile[tb + 7u])
        )`
    : `dot(q0, K_tile[tb + 0u]) + dot(q1, K_tile[tb + 1u]) +
        dot(q2, K_tile[tb + 2u]) + dot(q3, K_tile[tb + 3u]) +
        dot(q4, K_tile[tb + 4u]) + dot(q5, K_tile[tb + 5u]) +
        dot(q6, K_tile[tb + 6u]) + dot(q7, K_tile[tb + 7u])`;

  return /* wgsl */ `${enableHeader}
struct Params {
  n_tokens: u32,
  n_heads: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> geo: array<vec4<f32>>;         // [N, 4]
@group(0) @binding(2) var<storage, read> Q_in: array<vec4<f32>>;        // [N*H, 8]
@group(0) @binding(3) var<storage, read> K_in: array<vec4<f32>>;        // [N*H, 8]
@group(0) @binding(4) var<storage, read> V_in: array<vec4<f32>>;        // [N*H, 8]
@group(0) @binding(5) var<storage, read_write> out_merged: array<vec4<f32>>; // [N*H, 8]

// 32 keys x 8 vec4s = 256 vec4s (2 KB each in f16 = 4 KB total -> half the barriers on M1!)
var<workgroup> K_tile: array<${kvType}, 256>;
var<workgroup> V_tile: array<${kvType}, 256>;

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let h = wg.x;
  let i = wg.y * 32u + lid.x;
  let tid = lid.x; // 0..31
  let N_tok = params.n_tokens;
  let H = params.n_heads;
  let valid_i = i < N_tok && h < H;

  var q0 = ${kvZero}; var q1 = ${kvZero}; var q2 = ${kvZero}; var q3 = ${kvZero};
  var q4 = ${kvZero}; var q5 = ${kvZero}; var q6 = ${kvZero}; var q7 = ${kvZero};
  if (valid_i) {
    let qb = (i * H + h) << 3u;
    q0 = ${castKV('Q_in[qb + 0u]')}; q1 = ${castKV('Q_in[qb + 1u]')};
    q2 = ${castKV('Q_in[qb + 2u]')}; q3 = ${castKV('Q_in[qb + 3u]')};
    q4 = ${castKV('Q_in[qb + 4u]')}; q5 = ${castKV('Q_in[qb + 5u]')};
    q6 = ${castKV('Q_in[qb + 6u]')}; q7 = ${castKV('Q_in[qb + 7u]')};
  }

  var m_max = -1e30;
  var l_sum = 0.0;
  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
  var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0); var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);

  var j_base: u32 = 0u;
  while (j_base < N_tok) {
    let tile_len = min(32u, N_tok - j_base);
    // 32 threads cooperatively load 256 vec4s (8 coalesced 128-bit loads/thread!)
    for (var s: u32 = 0u; s < 8u; s++) {
      let slot = tid + (s << 5u); // 0..255
      let kj_loc = slot >> 3u;    // 0..31
      let v_idx = slot & 7u;      // 0..7
      let j = j_base + kj_loc;
      if (j < N_tok) {
        let kb = ((j * H + h) << 3u) + v_idx;
        K_tile[slot] = ${castKV('K_in[kb]')};
        V_tile[slot] = ${castKV('V_in[kb]')};
      }
    }
    workgroupBarrier();

    if (valid_i) {
      for (var kj: u32 = 0u; kj < tile_len; kj++) {
        let tb = kj << 3u;
        let score = ${scoreExpr};
        if (j_base == 0u && kj == 0u) {
          m_max = score;
          l_sum = 1.0;
          a0 = ${castF32('V_tile[tb + 0u]')}; a1 = ${castF32('V_tile[tb + 1u]')};
          a2 = ${castF32('V_tile[tb + 2u]')}; a3 = ${castF32('V_tile[tb + 3u]')};
          a4 = ${castF32('V_tile[tb + 4u]')}; a5 = ${castF32('V_tile[tb + 5u]')};
          a6 = ${castF32('V_tile[tb + 6u]')}; a7 = ${castF32('V_tile[tb + 7u]')};
        } else if (score <= m_max) {
          let w = exp2(max(score - m_max, -80.0) * 1.4426950408889634);
          l_sum += w;
          a0 += ${castF32('V_tile[tb + 0u]')} * w;
          a1 += ${castF32('V_tile[tb + 1u]')} * w;
          a2 += ${castF32('V_tile[tb + 2u]')} * w;
          a3 += ${castF32('V_tile[tb + 3u]')} * w;
          a4 += ${castF32('V_tile[tb + 4u]')} * w;
          a5 += ${castF32('V_tile[tb + 5u]')} * w;
          a6 += ${castF32('V_tile[tb + 6u]')} * w;
          a7 += ${castF32('V_tile[tb + 7u]')} * w;
        } else {
          let rescale = exp2(max(m_max - score, -80.0) * 1.4426950408889634);
          l_sum = l_sum * rescale + 1.0;
          a0 = a0 * rescale + ${castF32('V_tile[tb + 0u]')};
          a1 = a1 * rescale + ${castF32('V_tile[tb + 1u]')};
          a2 = a2 * rescale + ${castF32('V_tile[tb + 2u]')};
          a3 = a3 * rescale + ${castF32('V_tile[tb + 3u]')};
          a4 = a4 * rescale + ${castF32('V_tile[tb + 4u]')};
          a5 = a5 * rescale + ${castF32('V_tile[tb + 5u]')};
          a6 = a6 * rescale + ${castF32('V_tile[tb + 6u]')};
          a7 = a7 * rescale + ${castF32('V_tile[tb + 7u]')};
          m_max = score;
        }
      }
    }
    workgroupBarrier();
    j_base += 32u;
  }

  if (!valid_i) { return; }

  let inv_l = 1.0 / max(l_sum, 1e-12);
  a0 *= inv_l; a1 *= inv_l; a2 *= inv_l; a3 *= inv_l;
  a4 *= inv_l; a5 *= inv_l; a6 *= inv_l; a7 *= inv_l;

  let gb = i << 2u;
  let p_i = geo[gb + 0u].xyz;
  let T = geo[gb + 1u].xyz;
  let B = geo[gb + 2u].xyz;
  let N = geo[gb + 3u].xyz;
  let dp = vec3<f32>(a6.z, a6.w, a7.x) - p_i;
  let nb = vec3<f32>(a7.y, a7.z, a7.w);

  let ob = (i * H + h) << 3u;
  out_merged[ob + 0u] = a0;
  out_merged[ob + 1u] = a1;
  out_merged[ob + 2u] = a2;
  out_merged[ob + 3u] = a3;
  out_merged[ob + 4u] = a4;
  out_merged[ob + 5u] = a5;
  out_merged[ob + 6u] = vec4<f32>(a6.x, a6.y, dot(dp, T), dot(dp, B));
  out_merged[ob + 7u] = vec4<f32>(dot(dp, N), dot(nb, T), dot(nb, B), dot(nb, N));
}
`;
}

/**
 * Single-dispatch Fused Q, K, V builder using 128-bit `array<vec4<f32>>` output stores!
 * Cuts 96 scalar stores -> 24 vec4 stores (~5x faster on Apple M1 Metal 3!).
 */
export const WGSL_BUILD_SE3_QKV_FUSED = /* wgsl */ `
struct Params {
  n_tokens: u32,
  n_heads: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> geo: array<vec4<f32>>;         // [N, 4]
@group(0) @binding(2) var<storage, read> qkv_geo: array<f32>;           // [N, 624]
@group(0) @binding(3) var<storage, read_write> Q_out: array<vec4<f32>>; // [N*H, 8]
@group(0) @binding(4) var<storage, read_write> K_out: array<vec4<f32>>; // [N*H, 8]
@group(0) @binding(5) var<storage, read_write> V_out: array<vec4<f32>>; // [N*H, 8]

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
  let H = params.n_heads;
  let total = params.n_tokens * H;
  if (idx >= total) { return; }

  let i = idx / H;
  let h = idx - i * H;

  let gb = i << 2u;
  let g0 = geo[gb + 0u];
  let p = g0.xyz;
  let pn2 = g0.w;
  let g1 = geo[gb + 1u];
  let T = g1.xyz;
  let log_area_prior = g1.w;
  let B = geo[gb + 2u].xyz;
  let N = geo[gb + 3u].xyz;

  let row = i * (H * 78u);
  let qb = row + h * 19u;
  let kb = row + H * 19u + h * 19u;
  let vb = row + H * 38u + h * 26u;
  let gqb = row + H * 64u + h * 7u;
  let gkb = row + H * 71u + h * 7u;

  let ob = idx << 3u;
  let ss = 0.22941573387056177;

  let alpha = softplus_clamped(qkv_geo[gqb + 0u]);
  let u_w = qkv_geo[gqb + 1u] * T + qkv_geo[gqb + 2u] * B + qkv_geo[gqb + 3u] * N;
  let m_w = qkv_geo[gqb + 4u] * T + qkv_geo[gqb + 5u] * B + qkv_geo[gqb + 6u] * N;
  let q0 = 2.0 * alpha * p + u_w;

  Q_out[ob + 0u] = vec4<f32>(qkv_geo[qb + 0u], qkv_geo[qb + 1u], qkv_geo[qb + 2u], qkv_geo[qb + 3u]) * ss;
  Q_out[ob + 1u] = vec4<f32>(qkv_geo[qb + 4u], qkv_geo[qb + 5u], qkv_geo[qb + 6u], qkv_geo[qb + 7u]) * ss;
  Q_out[ob + 2u] = vec4<f32>(qkv_geo[qb + 8u], qkv_geo[qb + 9u], qkv_geo[qb + 10u], qkv_geo[qb + 11u]) * ss;
  Q_out[ob + 3u] = vec4<f32>(qkv_geo[qb + 12u], qkv_geo[qb + 13u], qkv_geo[qb + 14u], qkv_geo[qb + 15u]) * ss;
  Q_out[ob + 4u] = vec4<f32>(qkv_geo[qb + 16u] * ss, qkv_geo[qb + 17u] * ss, qkv_geo[qb + 18u] * ss, q0.x);
  Q_out[ob + 5u] = vec4<f32>(q0.y, q0.z, -alpha * pn2 - dot(u_w, p), -alpha);
  Q_out[ob + 6u] = vec4<f32>(p.x, p.y, p.z, 1.0);
  Q_out[ob + 7u] = vec4<f32>(pn2, m_w.x, m_w.y, m_w.z);

  let beta = softplus_clamped(qkv_geo[gkb + 0u]);
  let v_w = qkv_geo[gkb + 1u] * T + qkv_geo[gkb + 2u] * B + qkv_geo[gkb + 3u] * N;
  let r_w = qkv_geo[gkb + 4u] * T + qkv_geo[gkb + 5u] * B + qkv_geo[gkb + 6u] * N;
  let k3 = 2.0 * beta * p + v_w;

  K_out[ob + 0u] = vec4<f32>(qkv_geo[kb + 0u], qkv_geo[kb + 1u], qkv_geo[kb + 2u], qkv_geo[kb + 3u]);
  K_out[ob + 1u] = vec4<f32>(qkv_geo[kb + 4u], qkv_geo[kb + 5u], qkv_geo[kb + 6u], qkv_geo[kb + 7u]);
  K_out[ob + 2u] = vec4<f32>(qkv_geo[kb + 8u], qkv_geo[kb + 9u], qkv_geo[kb + 10u], qkv_geo[kb + 11u]);
  K_out[ob + 3u] = vec4<f32>(qkv_geo[kb + 12u], qkv_geo[kb + 13u], qkv_geo[kb + 14u], qkv_geo[kb + 15u]);
  K_out[ob + 4u] = vec4<f32>(qkv_geo[kb + 16u], qkv_geo[kb + 17u], qkv_geo[kb + 18u], p.x);
  K_out[ob + 5u] = vec4<f32>(p.y, p.z, 1.0, pn2);
  K_out[ob + 6u] = vec4<f32>(k3.x, k3.y, k3.z, -beta * pn2 - dot(v_w, p) + log_area_prior);
  K_out[ob + 7u] = vec4<f32>(-beta, r_w.x, r_w.y, r_w.z);

  V_out[ob + 0u] = vec4<f32>(qkv_geo[vb + 0u], qkv_geo[vb + 1u], qkv_geo[vb + 2u], qkv_geo[vb + 3u]);
  V_out[ob + 1u] = vec4<f32>(qkv_geo[vb + 4u], qkv_geo[vb + 5u], qkv_geo[vb + 6u], qkv_geo[vb + 7u]);
  V_out[ob + 2u] = vec4<f32>(qkv_geo[vb + 8u], qkv_geo[vb + 9u], qkv_geo[vb + 10u], qkv_geo[vb + 11u]);
  V_out[ob + 3u] = vec4<f32>(qkv_geo[vb + 12u], qkv_geo[vb + 13u], qkv_geo[vb + 14u], qkv_geo[vb + 15u]);
  V_out[ob + 4u] = vec4<f32>(qkv_geo[vb + 16u], qkv_geo[vb + 17u], qkv_geo[vb + 18u], qkv_geo[vb + 19u]);
  V_out[ob + 5u] = vec4<f32>(qkv_geo[vb + 20u], qkv_geo[vb + 21u], qkv_geo[vb + 22u], qkv_geo[vb + 23u]);
  V_out[ob + 6u] = vec4<f32>(qkv_geo[vb + 24u], qkv_geo[vb + 25u], p.x, p.y);
  V_out[ob + 7u] = vec4<f32>(p.z, N.x, N.y, N.z);
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
@group(0) @binding(3) var<storage, read_write> world_out: array<vec4<f32>>; // [N, 6]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n_tokens) { return; }

  let tb = i * 21u;
  let sb = i * 12u;
  let db = i * 6u;

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

  world_out[db + 0u] = vec4<f32>(px, py, pz, radius);
  world_out[db + 1u] = vec4<f32>(N.x, N.y, N.z, area);
  world_out[db + 2u] = vec4<f32>(i0r, i0g, i0b, inr);
  world_out[db + 3u] = vec4<f32>(wr.x, wr.y, wr.z, ing);
  world_out[db + 4u] = vec4<f32>(wg.x, wg.y, wg.z, inb);
  world_out[db + 5u] = vec4<f32>(wb.x, wb.y, wb.z, 0.0);
}
`;
