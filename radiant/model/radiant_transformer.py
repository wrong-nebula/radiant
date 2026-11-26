"""Exact SE(3)-Invariant Form-Factor Attention Transformer (H2) and Naive XYZ Baseline (H1).

Architecture designed for:
  1. Strict SE(3) invariance under arbitrary global rigid transforms (R, t) to float precision.
  2. Zero O(N^2) pairwise bias matrix materialization (pure FlashAttention / SDPA with d_qk=d_v=32).
  3. Exact bit-reproducible WebGPU WGSL Compute shader execution on Apple M1 Metal 3 (< 20 MB FP16).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from radiant.sh_math import POS_SCALE_METERS

INTRINSIC_FEAT_DIM = 13


def build_intrinsic_features(tokens: torch.Tensor) -> torch.Tensor:
    """Extract strictly SE(3)-invariant per-patch features [B, N, 13] from packed tokens [B, N, 21].

    Packed token layout:
      [0:3]   pos
      [3:6]   T
      [6:9]   B
      [9:12]  N
      [12]    area
      [13:16] albedo
      [16]    roughness
      [17]    metallic
      [18:21] emission
    """
    area = torch.clamp(tokens[..., 12:13], min=0.0)
    albedo = torch.clamp(tokens[..., 13:16], min=0.0, max=1.0)
    rough = torch.clamp(tokens[..., 16:17], min=0.0, max=1.0)
    metal = torch.clamp(tokens[..., 17:18], min=0.0, max=1.0)
    emit = torch.clamp(tokens[..., 18:21], min=0.0)

    log_area = torch.log1p(area)
    log_emit = torch.log1p(emit)
    is_emit = (emit.amax(dim=-1, keepdim=True) > 1e-3).to(tokens.dtype)
    log_flux = torch.log1p(emit * area)

    return torch.cat([log_area, albedo, rough, metal, log_emit, is_emit, log_flux], dim=-1)


def decode_sh_raw(raw: torch.Tensor) -> torch.Tensor:
    """Decode raw 12-channel head output into physically valid Order-1 SH [..., 12].

    Uses smooth F.softplus for I0 (strictly positive, bounded gradient in (0, 1],
    zero dead-ReLU collapse and zero exp() explosion).
    Guarantees:
      I0 > 0
      I1_T in [-I0, I0], I1_B in [-I0, I0], I1_N in [0, I0]
    """
    i0 = torch.clamp(F.softplus(raw[..., 0:3]), max=64.0)
    dT = torch.tanh(raw[..., 3:6])
    dB = torch.tanh(raw[..., 6:9])
    dN = torch.sigmoid(raw[..., 9:12])
    return torch.cat([i0, i0 * dT, i0 * dB, i0 * dN], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms).to(orig_dtype) * self.weight


class RadiantSE3FormFactorAttention(nn.Module):
    """Exact SE(3)-Invariant Form-Factor Self-Attention with d_qk = d_v = 32 (FlashAttention-ready).

    For each head h:
      Q_{i,h} . K_{j,h} =
        (q_sem . k_sem) / sqrt(19)
        - (alpha_{i,h} + beta_{j,h}) ||p_hat_i - p_hat_j||^2
        + u_{i,h} . (p_hat_j - p_hat_i)
        + v_{j,h} . (p_hat_i - p_hat_j)
        + m_{i,h} . r_{j,h}

    Value aggregation concatenates [v_inv (26), p_hat_j (3), N_j (3)] -> 32 dims, then projects
    (pos_bar - p_hat_i) and nrm_bar into patch i's local orthonormal frame (T_i, B_i, N_i),
    yielding 6 exact SE(3)-invariant directional geometric readout features per head.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8) -> None:
        super().__init__()
        assert d_model == n_heads * 32, "Requires d_head == 32 (d_sem=19 + d_geo=13, d_v_inv=26 + d_v_geo=6)"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_sem = 19
        self.d_geo = 13
        self.d_v_inv = 26
        self.sem_scale = 1.0 / math.sqrt(self.d_sem)

        self.w_q_sem = nn.Linear(d_model, n_heads * self.d_sem, bias=True)
        self.w_k_sem = nn.Linear(d_model, n_heads * self.d_sem, bias=True)
        self.w_v_inv = nn.Linear(d_model, n_heads * self.d_v_inv, bias=True)
        # Per-head geometric scalars: [alpha, uT, uB, uN, mT, mB, mN] (7) and [beta, vT, vB, vN, rT, rB, rN] (7)
        self.w_geo_q = nn.Linear(d_model, n_heads * 7, bias=True)
        self.w_geo_k = nn.Linear(d_model, n_heads * 7, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)

        self._init_geo_params()

    def _init_geo_params(self) -> None:
        nn.init.normal_(self.w_geo_q.weight, std=0.05)
        nn.init.normal_(self.w_geo_k.weight, std=0.05)
        nn.init.zeros_(self.w_geo_q.bias)
        nn.init.zeros_(self.w_geo_k.bias)
        with torch.no_grad():
            # Initialize alpha/beta bias to -1.0 -> softplus(-1.0) ~ 0.31 gentle room-scale distance kernel
            for h in range(self.n_heads):
                self.w_geo_q.bias[h * 7 + 0] = -1.0 + 0.25 * (h - 3.5)
                self.w_geo_k.bias[h * 7 + 0] = -1.0 + 0.25 * (h - 3.5)

    def forward(
        self,
        x: torch.Tensor,         # [B, N, d_model]
        p_hat: torch.Tensor,     # [B, N, 3] centered & scaled positions (float32)
        p_norm2: torch.Tensor,   # [B, N, 1] ||p_hat||^2 (float32)
        T: torch.Tensor,         # [B, N, 3] unit tangent (float32)
        B_vec: torch.Tensor,     # [B, N, 3] unit bitangent (float32)
        N_vec: torch.Tensor,     # [B, N, 3] unit normal (float32)
        attn_mask: torch.Tensor | None = None,  # [B, 1, 1, N] bool (True = valid key)
    ) -> torch.Tensor:
        B_sz, N_tok, _ = x.shape
        H = self.n_heads

        q_sem = self.w_q_sem(x).view(B_sz, N_tok, H, self.d_sem) * self.sem_scale
        k_sem = self.w_k_sem(x).view(B_sz, N_tok, H, self.d_sem)
        v_inv = self.w_v_inv(x).view(B_sz, N_tok, H, self.d_v_inv)

        # Compute geometric Q/K factors in float32 for precision under large translations/rotations
        gq = self.w_geo_q(x).float().view(B_sz, N_tok, H, 7)
        gk = self.w_geo_k(x).float().view(B_sz, N_tok, H, 7)

        alpha = torch.clamp(F.softplus(gq[..., 0:1]), max=20.0)  # [B, N, H, 1]
        beta = torch.clamp(F.softplus(gk[..., 0:1]), max=20.0)

        # Expand local basis to [B, N, 1, 3]
        Te = T.unsqueeze(2)
        Be = B_vec.unsqueeze(2)
        Ne = N_vec.unsqueeze(2)
        pe = p_hat.unsqueeze(2)
        pn2 = p_norm2.unsqueeze(2)

        u_w = gq[..., 1:2] * Te + gq[..., 2:3] * Be + gq[..., 3:4] * Ne
        m_w = gq[..., 4:5] * Te + gq[..., 5:6] * Be + gq[..., 6:7] * Ne
        v_w = gk[..., 1:2] * Te + gk[..., 2:3] * Be + gk[..., 3:4] * Ne
        r_w = gk[..., 4:5] * Te + gk[..., 5:6] * Be + gk[..., 6:7] * Ne

        ones = torch.ones_like(alpha)
        pe_h = pe.expand(B_sz, N_tok, H, 3)
        pn2_h = pn2.expand(B_sz, N_tok, H, 1)

        q_geo = torch.cat(
            [
                2.0 * alpha * pe_h + u_w,                              # 3
                -alpha * pn2_h - (u_w * pe_h).sum(dim=-1, keepdim=True),  # 1
                -alpha,                                                # 1
                pe_h,                                                  # 3
                ones,                                                  # 1
                pn2_h,                                                 # 1
                m_w,                                                   # 3
            ],
            dim=-1,
        ).to(q_sem.dtype)  # [B, N, H, 13]

        k_geo = torch.cat(
            [
                pe_h,                                                  # 3
                ones,                                                  # 1
                pn2_h,                                                 # 1
                2.0 * beta * pe_h + v_w,                               # 3
                -beta * pn2_h - (v_w * pe_h).sum(dim=-1, keepdim=True),  # 1
                -beta,                                                 # 1
                r_w,                                                   # 3
            ],
            dim=-1,
        ).to(k_sem.dtype)  # [B, N, H, 13]

        # Build Q, K, V all [B, H, N, 32]
        Q = torch.cat([q_sem, q_geo], dim=-1).transpose(1, 2)  # [B, H, N, 32]
        K = torch.cat([k_sem, k_geo], dim=-1).transpose(1, 2)  # [B, H, N, 32]

        Ne_h = Ne.expand(B_sz, N_tok, H, 3)
        v_geo = torch.cat([pe_h, Ne_h], dim=-1).to(v_inv.dtype)  # [B, N, H, 6]
        V = torch.cat([v_inv, v_geo], dim=-1).transpose(1, 2)    # [B, H, N, 32]

        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, scale=1.0)
        out = out.transpose(1, 2)  # [B, N, H, 32]

        out_inv = out[..., : self.d_v_inv]  # [B, N, H, 26]
        pos_bar = out[..., self.d_v_inv : self.d_v_inv + 3].float()
        nrm_bar = out[..., self.d_v_inv + 3 : self.d_v_inv + 6].float()

        delta_p = pos_bar - pe
        loc_dp = torch.stack(
            [
                (delta_p * Te).sum(dim=-1),
                (delta_p * Be).sum(dim=-1),
                (delta_p * Ne).sum(dim=-1),
            ],
            dim=-1,
        ).to(out_inv.dtype)
        loc_nr = torch.stack(
            [
                (nrm_bar * Te).sum(dim=-1),
                (nrm_bar * Be).sum(dim=-1),
                (nrm_bar * Ne).sum(dim=-1),
            ],
            dim=-1,
        ).to(out_inv.dtype)

        merged = torch.cat([out_inv, loc_dp, loc_nr], dim=-1).reshape(B_sz, N_tok, self.d_model)
        return self.w_o(merged)


class RadiantBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RadiantSE3FormFactorAttention(d_model=d_model, n_heads=n_heads)
        self.norm2 = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        p_hat: torch.Tensor,
        p_norm2: torch.Tensor,
        T: torch.Tensor,
        B_vec: torch.Tensor,
        N_vec: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), p_hat, p_norm2, T, B_vec, N_vec, attn_mask=attn_mask)
        h = F.gelu(self.fc1(self.norm2(x)), approximate="tanh")
        x = x + self.fc2(h)
        return x


class RadiantGITransformer(nn.Module):
    """H2: Exact SE(3)-Invariant RADIANT-GI Form-Factor Transformer."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 9,
        d_ff: int = 1024,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.stem1 = nn.Linear(INTRINSIC_FEAT_DIM, d_model, bias=True)
        self.stem2 = nn.Linear(d_model, d_model, bias=True)
        self.blocks = nn.ModuleList([RadiantBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.head1 = nn.Linear(d_model, d_model, bias=True)
        self.head2 = nn.Linear(d_model, 12, bias=True)

    def forward(
        self,
        tokens: torch.Tensor,                    # [B, N, 21]
        valid_mask: torch.Tensor | None = None,  # [B, N] bool (True = real patch, False = padding)
    ) -> torch.Tensor:
        """Return predicted local Order-1 SH [B, N, 12]."""
        pos = tokens[..., 0:3].float()
        T = tokens[..., 3:6].float()
        B_vec = tokens[..., 6:9].float()
        N_vec = tokens[..., 9:12].float()

        # Exact translation-invariant centering over valid patches only
        if valid_mask is not None:
            vm_f = valid_mask.float().unsqueeze(-1)  # [B, N, 1]
            denom = torch.clamp(vm_f.sum(dim=1, keepdim=True), min=1.0)
            centroid = (pos * vm_f).sum(dim=1, keepdim=True) / denom
            attn_mask = valid_mask[:, None, None, :]  # [B, 1, 1, N] bool
        else:
            centroid = pos.mean(dim=1, keepdim=True)
            attn_mask = None

        p_hat = (pos - centroid) / POS_SCALE_METERS
        p_norm2 = (p_hat * p_hat).sum(dim=-1, keepdim=True)

        feats = build_intrinsic_features(tokens)
        x = self.stem2(F.gelu(self.stem1(feats), approximate="tanh"))

        for blk in self.blocks:
            x = blk(x, p_hat, p_norm2, T, B_vec, N_vec, attn_mask=attn_mask)

        x = self.final_norm(x)
        raw = self.head2(F.gelu(self.head1(x), approximate="tanh"))
        return decode_sh_raw(raw.float())


# ---------------------------------------------------------------------------
# H1 Baseline for Slice 0 Falsification: Naive Global-XYZ Transformer
# ---------------------------------------------------------------------------

class NaiveXYZBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.norm2 = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B_sz, N_tok, C = x.shape
        qkv = self.qkv(self.norm1(x)).view(B_sz, N_tok, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B_sz, N_tok, C)
        x = x + self.proj(out)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate="tanh"))
        return x


class NaiveXYZTransformer(nn.Module):
    """H1: Standard global-XYZ Transformer baseline (lacks SE(3) Form-Factor Attention)."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
    ) -> None:
        super().__init__()
        # Input: raw world [pos/4, T, B, N] (12) + intrinsic (13) = 25 floats
        self.stem1 = nn.Linear(25, d_model, bias=True)
        self.stem2 = nn.Linear(d_model, d_model, bias=True)
        self.blocks = nn.ModuleList([NaiveXYZBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.head1 = nn.Linear(d_model, d_model, bias=True)
        self.head2 = nn.Linear(d_model, 12, bias=True)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        pos_scaled = tokens[..., 0:3] / POS_SCALE_METERS
        frame = tokens[..., 3:12]
        intr = build_intrinsic_features(tokens)
        feats = torch.cat([pos_scaled, frame, intr], dim=-1)
        attn_mask = valid_mask[:, None, None, :] if valid_mask is not None else None
        x = self.stem2(F.gelu(self.stem1(feats), approximate="tanh"))
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.final_norm(x)
        raw = self.head2(F.gelu(self.head1(x), approximate="tanh"))
        return decode_sh_raw(raw.float())


def radiant_gi_loss(
    pred_sh: torch.Tensor,      # [B, N, 12] float32
    gt_sh: torch.Tensor,        # [B, N, 12] float32
    valid_mask: torch.Tensor,   # [B, N] bool
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined log-irradiance + tone-mapped PSNR + chroma bleed + directional SH cosine loss."""
    m = valid_mask.float().unsqueeze(-1)  # [B, N, 1]
    n_valid = torch.clamp(m.sum(), min=1.0)

    p_i0 = pred_sh[..., 0:3]
    g_i0 = torch.clamp(gt_sh[..., 0:3], min=0.0)

    # 1. Log-compressed L0 irradiance SmoothL1
    loss_log = (F.smooth_l1_loss(torch.log1p(p_i0), torch.log1p(g_i0), reduction="none", beta=0.05) * m).sum() / (n_valid * 3.0)

    # 2. Smooth Reinhard + stabilized gamma tone-mapped L1 + MSE (bounded derivative near 0!)
    p_r = p_i0 / (1.0 + p_i0)
    g_r = g_i0 / (1.0 + g_i0)
    p_tm = torch.pow(p_r + 0.015, 1.0 / 2.2)
    g_tm = torch.pow(g_r + 0.015, 1.0 / 2.2)
    loss_tm_l1 = ((p_tm - g_tm).abs() * m).sum() / (n_valid * 3.0)
    loss_tm_mse = (((p_tm - g_tm) ** 2) * m).sum() / (n_valid * 3.0)

    # 3. Color-bleed chroma loss (normalized RGB hue on lit patches)
    p_sum = p_i0.sum(dim=-1, keepdim=True) + 0.03
    g_sum = g_i0.sum(dim=-1, keepdim=True) + 0.03
    lit_w = m * torch.clamp(g_i0.amax(dim=-1, keepdim=True) * 4.0, 0.0, 1.0)
    lit_denom = torch.clamp(lit_w.sum(), min=1.0)
    loss_chroma = (((p_i0 / p_sum) - (g_i0 / g_sum)).abs() * lit_w).sum() / (lit_denom * 3.0)

    # 4. Directional Order-1 SH vector L1 + cosine similarity loss
    p_l1 = pred_sh[..., 3:12]
    g_l1 = gt_sh[..., 3:12]
    # Scale-normalized L1 directional regression
    denom_g = g_i0.repeat(1, 1, 3) + 0.03
    loss_sh_l1 = (((p_l1 - g_l1) / denom_g).abs() * lit_w).sum() / (lit_denom * 9.0)

    # Cosine similarity of luminance-weighted L1 direction vector [T, B, N]
    w_rgb = torch.tensor([0.2126, 0.7152, 0.0722], device=pred_sh.device, dtype=pred_sh.dtype)
    p_dir = (p_l1.view(*p_l1.shape[:-1], 3, 3) * w_rgb).sum(dim=-1)  # [B, N, 3]
    g_dir = (g_l1.view(*g_l1.shape[:-1], 3, 3) * w_rgb).sum(dim=-1)  # [B, N, 3]
    cos_sim = F.cosine_similarity(p_dir, g_dir, dim=-1, eps=1e-5).unsqueeze(-1)
    loss_sh_cos = ((1.0 - cos_sim) * lit_w).sum() / lit_denom

    total = (
        1.5 * loss_log
        + 2.0 * loss_tm_l1
        + 4.0 * loss_tm_mse
        + 0.6 * loss_chroma
        + 0.5 * loss_sh_l1
        + 0.25 * loss_sh_cos
    )
    stats = {
        "loss": float(total.detach().item()),
        "log": float(loss_log.detach().item()),
        "tm_l1": float(loss_tm_l1.detach().item()),
        "sh_cos": float(loss_sh_cos.detach().item()),
    }
    return total, stats
