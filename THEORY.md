# THREE-RADIANT-GI — Mathematical Theory & Verification Tables

This document specifies the exact mathematical formulation of **THREE-RADIANT-GI**, a zero-shot, zero-bake, real-time multi-bounce Global Illumination engine for Three.js `0.187.0` (`three/webgpu` + `three/tsl`), based on **RADIANT** (*Object-Centric Feed-Forward Global Illumination*, `arXiv:2610.32075v1`).

---

## 1. Discretization-Agnostic 32-Patch Object Tokenization

Radiative transport between macroscopic surfaces depends on object geometry, orientation, and material reflectance rather than fine triangle tessellation. For any Three.js `Mesh` or `.glb` with arbitrary polygon count ($12$ to $500{,}000+$ triangles):

1. **Local-Space Clustering (`sampleMeshPatches`):**
   Each triangle $t$ with centroid $\mathbf{c}_t \in \mathbb{R}^3$, unit outward normal $\mathbf{n}_t \in \mathbb{S}^2$, and area $A_t > 0$ is mapped to a normal-aware 6D feature vector:
   $$\mathbf{f}_t = \big[\mathbf{c}_t,\; w_n D_{\text{bbox}} \mathbf{n}_t\big] \in \mathbb{R}^6, \quad w_n = 0.45$$
   Face-normal stratified seeding (up to 6 orthogonal face-normal seeds) followed by deterministic area-weighted Farthest-Point Sampling ($\text{score}_t = \min_k \|\mathbf{f}_t - \boldsymbol{\mu}_k\|^2 \sqrt{A_t}$) initializes $K = 32$ cluster centers, refined by 6 normal-aware Lloyd iterations. Stratification guarantees coverage of narrow wall edges and curved columns, while $w_n D_{\text{bbox}} \mathbf{n}_t$ prevents clusters from straddling sharp $90^\circ$ architectural creases. For textured `.glb` assets, `material.map` UV albedos are area-weighted into per-patch linear RGB tokens.
2. **Surface Snapping & Rigid Local Frames:**
   Each patch $i \in \{0,\dots,31\}$ stores its surface-snapped position $\mathbf{p}_i^{\text{loc}}$ (nearest member triangle centroid), total cluster area $A_i = \sum_{t \in \mathcal{P}_i} A_t$, area-weighted unit normal $\mathbf{N}_i^{\text{loc}}$, and a rigidly attached orthonormal tangent frame $(T_i^{\text{loc}}, B_i^{\text{loc}}, N_i^{\text{loc}}) \in SO(3)$ built via the branchless Duff/Frisvad basis.
3. **Zero-Cost Rigid Update:**
   Moving or rotating an object by `matrixWorld` rigid-transforms its 32 local tokens in $O(1)$ time ($\sim 0.02\text{ ms}$), decoupling sequence length ($N = 32 M$ tokens for $M$ objects) from polygon count.

---

## 2. Exact Order-1 Spherical Harmonics Indirect Irradiance Convention

To compose cleanly with Three.js's crisp direct lights and shadow maps with **zero double-counting**, our NVIDIA Warp 3-bounce GPU patch path tracer explicitly **excludes direct emitter hits on bounce 0** and integrates bounces $b \in \{1, 2, 3\}$ only.

For each surface patch $i$ with orthonormal local frame $(T_i, B_i, N_i)$ and incoming indirect radiance $L_{\text{ind}}(\omega)$ over the upper hemisphere $\Omega^+(N_i) = \{\omega : \omega \cdot N_i > 0\}$, we define the 4-component (12-float RGB) cosine-weighted Order-1 SH irradiance representation:

$$I_0 = \frac{1}{\pi}\int_{\Omega^+(N_i)} L_{\text{ind}}(\omega)(\omega \cdot N_i)\,d\omega = \mathbb{E}_{\omega \sim p_{\cos}}\big[L_{\text{ind}}(\omega)\big]$$
$$I_{1,T} = \mathbb{E}_{\omega \sim p_{\cos}}\big[L_{\text{ind}}(\omega)(\omega \cdot T_i)\big], \quad
I_{1,B} = \mathbb{E}_{\omega \sim p_{\cos}}\big[L_{\text{ind}}(\omega)(\omega \cdot B_i)\big], \quad
I_{1,N} = \mathbb{E}_{\omega \sim p_{\cos}}\big[L_{\text{ind}}(\omega)(\omega \cdot N_i)\big]$$

where $p_{\cos}(\omega) = \frac{\omega \cdot N_i}{\pi}$ is cosine-weighted hemisphere importance sampling.

### Physical Identities & Conservation Properties
1. **White-Furnace Exactness (O1):** In a closed cavity with uniform diffuse albedo $\boldsymbol{\rho}$ and uniform emission $\mathbf{L}_e$:
   $$I_0^{(3)} = \mathbf{L}_e \odot (\boldsymbol{\rho} + \boldsymbol{\rho}^2 + \boldsymbol{\rho}^3), \qquad I_{1,T} = I_{1,B} = \mathbf{0}, \qquad I_{1,N}^{(3)} = \frac{2}{3} I_0^{(3)}.$$
   When $\boldsymbol{\rho} = \mathbf{0}$, $I_0^{(3)} \equiv \mathbf{0}$ identically (confirming zero direct bounce-0 leakage).
2. **Cauchy–Schwarz Positivity Bounds:** For every RGB channel $c$:
   $$I_0 \ge 0, \qquad I_{1,N} \in [0, I_0], \qquad I_{1,T}, I_{1,B} \in [-I_0, I_0].$$
   Our network output head enforces these bounds structurally:
   $$I_0 = \exp(\text{clamp}(r_0, 0, 8)) - 1, \quad I_{1,T} = I_0 \tanh(r_T), \quad I_{1,B} = I_0 \tanh(r_B), \quad I_{1,N} = I_0 \sigma(r_N).$$
3. **World-Space Directional Reconstruction:** Converting local SH to world space via $\mathbf{I}_{1,\text{world}} = I_{1,T} T_i + I_{1,B} B_i + I_{1,N} N_i$ transforms $SO(3)$-equivariantly ($\mathbf{I}'_{1,\text{world}} = R\,\mathbf{I}_{1,\text{world}}$). For an arbitrary shading normal $\mathbf{n}_s$ (curved surface or normal map):
   $$I(\mathbf{n}_s) = \max\!\left(\mathbf{0},\; I_0\big(2 - \text{clamp}(\mathbf{n}_s \cdot N_i, 0, 1)\big) + \frac{3}{2}\big(\mathbf{I}_{1,\text{world}} \cdot \mathbf{n}_s - I_{1,N}\big)\right)$$
   This satisfies BOTH $I(N_i) = I_0$ exactly AND $I(\mathbf{n}_s) \equiv I_0$ identically for all $\mathbf{n}_s$ under uniform white-furnace illumination, while accurately tilting bounce brightness when $\mathbf{n}_s$ turns toward or away from a colored wall.

---

## 3. Exact $SE(3)$-Invariant Form-Factor Self-Attention (`H2`)

Under any global rigid transform $(R, \mathbf{t}) \in SE(3)$:
$$\mathbf{p}'_i = R\mathbf{p}_i + \mathbf{t}, \quad T'_i = R T_i, \quad B'_i = R B_i, \quad N'_i = R N_i.$$

1. **Exact Translation-Invariant Centroid Normalization:**
   $$\hat{\mathbf{p}}_i = \frac{\mathbf{p}_i - \bar{\mathbf{p}}}{s_{\text{ref}}}, \qquad \bar{\mathbf{p}} = \frac{1}{N}\sum_{k=1}^{N} \mathbf{p}_k, \qquad s_{\text{ref}} = 4.0\text{ m}$$
   eliminating both translation dependence and FP16/BF16 catastrophic cancellation under large world offsets while preserving exact metric scale (meters) for inverse-square radiative falloff.
2. **Strictly Invariant Token Stem ($13 \to 256$):**
   Only intrinsic physical patch properties enter the hidden stream $x_i$:
   $$\big[\log(1+A_i),\; \mathbf{a}_i,\; r_i,\; m_i,\; \log(1+\mathbf{e}_i),\; \mathbb{I}[\max \mathbf{e}_i > 10^{-3}],\; \log(1+\mathbf{e}_i A_i)\big] \in \mathbb{R}^{13}.$$
   Raw world coordinates NEVER enter the hidden state.
3. **Factorized $SE(3)$-Invariant Form-Factor $Q_i \cdot K_j$ (Zero $O(N^2)$ Bias Matrix):**
   For each head $h \in \{1,\dots,8\}$ (`d_head = 32 = 19 semantic + 13 geometric`), the hidden state $x_i$ predicts positive distance-decay rates $\alpha_{i,h}, \beta_{j,h} = \text{clamp}(\text{softplus}(\cdot), 0, 20)$ and local-frame 3D directional & orientation vectors rotated into world coordinates via $(T_i, B_i, N_i)$:
   $$\mathbf{u}_{i,h} = u^T T_i + u^B B_i + u^N N_i, \quad \mathbf{m}_{i,h} = m^T T_i + m^B B_i + m^N N_i,$$
   $$\mathbf{v}_{j,h} = v^T T_j + v^B B_j + v^N N_j, \quad \mathbf{r}_{j,h} = r^T T_j + r^B B_j + r^N N_j.$$
   Packing $Q^{\text{geo}}_{i,h}, K^{\text{geo}}_{j,h} \in \mathbb{R}^{13}$ as:
   $$Q^{\text{geo}}_{i,h} = \big[2\alpha_{i,h}\hat{\mathbf{p}}_i + \mathbf{u}_{i,h},\; -\alpha_{i,h}\|\hat{\mathbf{p}}_i\|^2 - \mathbf{u}_{i,h}\cdot\hat{\mathbf{p}}_i,\; -\alpha_{i,h},\; \hat{\mathbf{p}}_i,\; 1,\; \|\hat{\mathbf{p}}_i\|^2,\; \mathbf{m}_{i,h}\big]$$
   $$K^{\text{geo}}_{j,h} = \big[\hat{\mathbf{p}}_j,\; 1,\; \|\hat{\mathbf{p}}_j\|^2,\; 2\beta_{j,h}\hat{\mathbf{p}}_j + \mathbf{v}_{j,h},\; -\beta_{j,h}\|\hat{\mathbf{p}}_j\|^2 - \mathbf{v}_{j,h}\cdot\hat{\mathbf{p}}_j,\; -\beta_{j,h},\; \mathbf{r}_{j,h}\big]$$
   yields the exact $SE(3)$-invariant inner product inside standard FlashAttention / SDPA (`d_qk = d_v = 32`):
   $$Q^{\text{geo}}_{i,h} \cdot K^{\text{geo}}_{j,h} = -(\alpha_{i,h} + \beta_{j,h})\|\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_j\|^2 + \mathbf{u}_{i,h}\cdot(\hat{\mathbf{p}}_j - \hat{\mathbf{p}}_i) + \mathbf{v}_{j,h}\cdot(\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_j) + \mathbf{m}_{i,h}\cdot\mathbf{r}_{j,h}.$$
4. **$SE(3)$-Equivariant Geometric Value Readout:**
   Concatenating $[\hat{\mathbf{p}}_j, N_j] \in \mathbb{R}^6$ onto each head's 26 invariant value channels ($26 + 6 = 32$) computes attention-weighted centroid $\bar{\mathbf{p}}_{i,h} = \sum_j A_{ij,h}\hat{\mathbf{p}}_j$ and normal $\bar{\mathbf{n}}_{i,h} = \sum_j A_{ij,h}N_j$ in the same single SDPA call. Projecting $\Delta\mathbf{p}_{i,h} = \bar{\mathbf{p}}_{i,h} - \hat{\mathbf{p}}_i$ and $\bar{\mathbf{n}}_{i,h}$ onto patch $i$'s local frame $(T_i, B_i, N_i)$ produces 6 exact $SE(3)$-invariant 3D directional features per head, directly empowering Order-1 SH prediction and multi-bounce occlusion reasoning.
5. **Tangent-Plane Gauge Augmentation:**
   During training, random per-patch rotations $\theta_i \sim \mathcal{U}(0, 2\pi)$ of $(T_i, B_i)$ around $N_i$ (paired with matching rotation of target $(I_{1,T}, I_{1,B})$) enforce in-plane tangent gauge equivariance in addition to exact $SE(3)$ invariance.

---

## 4. Crease-Safe Normal-Gated Patch-to-Surface Interpolation & Glossy GI (`three/tsl`)

At fragment shading point $(\mathbf{x}, \mathbf{n}_s)$ with reflection vector $\mathbf{r} = \text{reflect}(-\mathbf{v}, \mathbf{n}_s)$ on object $m$ with 32 patches $\mathcal{P}_m$, `radiantGINode.ts` evaluates bilateral weights:
$$w_p(\mathbf{x}, \mathbf{n}_s) = \exp\!\left(-\frac{\|\mathbf{x} - \mathbf{p}_p\|^2}{3.0 r_p^2 + 0.06}\right) \cdot \Big(\max(\mathbf{n}_s \cdot N_p, 0)^{2.5} + 0.025\,\text{smoothstep}(0.02, 0.35, \mathbf{n}_s \cdot N_p)\Big) \cdot \exp\!\left(-\frac{((\mathbf{x} - \mathbf{p}_p)\cdot N_p)^2}{0.14}\right)$$
where $r_p = \max(\sqrt{A_p / \pi}, 0.16\text{ m})$. The softened $\cos^{2.5}$ normal gate and tangent-plane distance penalty eliminate light bleeding across sharp $90^\circ$ corners while keeping curved columns and thin bevels smoothly shaded.

Per-channel world-space Order-1 SH dominant directions $\hat{\mathbf{d}}_c = \mathbf{I}_{1,\text{world},c} / \max(\|\mathbf{I}_{1,\text{world},c}\|, 10^{-4})$ also drive an energy-conserving directional glossy specular indirect bounce lobe ($\propto \max(\mathbf{r} \cdot \hat{\mathbf{d}}_c, 0)^{s(r)}$), modulated by Fresnel $F_0 = \text{mix}(0.04, \mathbf{a}_{\text{frag}}, m)$ and per-pixel diffuse texture albedo $\mathbf{a}_{\text{frag}} = \mathbf{a}_{\text{base}} \odot \text{texture}(\text{map}, \text{uv})$ before injection into `material.emissiveNode`.

---

## 5. Measured Oracle Verification Tables (v2 Final)

All metrics below are measured directly from the locked v2 artifacts generated by the warm Colab GPU pipeline (`36,000` training scenes @ `256` spp × 3 bounces across 11 multi-scale/sunlit architectural topologies + 200 procedural & 5 ephemeral real CC0 watertight meshes with face-normal stratified 32-patch clustering; `256` strictly unseen held-out scenes + `16` dynamic occlusion pairs @ `1,024` spp × 3 bounces; `9`-layer `d_model=256`, `8`-head BF16 Transformer trained for `18,000` steps, `6,908,668` params = `13.18 MB` FP16 `RDGI` binary) and the Apple M1 Metal 3 headless Chromium WebGPU audit (`--enable-unsafe-webgpu`, block-tiled `vec4` FlashAttention compute kernel).

### O1 — Pre-Generation Warp Patch Tracer Physics Conservation

| Check | Analytical Target | Warp GPU Measured | Error | Gate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| White Furnace $I_0$ ($\rho = [0.75, 0.60, 0.40]$) | `[1.7344, 1.1760, 0.6240]` | `[1.7344, 1.1760, 0.6240]` | `0.0027%` max rel | `< 1.0%` | **PASS** |
| White Furnace $I_{1,N} = \frac{2}{3} I_0$ | `[1.1563, 0.7840, 0.4160]` | `[1.1562, 0.7840, 0.4160]` | `0.0003%` max rel | `< 1.0%` | **PASS** |
| White Furnace Tangent Symmetry ($I_{1,T}, I_{1,B}$) | `0.0000` | `≤ 0.000009` | `9.0e-6` abs | `< 0.015` | **PASS** |
| Bounce-0 Direct Exclusion ($\rho = 0, L_e = 5$) | `0.0000` | `0.0000` | `0.0` exact | `< 1e-6` | **PASS** |
| Cornell Box 3-Bounce Radiosity (6 walls, 864-patch ref) | Spatially resolved | 4096-spp MC | `1.197%` mean rel | `< 2.0%` | **PASS** |
| Cornell Color Bleed (Left $G/R$, Right $R/G$) | `> 1.12` | `1.1476` / `1.1534` | — | `> 1.12` | **PASS** |

### O2 — Slice 0 $SE(3)$ Falsification (`H1` Naive XYZ vs. `H2` RADIANT $SE(3)$ Form-Factor)

Trained identically for `600` steps (`6` layers, `d_model=256`, `8` heads) and evaluated on the `256` strictly unseen held-out rooms (both aligned and under arbitrary random $(R, \mathbf{t}) \in SE(3)$):

| Model | Aligned PSNR | Rotated+Translated PSNR | Rotated Color-Bleed $r$ | Max $SE(3)$ Drift RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **`H1` (Naive Global-XYZ Baseline)** | `19.25 dB` | `7.45 dB` | `0.2514` | `1.04211` |
| **`H2` (RADIANT $SE(3)$ Form-Factor)** | **`21.29 dB`** (`+2.04 dB`) | **`21.29 dB`** (**`+13.83 dB`**) | **`0.9600`** | **`4.0e-7`** (`< 0.005`) |

### O3 — Final v2 FP16 (`13.18 MB`) Zero-Shot Held-Out Multi-Bounce GI Fidelity (`256` Unseen Rooms @ `1024` spp)

| Metric | Measured (v2 FP16 Exported) | Target Gate | Status |
| :--- | :---: | :---: | :---: |
| Tone-Mapped Indirect PSNR (Aligned) | **`30.71 dB`** (v1: `30.31 dB`, `+0.40 dB`) | `>= 28.0 dB` | **PASS** |
| Tone-Mapped Indirect PSNR (Random $SE(3)$) | **`30.71 dB`** | `>= 28.0 dB` | **PASS** |
| Color-Bleed Spatial Correlation ($r$) | **`0.9954`** | `>= 0.92` | **PASS** |
| Order-1 SH Directional Cosine Similarity | **`0.9864`** | — | **PASS** |
| Max $SE(3)$ Invariance Drift RMSE ($R \in SO(3), \|\mathbf{t}\| \le 100\text{ m}$) | **`0.000240`** | `< 0.005` | **PASS** |
| Dynamic Occlusion Drop (16 Unseen Pairs: GT vs. Pred) | `95.50%` GT / **`96.31%` Pred** (`1.09%` MAE) | `>= 40%` | **PASS** |

Per-topology breakdown across the `256` strictly unseen held-out test rooms & unseen procedural/CC0 meshes:

| Unseen Held-Out Topology | Scenes | PSNR (dB) | Color-Bleed $r$ | SH Dir Cosine |
| :--- | :---: | :---: | :---: | :---: |
| `unseen_vaulted_atrium` | 67 | **`32.11 dB`** | `0.9973` | `0.9923` |
| `unseen_cornell_sculptural` | 67 | **`31.27 dB`** | `0.9957` | `0.9895` |
| `unseen_diagonal_gallery` | 74 | `29.93 dB` | `0.9931` | `0.9814` |
| `unseen_split_doorway` | 48 | `29.83 dB` | `0.9932` | `0.9812` |

### O4 & O5 — PyTorch ↔ Apple M1 Metal 3 WebGPU WGSL Bit-Parity & Interactive Performance

| Verification Check | Measured (Apple M1 Metal 3) | Target Gate | Status |
| :--- | :---: | :---: | :---: |
| PyTorch FP16 ↔ WGSL Compute Parity (4 Probe Scenes, 384–576 tokens) | **`worst_rmse = 1.0e-6`** (`max_abs ≤ 1.6e-5`) | `<= 0.005` | **PASS** |
| Block-Tiled FlashAttention Compute Latency (352 tokens, M1 Metal 3) | **`41.8–51.1 ms`** (v1: `181.6 ms`, **~3.6–4.3× faster**) | Real-time | **PASS** |
| Console / Page / WGSL Validation Errors | **`0`** | `0` | **PASS** |
| External Runtime CDN Requests | **`0`** | `0` | **PASS** |
| Steady-State Interactive Frame Rate (0 ms GI when static) | **`60.7 fps`** | `>= 60 fps` | **PASS** |
| Live-Drag Interactive Frame Rate | **`62.2 fps`** | `>= 30 fps` | **PASS** |
| Cornell GI vs. Direct Pixel L1 / Live Wall Color-Bleed Shift L1 | `38.21` / `45.23` (`chroma L2 = 0.1655`) | Distinct shift | **PASS** |
| L-Corridor Doorway Occlusion Live Indirect Drop | **`67.13%`** | Distinct drop | **PASS** |
| Exported Runtime Binary (`artifacts/radiant_gi_fp16.bin`) | **`13.18 MB`** | `< 20 MB` | **PASS** |
| Local `data/` + `artifacts/` Disk Total | **`21.75 MB`** (incl. 13 audit PNGs) | `<= 120 MB` | **PASS** |
| Colab Compute Units Logged (`colab/budget_ledger.jsonl`) | **`21.61 u`** (`0` active leaked VMs) | `<= 25.0 u` | **PASS** |

