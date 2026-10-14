# -*- coding: utf-8 -*-
"""
Gated DeltaNet —— 线性（递归）注意力，**支持 Scopus maintain delta-rule 精确改写记忆**。

来源
----
* DeltaNet     《Parallelizing Linear Transformers with the Delta Rule over Sequence
  Length》(Schlag et al., 2021 / 2023)。把线性注意力的状态更新从"累加"改成"delta 规则"。
* Gated DeltaNet 《Gated Delta Networks: Improving Mamba2 with the Delta Rule》(2024-12),
  作者 Yang Song Songlin Yang 等。加入 Mamba 式衰减 α，并提出 chunkwise 并行算法。
* Kimi K3 / Qwen3.5：2026 年把 GDN（Kimi 称 KDA）作为 75%~87.5% 的层，实现
  「400B 质量 / 17B 激活速度」。

为什么它比 Mamba2 强
--------------------
三类线性注意力的区别在于同一个递推关系：

    S_t = decay_t · S_{t-1} + write_t

    Mamba2  : 只忘不忘改，decay 是标量 Auch.write 是累加 ⊗ v kᵀ
    DeltaNet: 能改不能忘，S(I - β k kᵀ) 是 Householder 反射 —— 精确擦除 k 旧关联并写入 v
    **GDN** : 既能忘又能改 —— S_t = α_t S_{t-1}(I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ

在线学习视角下，这等价于在回归损失 L(S) = ½‖S k_t - v_t‖² 上做 SGD，且带自适应
权重衰减 α_t。这正是它能在 S-NIAH-2 检索任务上拿到 92 分（Mamba2 只有 56）的原因：
不同的 key-value 关联不会互相污染。

数学
----
约定状态 S ∈ R^{d_v × d_k}，输出 o_t = S_t q_t。递推式：

    S_t = α_t · S_{t-1} (I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ          (R)

当 ‖k‖₂ = 1 时，(I - β k kᵀ) 是严格 Householder 反射，所有状态转移矩阵特征值
模 ≤ 1 ⟹ 动力系统不发散。因此默认对 q,k 做 L2 归一化。

Chunkwise 并行（本文件的核心）
------------------------------
(R) 是逐 token 串行的，训练需要并行。做法：切成 chunk，块**间**串行的深度
从 N 降到 N/C，块**内**用稠密 matmul 一次算完。

第 1 步：把衰减变换掉。
定义 chunk 内相对累积衰减（相对于 chunk 第一个位置）

    г_i = Π_{j=0}^{i} α_{rC+j}

令 P_t = S_t / г_t，代入 (R)：

    P_t = P_{t-1}(I - β_t k_t k_tᵀ) + β_t (v_t / г_t) k_tᵀ ,   P_{-1} = S_{rC-1}

即：把 value 除以累积衰减后，就退化成了**无门控的 delta rule**。

第 2 步：解"什么被写入"。
改写 (R') 为 P_t = P_{t-1} + ỹ_t k_tᵀ，其中 ỹ_t = β_t(ṽ_t - P_{t-1}k_t)。
展开所有交叉项得到一个**下三角线性系统**：

    ũ_t + Σ_{s<t} β_s (k_s·k_t) ũ_s = v_t - S_{rC-1} k_t ,   ũ_t ≜ ỹ_t/β_t

    ⟹ (I + strictLower(diag(β)·K Kᵀ)) Ũ = V - S_{rC-1} Kᵀ
    ⟹ Ỹ = diag(β) Ũ = T·diag(β)·V - (T·(β⊙K))·S_{rC-1}ᵀ
    其中  T = (I + strictLower(diag(β)·K Kᵀ))⁻¹       (C×C 下三角，对角为 1，可逆)

这就是 WY 表示（Williams–Yadle der出色的矩阵连乘截断技巧）在此处的具体形态：一串
Householder 型连乘被压缩成"解一个 C×C 三角系统 + 两次 matmul"。

第 3 步：输出与状态。

    Q̃ = diag(г) Q
    O  = tril(Q̃ Kᵀ)·Ỹ  +  Q̃·S_{rC-1}ᵀ
    S_{rC+C-1} = г_{C-1}·( S_{rC-1} + Ỹᵀ K )

其中状态约定 S ∈ R^{Dv×Dk}（输出 o = S q）。

推导细节见 docs/MATH.md §3。

Invariants（本文件的硬约束）
----------------------------
1. chunk_gated_delta_rule 与逐 token 递推 recurrent_gated_delta_rule 数值等价（~1e-5 fp32）。
2. α≡1 时退化为标准 DeltaNet：chid crwd recurrent chunk 三者一致。
3. β≡0 时状态保持 S(v)，输出恒 0（无写入）→ 用于 sanity check。
4. 梯度可回传：chunkwise 版本参与 autograd。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.modules.norms import RMSNorm, GatedRMSNorm, l2_normalize


__all__ = [
    "GatedDeltaNet", "DeltaNetCache",
    "recurrent_gated_delta_rule", "chunk_gated_delta_rule",
    "gated_delta_rule", "causal_short_conv", "l2norm_fwd",
]


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """沿最后一维 L2 归一化。见 aurora.modules.norms.l2_normalize。"""
    return l2_normalize(x, dim=-1, eps=eps)


# ===========================================================================
# 1) 逐 token 递推 —— 黄金参照（gold reference）
# ===========================================================================
def recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """逐 token 递推。慢，但它是定义本身 —— 所有并行实现都必须与之对齐。

    Parameters
    ----------
    q : (B, H, N, Dk)
    k : (B, H, N, Dk)
    v : (B, H, N, Dv)
    beta  : (B, H, N)    写入强度 β ∈ (0,1)
    alpha : (B, H, N)    衰减 α ∈ (0,1)
    initial_state : (B, H, Dv, Dk)，可选

    Returns
    -------
    out : (B, H, N, Dv)
    final_state : (B, H, Dv, Dk) if output_final_state
    """
    B, H, N, Dk = k.shape
    Dv = v.shape[-1]
    dtype = torch.float32
    q32, k32, v32 = q.float(), k.float(), v.float()
    beta32 = beta.float().unsqueeze(-1)          # (B,H,N,1)
    alpha32 = alpha.float().unsqueeze(-1)

    if initial_state is None:
        S = torch.zeros(B, H, Dv, Dk, dtype=dtype, device=k.device)
    else:
        S = initial_state.float().clone()

    outs = []
    for t in range(N):
        kt = k32[:, :, t, :]                      # (B,H,Dk)
        vt = v32[:, :, t, :]                      # (B,H,Dv)
        qt = q32[:, :, t, :]
        bt = beta32[:, :, t, :]                   # (B,H,1)
        at = alpha32[:, :, t, :]

        # S k_t : (B,H,Dv)
        Skt = torch.einsum("bhid,bhd->bhi", S, kt)
        # (S k_t) ⊗ k_t : (B,H,Dv,Dk)
        remove = torch.einsum("bhi,bhd->bhid", Skt, kt)
        S = at.unsqueeze(-1) * (S - bt.unsqueeze(-1) * remove)
        S = S + bt.unsqueeze(-1) * torch.einsum("bhi,bhd->bhid", vt, kt)
        o_t = torch.einsum("bhid,bhd->bhi", S, qt)
        outs.append(o_t)

    out = torch.stack(outs, dim=2)                # (B,H,N,Dv)
    return out.to(q.dtype), (S.to(q.dtype) if output_final_state else None)


# ===========================================================================
# 2) 分块并行 —— 训练主力
# ===========================================================================
def _build_triangular_matrix(kbeta: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """构造 M = I + strictLower(diag(β)·K Kᵀ)，shape (B,H,C,C)。

    M 是下三角且对角为 1 ⟹ 可逆，且 torch.linalg.solve_triangular 稳定。
    """
    C = k.shape[2]
    G = kbeta @ k.transpose(-1, -2)                     # (B,H,C,C) = diag(β)·K Kᵀ
    strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=k.device), diagonal=-1)
    G = G.masked_fill(~strict, 0.0)
    eye = torch.eye(C, dtype=G.dtype, device=G.device)
    return G + eye


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunkwise 并行的 Gated DeltaNet。训练的 O(N) 主力实现。

    算法（每个 chunk，位置相对 chunk 内 [0, C)）:

        г_i = Π_{j≤i} α_{rC+j}                          相对累积衰减
        Ṽ   = V / г                                     衰减变换后的 value
        M   = I + strictLower(diag(β)·K Kᵀ)             下三角，对角 1
        T   = M⁻¹·diag(β)
        W   = T·K                                       (C,Dk)
        U   = T·Ṽ                                        (C,Dv)
        Ỹ   = U - W·S_prevᵀ                              「被真正写入的值」
        Q̃   = diag(г)·Q
        O   = tril(Q̃ Kᵀ)·Ỹ + Q̃·S_prevᵀ
        S_next = г_{C-1}·(S_prev + Kᵀ·Ỹ)

    Notes
    -----
    * 数值：г 可能下溢（α 很小且 chunk 长）。这里对 г 施加 clamp_min(1e-6)，
      并在 `\u007f GDN` 模块侧约束 α ∈ [α_min, 1]。这是工程折衷，不是数学缺陷。
    * 复杂度：时间 O(N·C·(Dk+Dv) + (N/C)·Dv·Dk)，内存 O(N·C + H·Dv·Dk)。

    Returns
    -------
    out : (B, H, N, Dv)
    final_state : (B, H, Dv, Dk) if output_final_state
    """
    B, H, N, Dk = k.shape
    Dv = v.shape[-1]
    dev = k.device
    dtype32 = torch.float32

    q32, k32, v32 = q.float(), k.float(), v.float()
    beta32, alpha32 = beta.float(), alpha.float()

    if initial_state is None:
        S = torch.zeros(B, H, Dv, Dk, dtype=dtype32, device=dev)
    else:
        S = initial_state.float().clone()

    C = int(chunk_size)
    pad_len = (C - (N % C)) % C
    if pad_len:
        q32 = F.pad(q32, (0, 0, 0, pad_len))
        k32 = F.pad(k32, (0, 0, 0, pad_len))
        v32 = F.pad(v32, (0, 0, 0, pad_len))
        beta32 = F.pad(beta32, (0, pad_len))
        alpha32 = F.pad(alpha32, (0, pad_len), value=1.0)
    Np = N + pad_len
    n_chunk = Np // C

    # ---- 分块 ----
    q_c = q32.reshape(B, H, n_chunk, C, Dk)
    k_c = k32.reshape(B, H, n_chunk, C, Dk)
    v_c = v32.reshape(B, H, n_chunk, C, Dv)
    beta_c = beta32.reshape(B, H, n_chunk, C)
    alpha_c = alpha32.reshape(B, H, n_chunk, C)

    # ---- 相对累积衰减 г（chunk 内 cumprod）----
    log_alpha = alpha_c.clamp_min(1e-12).log()
    gamma = torch.exp(torch.cumsum(log_alpha, dim=-1)).clamp_min(1e-6)   # (B,H,n,C)
    gamma_inv = (1.0 / gamma)

    outs = []
    eye_c = torch.eye(C, dtype=dtype32, device=dev)

    for r in range(n_chunk):
        Qb = q_c[:, :, r]        # (B,H,C,Dk)
        Kb = k_c[:, :, r]
        Vb = v_c[:, :, r]
        bb = beta_c[:, :, r]     # (B,H,C)
        gg = gamma[:, :, r]      # (B,H,C)

        Vt = Vb * gamma_inv[:, :, r].unsqueeze(-1)          # Ṽ = V / г
        kbeta = Kb * bb.unsqueeze(-1)                        # β⊙K

        M = _build_triangular_matrix(kbeta, Kb)              # (B,H,C,C)
        # T = (I + diag(β)·strictLower(K Kᵀ))⁻¹  —— 注意 RHS 是单位阵，
        # β 的补偿在 U/W 处通过 (β⊙·) 体现，不能在这里重复乘一次。
        T = torch.linalg.solve_triangular(M, eye_c.expand_as(M), upper=False, left=True)
        W = T @ kbeta                                        # (B,H,C,Dk)  = T(β⊙K)
        U = T @ (Vt * bb.unsqueeze(-1))                      # (B,H,C,Dv)  = T(β⊙Ṽ)

        # Ỹ = U - W·S_prevᵀ
        St = S.transpose(-1, -2)                             # (B,H,Dk,Dv)
        Y = U - (W @ St)                                     # (B,H,C,Dv)

        Qt = Qb * gg.unsqueeze(-1)                           # Q̃ = г⊙Q
        intra = torch.tril(Qt @ Kb.transpose(-1, -2))        # (B,H,C,C)
        o = intra @ Y + (Qt @ St)                            # (B,H,C,Dv)
        outs.append(o)

        # S_next = г_{C-1}·( S_prev + Σ_s ỹ_s k_sᵀ ) = г_{C-1}·( S_prev + Ỹᵀ K )
        # 注意状态约定是 S ∈ R^{Dv×Dk}，因此这里是 Ỹᵀ K 而不是 Kᵀ Ỹ。
        g_last = gg[:, :, -1].unsqueeze(-1).unsqueeze(-1)    # (B,H,1,1)
        S = g_last * (S + Y.transpose(-1, -2) @ Kb)          # (B,H,Dv,Dk)

    out = torch.cat(outs, dim=2).reshape(B, H, Np, Dv)[:, :, :N]
    return out.to(q.dtype), (S.to(q.dtype) if output_final_state else None)


# ---------------------------------------------------------------------------
# 统一入口：短序列(N≤64)或推理未打开 chunk 时用递推，否则用分块
# ---------------------------------------------------------------------------
def gated_delta_rule(
    q, k, v, beta, alpha,
    initial_state=None, output_final_state=False, chunk_size: int = 64,
    mode: str = "auto",
):
    """自动选择 chunkwise / recurrent。"""
    N = k.shape[2]
    if mode == "recurrent" or (mode == "auto" and N <= 64):
        return recurrent_gated_delta_rule(
            q, k, v, beta, alpha, initial_state, output_final_state)
    return chunk_gated_delta_rule(
        q, k, v, beta, alpha, initial_state, output_final_state, chunk_size)


# ===========================================================================
# 3) 因果短卷积（depthwise）
# ===========================================================================
def causal_short_conv(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
    conv_state: Optional[torch.Tensor] = None, activation: Optional[str] = "silu",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """因果 depthwise 1D 卷积。

    GDN 论文强烈建议保留它：短卷积提供局部上下文，禁用会明显掉质量。

    x       : (B, N, C)
    weight  : (C, K)
    Returns : (y (B,N,C), new_conv_state (B, K-1, C))
    """
    B, N, C = x.shape
    K = weight.shape[-1]
    if conv_state is not None:
        x_cat = torch.cat([conv_state, x], dim=1)
    else:
        x_cat = F.pad(x.transpose(1, 2), (K - 1, 0)).transpose(1, 2)
    n_pad = x_cat.shape[1] - N

    xb = x_cat.transpose(1, 2)                                  # (B,C,N+K-1)
    y = F.conv1d(xb, weight.unsqueeze(1), bias=bias, groups=C, padding=0)
    y = y.transpose(1, 2)

    if activation == "silu":
        y = F.silu(y)
    elif activation is not None:
        raise ValueError(f"不支持的 activation: {activation}")

    new_state = x_cat[:, -(K - 1):, :] if K > 1 else None
    return y[:, -N:], new_state


# ===========================================================================
# 4) 模块
# ===========================================================================
class DeltaNetCache:
    """GDN 推理状态：一个 (Dv, Dk) 矩阵 + 短卷积尾部。

    相比 attention 的 KV cache 随长度线性增长，这里是**常数大小**
    O(H·Dv·Dk) —— 这就是线性注意力在长上下文上的全部优势所在。
    """

    __slots__ = ("state", "conv_state_q", "conv_state_k", "conv_state_v")

    def __init__(self, state=None, conv_state_q=None, conv_state_k=None, conv_state_v=None):
        self.state = state
        self.conv_state_q = conv_state_q
        self.conv_state_k = conv_state_k
        self.conv_state_v = conv_state_v

    @classmethod
    def empty(cls, batch_size: int, num_heads: int, key_dim: int, value_dim: int,
              conv_size: int, dtype: torch.dtype, device: torch.device) -> "DeltaNetCache":
        return cls(
            state=torch.zeros(batch_size, num_heads, value_dim, key_dim,
                              dtype=dtype, device=device),
            conv_state_q=torch.zeros(batch_size, max(0, conv_size - 1), num_heads * key_dim,
                                     dtype=dtype, device=device),
            conv_state_k=torch.zeros(batch_size, max(0, conv_size - 1), num_heads * key_dim,
                                     dtype=dtype, device=device),
            conv_state_v=torch.zeros(batch_size, max(0, conv_size - 1), num_heads * value_dim,
                                     dtype=dtype, device=device),
        )


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer。

    结构（对齐 fla 的 GatedDeltaNetConfig 语义）:

        h ── q_proj ─┐
           ── k_proj ─┤  → short_conv → SiLU → L2Norm → delta rule → gated RMSNorm → o_proj
           ── v_proj ─┤
           ── b_proj ─┘  (β, sigmoid)
           ── A_log/g_proj → α (衰减) 与 输出 gate

    参数量 ≈ 6·d_h²/层，与 Mamba2 同量级。

    Parameters
    ----------
    hidden_size, num_heads, key_dim, value_dim : 主干维度
    conv_size : int   短卷积核宽，务必 ≥ 2
    use_beta, use_gate : 是否开启 β 写入门与输出门
    allow_neg_eigval : 允许负特征值（见《Unlocking State-Tracking in Linear RNNs
        Through Negative Eigenvalues》arXiv:2411.12537），可提升状态追踪能力
    chunk_size : 训练时 chunkwise 块大小；推理（N=1）自动走递推分支
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 12,
        key_dim: int = 128,
        value_dim: int = 128,
        conv_size: int = 4,
        use_beta: bool = True,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        layer_idx: int = 0,
        chunk_size: int = 64,
        norm_eps: float = 1e-6,
        use_l2norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.conv_size = int(conv_size)
        self.use_beta = bool(use_beta)
        self.use_gate = bool(use_gate)
        self.allow_neg_eigval = bool(allow_neg_eigval)
        self.chunk_size = int(chunk_size)
        self.use_l2norm = bool(use_l2norm)
        self.layer_idx = int(layer_idx)

        H, Dk, Dv, Cs = self.num_heads, self.key_dim, self.value_dim, self.conv_size

        self.q_proj = nn.Linear(self.hidden_size, H * Dk, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, H * Dk, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, H * Dv, bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(self.hidden_size, H, bias=False)
        else:
            self.register_parameter("b_proj", None)

        if self.use_gate:
            self.g_proj = nn.Linear(self.hidden_size, H * Dv, bias=False)
        else:
            self.register_parameter("g_proj", None)

        # 衰减 α ∈ (0,1)： learned parameterization (A_log → softplus → exp(-))
        self.A_log = nn.Parameter(torch.empty(H))

        # 短卷积：depthwise，per channel
        self.conv_size = max(2, Cs)
        conv_w = torch.empty(H * Dk, self.conv_size)
        conv_k = torch.empty(H * Dk, self.conv_size)
        conv_v = torch.empty(H * Dv, self.conv_size)
        conv_bq = torch.zeros(H * Dk)
        conv_bk = torch.zeros(H * Dk)
        conv_bv = torch.zeros(H * Dv)
        self.conv_weight_q = nn.Parameter(conv_w)
        self.conv_weight_k = nn.Parameter(conv_k)
        self.conv_weight_v = nn.Parameter(conv_v)
        self.conv_bias_q = nn.Parameter(conv_bq)
        self.conv_bias_k = nn.Parameter(conv_bk)
        self.conv_bias_v = nn.Parameter(conv_bv)

        if self.use_gate:
            self.o_norm = GatedRMSNorm(H * Dv, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(H * Dv, eps=norm_eps)
        self.o_proj = nn.Linear(H * Dv, self.hidden_size, bias=False)

        self.reset_parameters()

    # ------------------------------------------------------------------
    def reset_parameters(self):
        """短卷积按 depthwise 标准初始化（fan-in = 1）。"""
        for name in ("conv_weight_q", "conv_weight_k", "conv_weight_v"):
            p = getattr(self, name)
            C, K = p.shape
            nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(max(1, K)))
        nn.init.zeros_(self.conv_bias_q)
        nn.init.zeros_(self.conv_bias_k)
        nn.init.zeros_(self.conv_bias_v)
        # A_log 初始化在 [-6, -2] 附近，使初始 α≈1（几乎不遗忘，先学会连带 ∨ 再学遗忘）
        nn.init.uniform_(self.A_log, -6.0, -2.0)

    # ------------------------------------------------------------------
    def _decay(self, dtype: torch.dtype) -> torch.Tensor:
        """α_h = exp(-softplus(A_log_h))，shape (H,)。

        Mamba 式参数化：保证 α ∈ (0,1) 且初始接近 1。
        """
        a = -torch.nn.functional.softplus(self.A_log.float())
        return a.exp().to(dtype)

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[DeltaNetCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[DeltaNetCache]]:
        """hidden_states: (B, T, d_h) → (B, T, d_h)。"""
        B, T, _ = hidden_states.shape
        H, Dk, Dv = self.num_heads, self.key_dim, self.value_dim
        orig_dtype = hidden_states.dtype

        conv_s_q = cache.conv_state_q if cache is not None else None
        conv_s_k = cache.conv_state_k if cache is not None else None
        conv_s_v = cache.conv_state_v if cache is not None else None

        q, conv_s_q = causal_short_conv(self.q_proj(hidden_states),
                                        self.conv_weight_q, self.conv_bias_q, conv_s_q, "silu")
        k, conv_s_k = causal_short_conv(self.k_proj(hidden_states),
                                        self.conv_weight_k, self.conv_bias_k, conv_s_k, "silu")
        v, conv_s_v = causal_short_conv(self.v_proj(hidden_states),
                                        self.conv_weight_v, self.conv_bias_v, conv_s_v, "silu")

        q = q.view(B, T, H, Dk)
        k = k.view(B, T, H, Dk)
        v = v.view(B, T, H, Dv)

        if self.use_l2norm:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
        else:
            q = q * (Dk ** -0.5)

        if self.use_beta:
            beta = torch.sigmoid(self.b_proj(hidden_states))          # (B,T,H)
        else:
            beta = torch.ones(B, T, H, device=hidden_states.device, dtype=hidden_states.dtype)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        alpha = self._decay(hidden_states.dtype)                      # (H,)
        alpha = alpha.view(1, 1, H).expand(B, T, H)

        initial_state = cache.state if cache is not None else None

        # 训练必须走 chunkwise 以获得并行；推理（T=1）走递推，每步 O(1) 且状态常数大小。
        use_recurrent = (not self.training) or T <= 64
        # 推理时必须返回 final_state 并构建 cache，使增量 decode 能累积历史
        # （含首个 cache=None 的步骤）；训练期不构建，避免无谓保留计算图。
        want_cache = not self.training
        if use_recurrent:
            core_out, final_state = recurrent_gated_delta_rule(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                beta.transpose(1, 2), alpha.transpose(1, 2),
                initial_state=initial_state, output_final_state=want_cache)
        else:
            core_out, final_state = chunk_gated_delta_rule(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                beta.transpose(1, 2), alpha.transpose(1, 2),
                initial_state=initial_state, output_final_state=want_cache,
                chunk_size=self.chunk_size)

        core_out = core_out.transpose(1, 2).reshape(B, T, H * Dv).to(orig_dtype)

        if self.use_gate:
            gate = self.g_proj(hidden_states)
            out = self.o_norm(core_out, gate)
        else:
            out = self.o_norm(core_out)

        new_cache = DeltaNetCache(state=final_state, conv_state_q=conv_s_q,
                                  conv_state_k=conv_s_k, conv_state_v=conv_s_v) \
            if want_cache else None
        return self.o_proj(out), new_cache

    def extra_repr(self) -> str:
        return (f"heads={self.num_heads}, dk={self.key_dim}, dv={self.value_dim}, "
                f"conv={self.conv_size}, chunk={self.chunk_size}")
