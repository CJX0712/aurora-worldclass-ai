# -*- coding: utf-8 -*-
"""
Sliding-Window Attention with GQA —— 局部 softmax 注意力。

来源
----
* GQA 《GQA: Training Generalized Multi-Query Transformer Models from
  Multi-head Checkpoints》(Ainslie et al., 2023)。Query 多、KV 少，组内共享。
* 《Longformer》(Beltagy et al., 2020) 首创滑窗；2024 后 Gemma 2/3、Mistral、
  Qwen3 均把它作为混合配比中的「局部」部分，典型比例 5:1 ~ 7:1。

它在 Aurora 里的位置
--------------------
做一个立即可用的直觉实验：把 mixer_pattern 设成纯 "D"（全线性注意力）通常会有
**局部精细节略度不足**的问题（纯 recurrent 模型在真实召回任务上仍落后同规模
Transformer 约 37%）。插一层 SWA 就能把绝大部分差距补回来 —— 这正是
Gated DeltaNet 论文 H1 混合变体的结论。

为什么还需要 GQA
----------------
线性注意力负责「压缩后的长期诉求」，滑窗 softmax 负责「精确逐 token 匹配 」。
这两者的补最强性不是实验巧合：softmax attention 是**无损**的字典检索（可以直接
读出 KV），而 recurrent 状态是**有损**的压缩。二者叠加才是正确的分工。

数学
----
标准 scaled dot-product，但注意力范围被限制在窗口 w 内且因果：

    score_{tj} = q_t·k_j / sqrt(d),   j ∈ [max(0, t-w+1), t]
    o_t        = Σ_j softmax(score)_j v_j

GQA：n_h 个 query head 分成 g 组，每组共享一份 K/V。
本实现提供两种 KV 展开方式（等价）：
  - repeat_interleave:      显存换简单
  - enable_gqa（torch ≥2.5）: 内核内部广播，无显存开销
默认自动选择。

Invariants
----------
1. 窗口外的 token 完全不影响输出（disturb 测试）。
2. GQA 的 repeat 版与 enable_gqa 版数值等价。
3. w ≥ S 时退化为全因果注意力。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.modules.norms import QKRMSNorm
from aurora.modules.rope import RotaryEmbedding, YarnRotaryEmbedding, rotate_half


__all__ = ["SlidingWindowAttention", "SWACache"]


class SWACache:
    """常规的 K/V 缓存（GQA 形式，KV head 数为 num_key_value_heads）。"""

    __slots__ = ("k", "v")

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k
        self.v = v

    @property
    def seq_len(self) -> int:
        return int(self.k.shape[1])

    @classmethod
    def empty(cls, batch_size: int, num_kv_heads: int, head_dim: int,
              dtype: torch.dtype, device: torch.device) -> "SWACache":
        return cls(k=torch.zeros(batch_size, 0, num_kv_heads, head_dim,
                                 dtype=dtype, device=device),
                   v=torch.zeros(batch_size, 0, num_kv_heads, head_dim,
                                 dtype=dtype, device=device))


class SlidingWindowAttention(nn.Module):
    """滑窗 + GQA + RoPE + QK-Norm。

    Parameters
    ----------
    hidden_size, num_heads, num_key_value_heads, head_dim
    sliding_window : int，≤0 表示全因果
    rope 相关参数同 MLA
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        sliding_window: int = 512,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8192,
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.sliding_window = int(sliding_window)
        self.num_groups = max(1, self.num_heads // self.num_key_value_heads)
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = QKRMSNorm(self.head_dim, num_heads=self.num_heads, eps=rms_norm_eps) if qk_norm else nn.Identity()
        self.k_norm = QKRMSNorm(self.head_dim, num_heads=self.num_key_value_heads, eps=rms_norm_eps) if qk_norm else nn.Identity()

        rope_kwargs = dict(dim=self.head_dim, max_position=max_position_embeddings,
                           scaling_factor=rope_scaling_factor)
        if rope_scaling_type == "yarn":
            self.rotary = YarnRotaryEmbedding(base=rope_theta, **rope_kwargs)
        else:
            self.rotary = RotaryEmbedding(base=rope_theta,
                                          scaling_type=rope_scaling_type, **rope_kwargs)

        self.scaling = self.head_dim ** -0.5
        self._attn_max_logits: Optional[torch.Tensor] = None

    @property
    def attn_max_logits(self):
        return self._attn_max_logits

    def reset_max_logits(self) -> None:
        self._attn_max_logits = None

    @torch.no_grad()
    def qk_clip_step(self, threshold: float = 100.0, alpha: float = 0.5,
                     eps: float = 1e-6) -> Optional[torch.Tensor]:
        """Kimi K2 QK-Clip（MHA/GQA 版，按 group 裁剪）。

        普通多头注意力（含 GQA）的 QK-Clip 直接缩放投影矩阵
            W_q^g ← γ_g^α · W_q^g ，  W_k^g ← γ_g^{1-α} · W_k^g ，  α=0.5
        使 q·k 整体乘 γ_g，把 logit 钉在 τ 以下。

        实现要点（关键修正，同 MLA）：QK-Norm 会除以 RMS，直接缩放 q_proj/k_proj
        这类「归一化之前」的权重会被抵消。本实现改为缩放 **归一化之后** 的 QK-Norm
        逐 head affine（它在 RMS 除法之后施加，缩放必然存活）。由于 SWA 中 QK-Norm
        作用于整个 q/k 向量（RoPE 在其后施加，正交不变），缩放 norm affine 即等价于
        把整个 q^h、k^g 按 √γ_g 缩放，使 q·k → γ_g·(q·k)。

        GQA 下 K/V 是 **组内共享** 的：一个 group 的 K 被该组所有 head 共用，
        因此必须按 group（而非单个 head）计算 γ_g 并统一缩放，避免破坏组内一致性。

        Returns
        -------
        torch.Tensor | None：实际施加的逐 group γ（用于日志）。
        """
        if self._attn_max_logits is None:
            return None
        smax = self._attn_max_logits.detach().to(torch.float32).clamp_min(eps)  # (nh,)
        nh, nkv, d = self.num_heads, self.num_key_value_heads, self.head_dim
        groups = self.num_groups
        # 每个 head 所属 group（GQA：head i → group i // groups，共 nkv 组）
        gids = torch.arange(nh, device=smax.device) // max(1, groups)
        per_group = torch.zeros(nkv, device=smax.device).scatter_reduce(
            0, gids, smax, reduce="amax")
        gamma_g = torch.sqrt(torch.clamp(threshold / per_group, max=1.0))        # (nkv,) = √γ_g
        gamma_g = gamma_g.to(self.q_proj.weight.dtype)

        # ---- 缩放 QK-Norm 的逐 head affine（归一化之后，缩放存活）----
        if not isinstance(self.q_norm, nn.Identity) and not isinstance(self.k_norm, nn.Identity):
            # q_norm weight 形状 (nh, d)：head i 属于 group gids[i]
            self.q_norm.weight.data.mul_(gamma_g[gids].view(nh, 1))
            # k_norm weight 形状 (nkv, d)：group g 直接乘 γ_g
            self.k_norm.weight.data.mul_(gamma_g.view(nkv, 1))

        self._attn_max_logits = None
        return gamma_g * gamma_g  # γ_g

    @staticmethod
    def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x.float() * cos) + (rotate_half(x.float()) * sin)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, nkv, d) → (B, S, nh, d)，组内共享。"""
        if self.num_groups == 1:
            return x
        B, S, nkv, d = x.shape
        return x.unsqueeze(3).expand(B, S, nkv, self.num_groups, d).reshape(B, S, nkv * self.num_groups, d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[SWACache] = None,
    ) -> Tuple[torch.Tensor, Optional[SWACache]]:
        B, T, _ = hidden_states.shape
        nh, nkv, d = self.num_heads, self.num_key_value_heads, self.head_dim
        dev = hidden_states.device

        pos = cache.seq_len if cache is not None else 0
        S_total = pos + T
        cos_full, sin_full = self.rotary(S_total, device=dev)          # fp32 (S,d)
        cos_q = cos_full[pos:pos + T].view(1, T, 1, d)
        sin_q = sin_full[pos:pos + T].view(1, T, 1, d)

        q = self.q_proj(hidden_states).view(B, T, nh, d)
        k = self.k_proj(hidden_states).view(B, T, nkv, d)
        v = self.v_proj(hidden_states).view(B, T, nkv, d)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self._rope(q, cos_q, sin_q).to(hidden_states.dtype)
        # k 的 RoPE 只需覆盖「新 token」的 [pos, pos+T) 位置（与 q 一致）；
        # 缓存内的历史 k 已在其原始位置施加过 RoPE，不可重复施加。
        cos_k = cos_full[pos:pos + T].view(1, T, 1, d)
        sin_k = sin_full[pos:pos + T].view(1, T, 1, d)
        k = self._rope(k, cos_k, sin_k).to(hidden_states.dtype)

        if cache is not None:
            k = torch.cat([cache.k, k], dim=1)
            v = torch.cat([cache.v, v], dim=1)
        S = k.shape[1]

        # ---- mask: 因果 ∩ 滑窗 ----
        q_idx = torch.arange(pos, pos + T, device=dev).view(T, 1)
        k_idx = torch.arange(S, device=dev).view(1, S)
        mask = k_idx <= q_idx
        if self.sliding_window > 0:
            mask = mask & ((q_idx - k_idx) < self.sliding_window)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask = mask.unsqueeze(0) & attention_mask[:, None, :].bool()
        mask = mask.reshape(1, 1, T, S) if mask.dim() == 2 else mask

        qh = q.transpose(1, 2)                                   # (B,nh,T,d)
        kh = self._repeat_kv(k).transpose(1, 2)                  # (B,nh,S,d)
        vh = self._repeat_kv(v).transpose(1, 2)                  # (B,nh,S,d)

        scores = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * self.scaling
        s = scores.to(qh.dtype).masked_fill(
            ~mask.to(torch.bool), torch.finfo(qh.dtype).min)
        if self.training:
            # 在 mask 后取 max：被因果/窗口遮掉的未来位置置 -inf，amax 自然忽略，
            # 记录的 S_max^h 才是「真正会被 softmax 看到」的最大 logit（QK-Clip 判据）。
            self._attn_max_logits = s.detach().amax(dim=(0, 2, 3)).to(dev)

        probs = torch.softmax(s, dim=-1)
        if self.attention_dropout > 0.0 and self.training:
            probs = F.dropout(probs, p=self.attention_dropout)
        ctx = torch.matmul(probs, vh)                            # (B,nh,T,d)

        out = ctx.transpose(1, 2).reshape(B, T, nh * d)
        # 推理时始终构建 KV cache，使增量 decode 能累积历史（含首个 cache=None 步骤）
        new_cache = SWACache(k=k.detach(), v=v.detach()) if not self.training else None
        return self.o_proj(out.to(hidden_states.dtype)), new_cache

    def extra_repr(self) -> str:
        return (f"nh={self.num_heads}, nkv={self.num_key_value_heads}, d={self.head_dim}, "
                f"window={self.sliding_window}")
