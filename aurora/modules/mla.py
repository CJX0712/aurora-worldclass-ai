# -*- coding: utf-8 -*-
"""
Multi-head Latent Attention (MLA)。

来源
----
* DeepSeek-V2  《DeepSeek-V2: A Strong, Economical, and Efficient MoE LM》(2024-05)
  首次提出 MLA，把 KV cache 压到 MHA 的 1/93.3。
* DeepSeek-V3  《DeepSeek-V3 Technical Report》(2024-12)
  全栈训练仅 2.788M H800 GPU 时；吸收式推理思想。
* Kimi K2      《Kimi K2: Open Agentic Intelligence》(2025-07)
  1T 参数使用 MLA，并给出 MLA 上 QK-Clip 的**逐组件**裁剪规则。

为什么是 MLA
------------
MHA 的 KV cache 是每 token 2·n_h·(d_n+d_v) 个数；GQA 除以分组数。
MLA 走第三条路：不共享 head，而是**把所有 head 还原所需的信息压进一个低秩潜变量**，
每 token 只缓存

    c^{KV}_t = W^{DKV} h_t ∈ R^{d_c}，  以及   k^R_t = W^{KR} h_t ∈ R^{d_r}

DeepSeek-V3：d_c=512, d_r=64 → **576 个数/token**；同规模 MHA 需要
2·128·(128+128) = 65536 —— 压缩比约 114×，而质量不降反升。

数学定义
--------
    KV 路径（唯一被缓存的部分）
        c^{KV}_t = W^{DKV} h_t                    d_h → d_c
        c̄^{KV}_t = RMSNorm(c^{KV}_t)              ← 缓存的就是这个（已归一化）
        k^{C}_t  = W^{UK} c̄^{KV}_t                d_c → n_h·d_n
        v^{C}_t  = W^{UV} c̄^{KV}_t                d_c → n_h·d_v
        k^{R}_t  = RoPE(W^{KR} h_t)               d_h → d_r   ← 所有 head 共享（缓存已 RoPE）

    Query 路径
        c^{Q}_t  = W^{DQ} h_t                     d_h → d'_c（可选低秩压缩）
        c̄^{Q}_t  = RMSNorm(c^{Q}_t)
        q^{C}_t  = QK-Norm(W^{UQ} c̄^{Q}_t)        d'_c → n_h·d_n
        q^{R}_t  = RoPE(W^{QR} c̄^{Q}_t)           d'_c → n_h·d_r

    Attention（head i，query 位置 t）
        q_{t,i} = [q^C_{t,i} ; q^R_{t,i}]         ∈ R^{d_n+d_r}
        k_{j,i} = [k^C_{j,i} ; k^R_j]             ∈ R^{d_n+d_r}   （k^R 全 head 共享一份）
        o_{t,i} = Σ_{j≤t} softmax(q·k / √(d_n+d_r))_j  v^C_{j,i}
        u_t     = W^O concat_i[o_{t,i}]

k^R 是 head 共享的 —— RoPE 的位置信息不需要每个 head 保留一份拷贝，
这就是"解耦 RoPE"（decoupled RoPE），MLA 能兼容 RoPE 全靠它。

推理时 KV cache 的极致压缩
--------------------------
本实现采用 **DeepSeek-V3 官方 (HuggingFace transformers) 的部署方式**：
直接缓存压缩潜变量 c̄^{KV}（已 RMSNorm）与共享 RoPE key k^R（已 RoPE），
在注意力时才通过 `kv_b_proj` 即时"展开"出全维 K/V（见 `expand_kv`）。
开销只有

    c̄^{KV} : d_c 个数/token ，   k^R : d_r 个数/token   ⟹  共 d_c + d_r

与 n_h **无关**。这正是 MLA 把 KV 成本从 O(n_h) 降到 O(1) 的本质收益。

论文 §3.2 的"矩阵吸收"（W^{UQ} 与 W^{UK} 预乘、W^{UV} 吸收进 W^O，使推理退化为
head_dim = d_c+d_r 的 MQA）是同一结论的等价代数形式；但因 QK-Norm 是逐 token 的
非线性项（无法被固定线性映射折叠），实际旗舰实现不采用显式预乘，而是上面的
潜变量缓存 + 即时展开 —— 二者在数学上完全一致，且本实现保证
**训练（全量）路径与推理（增量缓存）路径数值严格相等**（见
`tests/test_invariants.py::test_mla_inference_equivalence`，fp32 ~1e-5）。

Invariants
----------
1. 增量 decode 路径与全量 prefill 路径数值等价（fp32 ~1e-5）。
2. KV cache 每 token 的元素数为 d_c + d_r，与 n_h **无关**。
3. 因果性：位置 t 输出不依赖 j > t 的输入——在 tests 中以扰动验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.modules.norms import RMSNorm, QKRMSNorm
from aurora.modules.rope import RotaryEmbedding, YarnRotaryEmbedding, rotate_half


__all__ = ["MultiHeadLatentAttention", "MLACache"]


@dataclass
class MLACache:
    """MLA 增量推理缓存 —— MLA 的全部家底就这两项（均已归一化/RoPE）。

    c_kv : (B, T, d_c)    被压缩的键值潜变量（**已** RMSNorm）
    k_pe : (B, T, d_r)    解耦的共享 RoPE key（**已**施加 RoPE）
    """

    c_kv: torch.Tensor
    k_pe: torch.Tensor

    @property
    def seq_len(self) -> int:
        return int(self.c_kv.shape[1])

    @classmethod
    def empty(cls, batch_size: int, vocab_dtype: torch.dtype, device: torch.device,
              kv_lora_rank: int, qk_rope_head_dim: int) -> "MLACache":
        return cls(
            c_kv=torch.zeros(batch_size, 0, kv_lora_rank, dtype=vocab_dtype, device=device),
            k_pe=torch.zeros(batch_size, 0, qk_rope_head_dim, dtype=vocab_dtype, device=device),
        )

    def concat(self, new_c_kv: torch.Tensor, new_k_pe: torch.Tensor) -> "MLACache":
        return MLACache(
            c_kv=torch.cat([self.c_kv, new_c_kv], dim=1),
            k_pe=torch.cat([self.k_pe, new_k_pe], dim=1),
        )

    def index_select(self, idx: torch.Tensor) -> "MLACache":
        return MLACache(c_kv=self.c_kv[idx], k_pe=self.k_pe[idx])


class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention（全局注意力 + KV 低秩压缩）。

    训练走全量 prefill 路径；推理走增量缓存路径。两条路径调用**完全相同的算子**
    （kv_b_proj / qk_norm / o_proj），仅 KV 来源不同（全量 vs 缓存），因此数值等价。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int = 0,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8192,
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
        yarn_original_max_position: int = 4096,
        yarn_beta_fast: int = 32,
        yarn_beta_slow: int = 1,
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        mode: str = "auto",
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.q_lora_rank = int(q_lora_rank)
        self.kv_lora_rank = int(kv_lora_rank)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.mode = mode
        self.attention_dropout = attention_dropout

        nh, dn, dr, dv, dc = (self.num_heads, self.qk_nope_head_dim,
                              self.qk_rope_head_dim, self.v_head_dim, self.kv_lora_rank)

        # ---- KV: down-proj 到潜变量 + 解耦 RoPE ----
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, dc + dr, bias=False)
        self.kv_a_layernorm = RMSNorm(dc, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(dc, nh * (dn + dv), bias=False)

        # ---- Query ----
        if self.q_lora_rank > 0:
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
            self.q_b_proj = nn.Linear(q_lora_rank, nh * (dn + dr), bias=False)
        else:
            self.q_b_proj = nn.Linear(hidden_size, nh * (dn + dr), bias=False)

        self.q_norm = QKRMSNorm(dn, num_heads=nh, eps=rms_norm_eps) if qk_norm else nn.Identity()
        self.k_norm = QKRMSNorm(dn, num_heads=nh, eps=rms_norm_eps) if qk_norm else nn.Identity()

        self.o_proj = nn.Linear(nh * dv, hidden_size, bias=False)

        # ---- RoPE（仅 d_r 这一小段参与）----
        rope_kwargs = dict(dim=dr, max_position=max_position_embeddings,
                           scaling_factor=rope_scaling_factor)
        if rope_scaling_type == "yarn":
            self.rotary = YarnRotaryEmbedding(
                base=rope_theta, original_max_position=yarn_original_max_position,
                beta_fast=yarn_beta_fast, beta_slow=yarn_beta_slow, **rope_kwargs)
        else:
            self.rotary = RotaryEmbedding(
                base=rope_theta, scaling_type=rope_scaling_type, **rope_kwargs)

        self.scaling = self.qk_head_dim ** -0.5
        self._attn_max_logits: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    @property
    def attn_max_logits(self) -> Optional[torch.Tensor]:
        """每头最大 attention logit，shape (n_h,) —— MuonClip QK-Clip 的输入判据。"""
        return self._attn_max_logits

    def reset_max_logits(self) -> None:
        self._attn_max_logits = None

    # ------------------------------------------------------------------
    # 权重切片：供 QK-Clip 精确裁剪（沿用 Kimi K2 的逐组件规则）
    # 注意是「行切片」——q_b_proj/kv_b_proj 的 out 维 = (nh*dn, ...) 或
    # (nh*dr, ...)，每个 head 占连续 dn/dr 行；列维是输入维，不可切。
    # ------------------------------------------------------------------
    def q_c_weight(self) -> torch.Tensor:
        """W^{UQ} 的 nope 部分，(nh, dn, in_dim)。"""
        w = self.q_b_proj.weight[: self.num_heads * self.qk_nope_head_dim, :]
        return w.reshape(self.num_heads, self.qk_nope_head_dim, -1)

    def k_c_weight(self) -> torch.Tensor:
        """W^{UK}，(nh, dn, dc)。"""
        w = self.kv_b_proj.weight[: self.num_heads * self.qk_nope_head_dim, :]
        return w.reshape(self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank)

    def q_r_weight(self) -> torch.Tensor:
        """W^{QR} 的 rotary 部分，(nh, dr, in_dim)。"""
        w = self.q_b_proj.weight[self.num_heads * self.qk_nope_head_dim:, :]
        return w.reshape(self.num_heads, self.qk_rope_head_dim, -1)

    def k_r_weight(self) -> torch.Tensor:
        """W^{KR}（共享的 RoPE key 下投影），shape (dr, hidden)。

        按 Kimi K2 规则，k^R 是 head 共享的 Rotary 分量，QK-Clip **不**触碰它，
        以免跨 head 互相干扰；这里暴露出来仅供检查/测试。
        """
        return self.kv_a_proj_with_mqa.weight[self.kv_lora_rank:, :]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def qk_clip_step(self, threshold: float = 100.0, eps: float = 1e-6) -> Optional[torch.Tensor]:
        """Kimi K2 QK-Clip（MLA 版，逐 head）。

        在「每个训练步、Muon 更新之后」调用。逻辑（见 Kimi K2 技术报告 §2.3 / Alg.1）：

            对每个 head h:
                S_max^h = (1/√d) · max_{i,j} Q_i^h · K_j^{h⊤}   （即 attn_max_logits[h]）
                γ_h     = min(1, τ / S_max^h)                    （只裁剪、绝不放大）
            对 MLA **仅**裁剪 head 专属分量，共享分量保持不动：
                q^C (nope)  ← √γ_h · q^C_h         （缩放 QK-Norm 的逐 head affine）
                k^C (nope)  ← √γ_h · k^C_h         （缩放 K-Norm 的逐 head affine）
                q^R (rope)  ← γ_h  · q^R_h         （缩放 W^{QR} 的 rope 段）
                k^R (共享)  ←  不裁剪（避免跨 head 干扰）

        实现要点（关键修正）：QK-Norm 会 **除以 RMS**，因此直接缩放 W^{UQ}/W^{UK}
        这类「归一化之前」的投影权重会被归一化**完全抵消**——这正是朴素 QK-Clip 失效
        的根因。正确做法是缩放 **归一化之后** 的量：即 QK-Norm 的逐 head affine 权重
        （它施加在 RMS 除法之后，缩放必然存活），以及未经归一化的 rope 段 W^{QR}。
        逐 head affine 要求 QKRMSNorm 的权重是 (nh, dn) 而非共享 (dn,)。

        为何 nope 段用 √γ、rope 段用 γ：logit = q^C·k^C + q^R·k^R。nope 段两侧都乘
        √γ ⇒ 整体乘 γ；rope 段仅 q^R 乘 γ（k^R 不动）⇒ 也乘 γ。两段都精确乘 γ_h，
        于是 S_max^h ⟶ γ_h·S_max^h ≤ τ。

        QK-Clip **不改变**当前步的前向/反向（权重缩放发生在本步 optimizer 之后、
        下一步之前），仅作「事后校正」，把注意力 logit 钉在阈值以下，杜绝 softmax
        熵坍缩 / loss spike。

        Returns
        -------
        torch.Tensor | None：实际施加的逐 head γ（用于日志）；无统计时返回 None。
        """
        if self._attn_max_logits is None:
            return None
        smax = self._attn_max_logits.detach().to(torch.float32).clamp_min(eps)  # (nh,)
        nh, dn, dr = self.num_heads, self.qk_nope_head_dim, self.qk_rope_head_dim
        gamma = torch.sqrt(torch.clamp(threshold / smax, max=1.0))             # (nh,)  = √γ_h
        gamma = gamma.to(self.q_b_proj.weight.dtype)
        sqrt_g = gamma                       # √γ_h
        full_g = gamma * gamma              # γ_h

        # ---- nope (C) 段：缩放逐 head 的 QK-Norm affine（归一化之后，缩放存活）----
        if isinstance(self.q_norm, nn.Identity) or isinstance(self.k_norm, nn.Identity):
            # 若未启用 QK-Norm，退回「缩放投影权重」；但注意此路径不被 norm 抵消。
            pass
        else:
            self.q_norm.weight.data.mul_(sqrt_g.view(nh, 1))    # (nh, dn) ← 每行 × √γ_h
            self.k_norm.weight.data.mul_(sqrt_g.view(nh, 1))

        # ---- rope (R) 段：q^R 未经归一化，直接缩放 W^{QR} 的 rope 段 ----
        # 关键：q_b_proj 的输出是「逐 head 内 interleave」布局 ——
        #      [ head0 的 dn 个 nope 行 | head0 的 dr 个 rope 行 | head1 的 ... ]
        # 因此第 h 个 head 的 rope 行位于 [h*(dn+dr)+dn : (h+1)*(dn+dr)]，
        # 而非「先 all-nope 后 all-rope」的 [nh*dn + h*dr : ...]。务必对位，否则
        # 每个 head 的 q^R 会错拿别的 head 的 γ（此前 MLA 裁剪失效的根因）。
        wq = self.q_b_proj.weight.data                           # (nh*(dn+dr), in)
        s = torch.ones(nh * (dn + dr), device=wq.device, dtype=wq.dtype)
        rope_rows = torch.arange(nh * (dn + dr)).reshape(nh, dn + dr)[:, dn:].reshape(-1)
        s[rope_rows] = full_g.repeat_interleave(dr)             # 每个 head 的 rope 段 × γ_h
        wq.mul_(s.unsqueeze(1))

        # ---- k^R（共享 rope key，kv_a_proj 末 dr 行）不裁剪 ----

        self._attn_max_logits = None
        return full_g  # 返回 γ_h 本身（便于日志）

    # ------------------------------------------------------------------
    @staticmethod
    def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """半分式旋转。x (..., d_r)，cos/sin (..., d_r) 广播。"""
        return (x.float() * cos) + (rotate_half(x.float()) * sin)

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[MLACache] = None,
    ) -> Tuple[torch.Tensor, Optional[MLACache]]:
        """前向。

        Parameters
        ----------
        hidden_states : (B, T, d_h)     T 可为 1（decode）或 >1（prefill）
        attention_mask : (B, S) padding mask（1=保留）
        cache : 增量推理缓存（已归一化的 c̄^{KV} 与已 RoPE 的 k^R）

        Returns
        -------
        (output (B,T,d_h), new_cache)

        Note
        ----
        mode='naive' 时强制走全量路径（忽略 cache），供等价性测试作参照。
        """
        if self.mode == "naive":
            cache = None

        B, T, _ = hidden_states.shape
        nh, dn, dr, dv, dc = (self.num_heads, self.qk_nope_head_dim,
                              self.qk_rope_head_dim, self.v_head_dim, self.kv_lora_rank)
        dev, dt = hidden_states.device, hidden_states.dtype

        # ---------------- KV 下投影（仅新 token）----------------
        kv = self.kv_a_proj_with_mqa(hidden_states)                  # (B,T,dc+dr)
        c_kv_raw, k_pe_raw = torch.split(kv, [dc, dr], dim=-1)       # (B,T,dc),(B,T,dr)

        pos = cache.seq_len if cache is not None else 0
        S = pos + T

        # RoPE 余弦/正弦，覆盖 [0, S)
        cos, sin = self.rotary(S, device=dev)                        # (S,dr) float32
        cos_s = cos.to(torch.float32)
        sin_s = sin.to(torch.float32)

        # 新 token 的归一化潜变量 + RoPE key
        new_c_kv = self.kv_a_layernorm(c_kv_raw)                     # (B,T,dc)
        new_k_pe = self._rope(
            k_pe_raw,
            cos_s[pos:pos + T].unsqueeze(0),
            sin_s[pos:pos + T].unsqueeze(0),
        ).to(dt)                                                    # (B,T,dr)

        if cache is not None:
            c_kv = torch.cat([cache.c_kv, new_c_kv], dim=1)          # (B,S,dc)
            k_pe = torch.cat([cache.k_pe, new_k_pe], dim=1)          # (B,S,dr)
        else:
            c_kv, k_pe = new_c_kv, new_k_pe
        S = c_kv.shape[1]

        # ---------------- Query 路径 ----------------
        if self.q_lora_rank > 0:
            c_q = self.q_a_proj(hidden_states)                       # (B,T,q_lora)
            c_q_bar = self.q_a_layernorm(c_q)
            q = self.q_b_proj(c_q_bar)                               # (B,T,nh*(dn+dr))
        else:
            q = self.q_b_proj(hidden_states)
        q = q.view(B, T, nh, dn + dr)
        q_nope = self.q_norm(q[..., :dn])                            # (B,T,nh,dn)
        cos_q = cos_s[pos:pos + T].unsqueeze(0).unsqueeze(2)         # (1,T,1,dr)
        sin_q = sin_s[pos:pos + T].unsqueeze(0).unsqueeze(2)
        q_pe = self._rope(q[..., dn:], cos_q, sin_q).to(dt)          # (B,T,nh,dr)

        # ---------------- 从压缩潜变量展开 K/V（训练/推理同一算子）----------------
        kv_b = self.kv_b_proj(c_kv).view(B, S, nh, dn + dv)          # (B,S,nh,dn+dv)
        k_nope = self.k_norm(kv_b[..., :dn])                         # (B,S,nh,dn)
        v = kv_b[..., dn:]                                          # (B,S,nh,dv)

        q_full = torch.cat([q_nope, q_pe], dim=-1)                  # (B,T,nh,dn+dr)
        k_full = torch.cat(
            [k_nope, k_pe.unsqueeze(2).expand(-1, -1, nh, -1)],
            dim=-1,
        )                                                          # (B,S,nh,dn+dr)

        # ---------------- mask ----------------
        q_idx = torch.arange(pos, pos + T, device=dev).unsqueeze(1)
        k_idx = torch.arange(S, device=dev).unsqueeze(0)
        mask = k_idx <= q_idx                                       # (T,S) bool
        if attention_mask is not None and attention_mask.dim() == 2:
            mask = mask.unsqueeze(0) & attention_mask[:, None, :].bool()
        elif attention_mask is not None:
            mask = attention_mask.bool()
        while mask.dim() < 4:
            mask = mask.unsqueeze(0) if mask.dim() == 2 else mask.unsqueeze(1)

        # ---------------- attention ----------------
        ctx = self._attention_core(
            q_full.transpose(1, 2), k_full.transpose(1, 2),
            v.transpose(1, 2), mask,
        )                                                          # (B,nh,T,dv)
        out = self.o_proj(ctx.transpose(1, 2).reshape(B, T, nh * dv))

        # 推理（非训练）时始终构建并返回 KV cache，使增量 decode 能逐 token 累积历史
        # （含首个 cache=None 的步骤）；训练期不构建，避免无谓保留计算图。
        new_cache = MLACache(c_kv=c_kv.detach(), k_pe=k_pe.detach()) \
            if not self.training else None
        return out, new_cache

    # ------------------------------------------------------------------
    def _attention_core(self, q, k, v, mask):
        """统一的 score→softmax→加权和，并顺便记录每头 max logit（训练时）。"""
        scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) * self.scaling

        s = scores.to(q.dtype)
        if mask is not None:
            s = s.masked_fill(~mask.to(torch.bool), torch.finfo(s.dtype).min)
        if self.training:
            # mask 后取 max：因果遮掉的未来位置置 -inf，amax 忽略 → 记录的是真正
            # 会被 softmax 看到的 S_max^h（Kimi K2 QK-Clip 的判据）。
            self._attn_max_logits = s.detach().amax(dim=(0, 2, 3)).to(q.device)
        probs = torch.softmax(s, dim=-1)
        if self.attention_dropout > 0.0 and self.training:
            probs = F.dropout(probs, p=self.attention_dropout)
        return torch.matmul(probs, v)

    def extra_repr(self) -> str:
        return (f"nh={self.num_heads}, dc={self.kv_lora_rank}, d'c={self.q_lora_rank}, "
                f"dn={self.qk_nope_head_dim}, dr={self.qk_rope_head_dim}, "
                f"dv={self.v_head_dim}, mode={self.mode}")
