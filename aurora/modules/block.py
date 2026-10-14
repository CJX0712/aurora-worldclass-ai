# -*- coding: utf-8 -*-
"""
Aurora Block —— 一个模块同时覆盖 M / D / S 三种 token mixer。

这是 Aurora 与「每个模型一个结构」范式的根本差异：在这里，mixer 是一个**可插拔的
slot**。同一份代码跑 DeepSeek-V3（全 MLA）、Gemma 3（5:1 局部:全局）、Qwen3.5
（~75% GDN）或纯 recurrent，只是 `mixer_pattern` 字符串不同。

Pre-norm 残差结构（2026 旗舰统一形态）:

    h = h + mixer( RMSNorm(h) )
    h = h + ffn(   RMSNorm(h) )

注意顺序：RMSNorm 在残差**内部**（pre-norm），而非外层。这是 DeepSeek / LLaMA
系列的默认，比 Post-LN 训练更稳定（不需要 warmup 到极致）。

mixer 选择（由 config.mixer_plan[layer_idx] 决定）:

    M  → MultiHeadLatentAttention   （全局，KV 低秩压缩）
    D  → GatedDeltaNet              （线性注意力， O(1) 状态）
    S  → SlidingWindowAttention     （局部 softmax + GQA）

FFN 选择（由 config.is_moe_layer 决定）:

    MoE 层 → DeepSeekMoE（细粒度专家 + 共享专家 + Quantile Balancing）
    稠密层 → SwiGLU FFN

Invariants（架构层）
-------------------
1. 每个 mixer 的输入/输出都是 (B,T,d_h)，可无感替换。
2. 残差连接不破坏 d_h 维度（gradient 可直通）。
3. 无论 mixer 类型，forward 返回 (hidden, new_cache)，cache 类型随 mixer 变化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from aurora.config import AuroraConfig
from aurora.modules.norms import RMSNorm
from aurora.modules.mla import MultiHeadLatentAttention, MLACache
from aurora.modules.gdn import GatedDeltaNet, DeltaNetCache
from aurora.modules.attention import SlidingWindowAttention, SWACache
from aurora.modules.moe import DeepSeekMoE, SwiGLUExpert


__all__ = ["AuroraBlock", "BlockCache", "USE_MIXER_KIND"]


USE_MIXER_KIND = {"M": "mla", "D": "gdn", "S": "swa"}


@dataclass
class BlockCache:
    """一个 Block 的增量缓存。其 .mixer 字段的具体类型取决于该层的 mixer。"""

    mixer: object

    @property
    def seq_len(self) -> int:
        return getattr(self.mixer, "seq_len", 0)


def build_mixer(config: AuroraConfig, layer_idx: int) -> nn.Module:
    kind = config.mixer_plan[layer_idx]
    if kind == "M":
        return MultiHeadLatentAttention(
            hidden_size=config.hidden_size, num_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank, kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim, qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim, rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            rope_scaling_type=config.rope_scaling_type,
            rope_scaling_factor=config.rope_scaling_factor,
            yarn_original_max_position=config.yarn_original_max_position,
            yarn_beta_fast=config.yarn_beta_fast, yarn_beta_slow=config.yarn_beta_slow,
            rms_norm_eps=config.rms_norm_eps, mode="auto",
        )
    if kind == "D":
        return GatedDeltaNet(
            hidden_size=config.hidden_size, num_heads=config.gdn_num_heads,
            key_dim=config.gdn_key_dim, value_dim=config.gdn_value_dim,
            conv_size=config.gdn_conv_size, use_beta=config.gdn_use_beta,
            use_gate=config.gdn_use_gate, allow_neg_eigval=config.gdn_allow_neg_eigval,
            layer_idx=layer_idx, chunk_size=config.gdn_chunk_size,
            norm_eps=config.gdn_norm_eps,
        )
    if kind == "S":
        return SlidingWindowAttention(
            hidden_size=config.hidden_size, num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads, head_dim=config.swa_head_dim,
            sliding_window=config.sliding_window, rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            rope_scaling_type=config.rope_scaling_type,
            rope_scaling_factor=config.rope_scaling_factor,
            rms_norm_eps=config.rms_norm_eps,
        )
    raise ValueError(f"未知 mixer kind: {kind}")


def build_ffn(config: AuroraConfig, layer_idx: int) -> nn.Module:
    if config.is_moe_layer(layer_idx) and config.n_routed_experts > 0:
        return DeepSeekMoE(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            scoring_func=config.scoring_func, norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor, n_group=config.n_group,
            topk_group=config.topk_group, topk_method=config.topk_method,
            use_quantile_balancing=config.use_quantile_balancing,
            quantile_lr=config.quantile_lambda, balance_steps=8,
            use_aux_loss=config.use_aux_loss, aux_loss_alpha=config.aux_loss_alpha,
        )
    return SwiGLUExpert(config.hidden_size, config.intermediate_size)


class AuroraBlock(nn.Module):
    """统一的 Transformer Block。详见模块 docstring。"""

    def __init__(self, config: AuroraConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.mixer_kind = config.mixer_plan[layer_idx]
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mixer = build_mixer(config, layer_idx)
        self.feed_forward = build_ffn(config, layer_idx)

        # 用于在不修改 mixer 内部的情况下访问其缓存类型
        self._mixer_cache_cls = {
            "M": MLACache, "D": DeltaNetCache, "S": SWACache,
        }[self.mixer_kind]

    # ------------------------------------------------------------------
    def empty_cache(self, batch_size, dtype, device) -> BlockCache:
        if self.mixer_kind == "M":
            c = MLACache.empty(batch_size, dtype, device,
                               self.mixer.kv_lora_rank, self.mixer.qk_rope_head_dim)
        elif self.mixer_kind == "D":
            c = DeltaNetCache.empty(batch_size, self.mixer.num_heads, self.mixer.key_dim,
                                    self.mixer.value_dim, self.mixer.conv_size, dtype, device)
        else:
            c = SWACache.empty(batch_size, self.mixer.num_key_value_heads,
                              self.mixer.head_dim, dtype, device)
        return BlockCache(mixer=c)

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[BlockCache] = None,
    ) -> Tuple[torch.Tensor, Optional[BlockCache]]:
        B, T, _ = hidden_states.shape
        residual = hidden_states

        # ---- mixer（pre-norm）----
        normed = self.input_layernorm(hidden_states)
        mixer_out, new_mixer_cache = self.mixer(
            normed, attention_mask=attention_mask,
            cache=(cache.mixer if cache is not None else None),
        )
        hidden_states = residual + mixer_out

        # ---- ffn（pre-norm）----
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        ffn_out = self.feed_forward(normed)
        hidden_states = residual + ffn_out

        new_cache = BlockCache(mixer=new_mixer_cache) if new_mixer_cache is not None else None
        return hidden_states, new_cache

    def extra_repr(self) -> str:
        return f"layer={self.layer_idx}, mixer={self.mixer_kind} ({USE_MIXER_KIND[self.mixer_kind]})"
