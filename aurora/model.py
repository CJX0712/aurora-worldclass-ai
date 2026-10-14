# -*- coding: utf-8 -*-
"""
Aurora 模型骨架。

    AuroraModel         主干（embedding → N 个 AuroraBlock → final RMSNorm）
    AuroraForCausalLM   加 LM head + MTP 头 + 损失函数

设计要点
--------
* **权重绑定（weight tying）**：lm_head 与 embedding 共享权重，这是标准做法，
  既能省参数又能在词表头部形成「embedding ↔ logit」对称。DeepSeek-V3、Qwen、LLaMA
  全部采用。
* **MTP 作为一等公民**：Multi-Token Prediction 头直接挂在主干之上，与主模型共享
  embedding 与 lm_head。它在训练时提供 n 倍密度的监督信号，在推理时自动变成
  自投机解码的草稿器 —— 零额外参数、零分布漂移。
* **缓存是 per-layer 的异构对象**：MLA 产 MLACache（小），GDN 产 DeltaNetCache
  （常数大小），SWA 产 SWACache（随长度线性）。AuroraBlock 把这种异构性封装掉，
  对外只暴露统一的 `BlockCache`。

损失（训练）
--------
    L = L_CE(logits, labels)
      + (λ_mtp / n) · Σ_{k=1}^{n} L_CE(mtp_logits_k, labels shifted by k)

其中 mtp_logits_k 由第 k 个 MTP 头对「深度 k 的隐状态」做 predicts。这正是
DeepSeek-V3 的 MTP loss 定义。

推理
----
提供两种入口：
  - `forward(...)`               完整前向（prefill / 单步）
  - `generate(...)`             便捷自回归循环（教学 / 小模型 demo 用）
  真正的**高吞吐引擎**在 aurora.inference，包含分页 KV、连续批处理、MTP 自投机。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from aurora.config import AuroraConfig
from aurora.modules.block import AuroraBlock, BlockCache
from aurora.modules.mtp import MultiTokenPredictor
from aurora.modules.norms import RMSNorm


__all__ = ["AuroraModel", "AuroraForCausalLM", "CausalLMOutput"]


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    mtp_logits: Optional[List[torch.Tensor]] = None
    past_key_values: Optional[List[Optional[BlockCache]]] = None
    hidden_states: Optional[torch.Tensor] = None


def shift_labels(logits: torch.Tensor, labels: torch.Tensor):
    """把 logits 与目标对齐，并计算 CE。返回 (loss, )。

    logits: (B, T, V)；labels: (B, T)。第 t 个位置的 logit 预测第 t+1 个 token。
    """
    B, T, V = logits.shape
    shift_logits = logits[:, :-1].contiguous().view(-1, V)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    loss = nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=-100)
    return loss


class AuroraModel(nn.Module):
    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [AuroraBlock(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ---- MTP 头（与主模型共享 embed / lm_head）----
        self.mtp: Optional[MultiTokenPredictor] = None
        if config.num_nextn_predict_layers > 0:
            n = config.num_nextn_predict_layers
            # MTP 的每个深度头是一个完整 block；这里统一用 MLA 作其 mixer，
            # 并把子配置层数设为 n，避免 build_mixer_plan 的长度校验误报。
            mtp_cfg = config.evolve(mixer_pattern="M" * n, num_layers=n)
            mtp_blocks = nn.ModuleList(
                [AuroraBlock(mtp_cfg, 0) for _ in range(n)])
            self.mtp = MultiTokenPredictor(
                config.hidden_size, mtp_blocks, rms_norm_eps=config.rms_norm_eps)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, std=self.config.initializer_range)
        # 深层残差的标准做法：N(0, σ) 且 σ 随深度退火一点，这里用统一 σ
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.normal_(m.weight, std=self.config.initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, w):
        self.embed_tokens = w

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[BlockCache]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[BlockCache]]]]:
        B, T = input_ids.shape
        device = input_ids.device
        if past_key_values is None:
            past_key_values = [None] * self.config.num_layers

        h = self.embed_tokens(input_ids)

        new_caches = []
        for i, block in enumerate(self.layers):
            cache_i = past_key_values[i]
            h, new_cache = block(
                h, attention_mask=attention_mask, position_ids=position_ids,
                cache=cache_i)
            new_caches.append(new_cache)

        h = self.norm(h)
        return h, (new_caches if use_cache else None)

    def extra_repr(self) -> str:
        c = self.config
        return (f"layers={c.num_layers} ({c.num_layers_by_kind}), "
                f"d={c.hidden_size}, V={c.vocab_size}, mtp={c.num_nextn_predict_layers}")


class AuroraForCausalLM(nn.Module):
    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.config = config
        self.model = AuroraModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.mtp_loss_factor = config.mtp_loss_factor

    # ------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def num_parameters(self, trainable_only: bool = False) -> int:
        n = 0
        for p in self.parameters():
            if p.requires_grad or not trainable_only:
                n += p.numel()
        return n

    # ------------------------------------------------------------------
    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden.to(self.lm_head.weight.dtype))

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[BlockCache]]] = None,
        use_cache: bool = False,
        output_mtp_logits: bool = False,
    ) -> CausalLMOutput:
        trunk_hidden, new_caches = self.model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache)

        logits = self._logits(trunk_hidden)

        loss = None
        mtp_logits: Optional[List[torch.Tensor]] = None
        if labels is not None:
            loss = shift_labels(logits, labels)

        if (labels is not None and self.model.mtp is not None) or \
           (output_mtp_logits and self.model.mtp is not None):
            mtp_hiddens = self.model.mtp(
                trunk_hidden, embed_fn=self.model.embed_tokens, input_ids=input_ids,
                attention_mask=attention_mask, position_ids=position_ids)
            mtp_logits = [self._logits(h) for h in mtp_hiddens]
            if labels is not None:
                mtp_loss = 0.0
                for k, ml in enumerate(mtp_logits, start=1):
                    if ml.shape[1] < 2:
                        continue
                    mtp_loss = mtp_loss + shift_labels(ml, labels[:, k:])
                mtp_loss = mtp_loss / max(1, len(mtp_logits))
                loss = loss + self.mtp_loss_factor * mtp_loss

        return CausalLMOutput(
            logits=logits, loss=loss, mtp_logits=mtp_logits, past_key_values=new_caches,
            hidden_states=trunk_hidden)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """一个朴素但正确的自回归循环，供教学 / 小规模 demo 使用。

        高吞吐推理请使用 aurora.inference.AuroraEngine（分页 KV + 连续批处理 +
        MTP 自投机）。
        """
        self.eval()
        device = input_ids.device
        generated = input_ids
        past = None
        for _ in range(max_new_tokens):
            # 增量 decode：有 KV cache 时只喂入最新一个 token，避免与 cache 重复计算。
            cur_input = generated[:, -1:] if past is not None else generated
            out = self.forward(cur_input, past_key_values=past, use_cache=use_cache)
            past = out.past_key_values
            logits = out.logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, -float("inf"))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(sorted_logits, -1), -1)
                remove = cum > top_p
                remove[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                logits = torch.gather(sorted_logits, -1, sorted_idx.argsort(-1))
            probs = torch.softmax(logits, -1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=-1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
        return generated

    def extra_repr(self) -> str:
        return (f"params≈{self.num_parameters()/1e6:.1f}M, "
                f"mtp={self.config.num_nextn_predict_layers}")
