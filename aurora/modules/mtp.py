# -*- coding: utf-8 -*-
"""
Multi-Token Prediction (MTP)。

来源
----
* 《Better & Faster Large Language Models via Multi-token Prediction》(Gloeckle et al., 2024)
  Meta。首次系统提出 n-gram 式多步预测，训练信号密度从 1×(T) 提升到 n×(T)。
* DeepSeek-V3 Technical Report (2024-12)
  把它做成了可工程化的「MTP Module」，并在推理时**直接复用为自投机解码的草稿器**，
  白捡 ~1.8× 的解码加速。这是教科书级别的"一个模块两个用途"。
* MiMo / Nemotron 亦采用。

为什么有效（两条独立的理由）
----------------------------
1. **训练侧**：每一个位置的监督信号从 1 个 token 变成 n 个。对于难样本（长尾知识、
   推理链），单 token 监督常常给出误导性的局部最优；多步目标强制模型内部形成
   「向前看」的隐式规划表征。

2. **推理侧**：MTP 的第 k 个头天然是第 k 步草稿器。传统投机解码需要额外训练一个小
   草稿模型（Medusa/EAGLE 还要解决一致性问题）；MTP 的草稿器**与主模型共享全部
   权重和 trunk**，不存在分布漂移，且零额外参数开销（只有 n 个轻量投影 + n 个 block）。

结构（严格遵循 DeepSeek-V3）
----------------------------
共享 embedding E 与 output head H（不新增）；
每个深度 k = 1..n 拥有自己的一套：

    MTPModule_k : ℎ_{T-k} , embed(t_{T-k+1:...}) ⟶ concat ⟶ Linear ⟶ Block_k ⟶ RMSNorm

具体实现：

    h_cat = concat[ RMSNorm(h_prev) , RMSNorm(E(t_next)) ]      2d_h
    g     = Linear(2d_h → d_h)(h_cat)
    h^(k) = TransformerBlock_k(hidden = g, causal mask, positions)
    y^(k) = H( RMSNorm(h^(k)) )

注意 TransformerBlock_k 是**完整的一层**（带自己的 attention/FFN），不是纯 MLP。
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from aurora.modules.norms import RMSNorm


__all__ = ["AuroraMTPHead", "MultiTokenPredictor"]


class AuroraMTPHead(nn.Module):
    """单个深度的 MTP 头。

    Parameters
    ----------
    hidden_size : d_h
    block : nn.Module
        一个完整的 Aurora Block。为保持因果性，它内部会用**同一份** causal mask。
    rms_norm_eps : 归一化 eps
    """

    def __init__(self, hidden_size: int, block: nn.Module, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.enorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.hnorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.block = block
        self.shared_head_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        next_embed: torch.Tensor,
        **block_kwargs,
    ) -> torch.Tensor:
        """prev_hidden: (B, T, d) ; next_embed: (B, T, d)。返回 (B, T, d)。"""
        h = self.eh_proj(torch.cat([self.hnorm(prev_hidden), self.enorm(next_embed)], dim=-1))
        out = self.block(h, **block_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return self.shared_head_norm(out)


class MultiTokenPredictor(nn.Module):
    """管理 n 层 MTP，并提供训练损失与推理草稿两个入口。

    Parameters
    ----------
    num_layers : n ≥ 0；0 表示关闭 MTP
    hidden_size : d_h
    blocks : 用于构建每个 MTP 深度的 block 工厂（callable → nn.Module）

    Notes
    -----
    第 k 个 MTP 模块的输入是「深度 k-1 的隐状态」与「目标位置的 embedding」，
    两者 concat 后投影回 d_h —— 这就是所谓 "chain of representations"。
    """

    def __init__(self, hidden_size: int, blocks: nn.ModuleList, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.heads = nn.ModuleList([
            AuroraMTPHead(hidden_size, blk, rms_norm_eps) for blk in blocks
        ])
        self.num_layers = len(self.heads)

    # ------------------------------------------------------------------
    def forward(
        self, trunk_hidden: torch.Tensor, embed_weight: Callable = None,  # type: ignore
        input_ids: Optional[torch.Tensor] = None,
        embed_fn=None,
        **block_kwargs,
    ):
        """自回归地跑 n 层 MTP。

        Parameters
        ----------
        trunk_hidden : (B, T, d)   主干最后一层隐状态
        embed_fn : callable(ids) → embedding，通常是 model.get_input_embeddings
        input_ids : (B, T)         当前序列；第 k 步需用 input_ids[:, k:] 对齐

        Returns
        -------
        hidden_states : List[Tensor]，第 k 项为深度 k 的隐状态，形状 (B, T-k, d)
        """
        assert embed_fn is not None and input_ids is not None, \
            "MTP 需要 embed_fn 与 input_ids 以构造下一位置的 embedding"

        B, T, d = trunk_hidden.shape
        hiddens = []
        prev = trunk_hidden
        for k, head in enumerate(self.heads, start=1):
            if T - k <= 0:
                break
            prev_slice = prev[:, : T - k, :]
            next_ids = input_ids[:, k:]
            next_embed = embed_fn(next_ids)
            cur = head(prev_slice, next_embed, **block_kwargs)
            hiddens.append(cur)
            prev = cur
        return hiddens

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}, hidden={self.hidden_size}"
