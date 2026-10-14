# -*- coding: utf-8 -*-
"""
Aurora 模型内核。

本层实现的是**数学**，不是胶水。每个算子都给出其定义式、数值稳定理由、
以及一条可被独立交叉验证的不变量（invariant）。不变量写在 docstring 的
`Invariant` 段落中，并在 `tests/test_invariants.py` 中被逐一断言。

为什么如此强调不变量？因为一个算法"看起来对"和"真的对"之间，隔着一堆
可以互相抵消的符号错误。推导 ⟹ 实现 ⟹ 验证，三者缺一不可。

本模块包含：
    norms.py  RMSNorm / QK-RMSNorm / GatedRMSNorm
    rope.py   RoPE + YaRN + NTK-aware scaling
    mla.py    Multi-head Latent Attention（低秩 KV 联合压缩 + 吸收式推理）
    gdn.py    Gated DeltaNet（chunkwise 并行 + 递推黄金参照）
    moe.py    DeepSeekMoE + Quantile Balancing
    mtp.py    Multi-Token Prediction 头
    block.py  统一 Transformer Block（1 个 Block 覆盖 M/D/S 三种 mixer）
"""

from aurora.modules.norms import RMSNorm, QKRMSNorm, GatedRMSNorm, l2_normalize
from aurora.modules.rope import (
    RotaryEmbedding, YarnRotaryEmbedding, rotate_half, apply_rotary_pos_emb,
    yarn_find_correction_range, yarn_get_mscale, yarn_linear_ramp_mask,
)
from aurora.modules.mla import MultiHeadLatentAttention
from aurora.modules.gdn import (
    GatedDeltaNet, recurrent_gated_delta_rule, chunk_gated_delta_rule,
    causal_short_conv, l2norm_fwd,
)
from aurora.modules.moe import DeepSeekMoE, QuantileRouter, build_expert_mlp
from aurora.modules.mtp import MultiTokenPredictor, AuroraMTPHead

__all__ = [
    "RMSNorm", "QKRMSNorm", "GatedRMSNorm", "l2_normalize",
    "RotaryEmbedding", "YarnRotaryEmbedding", "rotate_half", "apply_rotary_pos_emb",
    "yarn_find_correction_range", "yarn_get_mscale", "yarn_linear_ramp_mask",
    "MultiHeadLatentAttention",
    "GatedDeltaNet", "recurrent_gated_delta_rule", "chunk_gated_delta_rule",
    "causal_short_conv", "l2norm_fwd",
    "DeepSeekMoE", "QuantileRouter", "build_expert_mlp",
    "MultiTokenPredictor", "AuroraMTPHead",
]
