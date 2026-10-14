# -*- coding: utf-8 -*-
"""
MuonClip 优化器包。

把 2024-2026 年间被万亿参数级模型逐一验证过的优化器收敛进来：

* Muon        《Muon: An optimizer for matrix-based language modeling》
              (Keller Jordan et al., 2024-12)。对 2D 权重矩阵做 Nesterov 动量 +
              牛顿-舒尔茨(Newton-Schulz)正交化，把更新方向推到「谱范数最近的正交矩阵」，
              让矩阵每个方向都拿到同等步长。
* MuonClip    《Kimi K2: Open Agentic Intelligence》(Moonshot AI, 2025-07)。
              在 Muon 上叠加 (a) 解耦权重衰减 (b) 一致的 RMS 缩放（使 Muon 单参数
              更新 RMS 与 Adam 对齐，可直接迁移 Adam 超参）(c) QK-Clip——
              每当注意力 logit 超过阈值 τ 就回缩 Q/K 投影权重。Kimi K2 用它在
              15.5T token 上做到**零 loss spike** 预训练。

作者: 晨星
许可: Apache-2.0
"""

from aurora.optim.muon import (
    Muon,
    MuonClip,
    apply_qk_clip,
    muon_clip_split_params,
    newton_schulz_5,
)

__all__ = [
    "Muon",
    "MuonClip",
    "apply_qk_clip",
    "muon_clip_split_params",
    "newton_schulz_5",
]
