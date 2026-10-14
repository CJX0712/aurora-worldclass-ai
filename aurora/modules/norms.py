# -*- coding: utf-8 -*-
"""
归一化算子。

为什么不再用 LayerNorm？
    2019 年后，几乎所有前沿模型（LLaMA / Qwen / DeepSeek / Gemma / Mistral）
    都换成了 RMSNorm。它不是"更便宜的 LayerNorm"，而是**更正确的归纳偏置**：
    它归一化的尺度，但不修改方向、也不施加分量间的耦合偏置。

        RMSNorm:   y = x / sqrt(mean(x^2) + eps) * g

    数学上，RMSNorm 保留了解的天然正交性：LayerNorm 的减均值操作会把向量投影
    到超平面上，损失一维信息；RMSNorm 只做纯缩放。

Tokenizer 之外，我们在两处额外使用带 gating 的 RMSNorm：
    - QK-RMSNorm:   attention 前对 q/k 施加，抑制 logit 爆炸
                    （2025 年后成为旗舰标配，见 Gemma 2/3、Qwen3）
    - GatedRMSNorm: GDN 输出前，配合 Mamba 式 gate 使用
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["RMSNorm", "QKRMSNorm", "GatedRMSNorm", "l2_normalize"]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

        RMS(x) = sqrt( (1/d) Σ x_i² + eps )
        y      = x / RMS(x) ⊙ g

    Parameters
    ----------
    hidden_size : int
        最后一维大小 d。
    eps : float
        数值稳定项。注意：我们在 sqrt **内部**加 eps（与原始定义一致），
        这与某些实现的 sqrt(mean + eps) 完全等价；不要两个地方都加。

    Invariant
    ---------
    对 g ≡ 1 的输入，输出的 RMS 恒为 sqrt(1 - eps/RMS²) ≈ 1。
    即 ‖f(x)‖₂ ≈ sqrt(d)。测试中对 g=1 的输出断言 mean(x²)=1+O(eps)。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 elementwise_affine: bool = True,
                 bias: bool = False):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.float()                      # 一律用 fp32 做归一化统计
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight.float()
        if self.bias is not None:
            x = x + self.bias.float()
        return x.to(dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


class QKRMSNorm(nn.Module):
    """作用在 head 维度上的 RMSNorm。

    用于 attention 之前稳定 q/k 的内积尺度：

        q̂ = RMSNorm(q) ,  k̂ = RMSNorm(k)
        score = q̂·k̂ / sqrt(d)

    这一步为什么关键？因为它把 attention logit 的尺度与投影矩阵 W 的谱范数
    解耦：‖q̂·k̂‖ ≤ ‖q̂‖·‖k̂‖ = sqrt(d)² = d（Cauchy-Schwarz）。这为后续
    MuonClip 的 QK-Clip 提供了有意义的目标量。

    Per-head 权重（num_heads 给定时）
    ---------------------------------
    当 ``num_heads`` 给出，affine 权重形状为 ``(num_heads, head_dim)``，对
    每个 head 独立施加。这是 **QK-Clip 逐 head 裁剪** 能够精确生效的前提：
    QK-Clip 需要把第 h 个 head 的 q/k 精确缩放 √γ_h，但若 QK-Norm 的权重是
    head 间共享的，则无法逐 head 区分。逐 head 权重让「缩放 norm 的 affine」
    这一操作（它发生在归一化**之后**、不会被 RMS 除法抵消）天然支持逐 head。
    （共享权重是 ``(head_dim,)``，逐 head 是 ``(num_heads, head_dim)``，二者
    数学等价，只是参数粒度不同。）

    Invariant
    ---------
    ‖f(x)‖₂ = sqrt(d_head)（当 weight ≡ 1 时），与输入尺度无关。
    """

    def __init__(self, head_dim: int, eps: float = 1e-6, num_heads: Optional[int] = None):
        super().__init__()
        self.head_dim = int(head_dim)
        self.eps = float(eps)
        self.num_heads = int(num_heads) if num_heads is not None else None
        if num_heads is not None:
            self.weight = nn.Parameter(torch.ones(num_heads, head_dim))
        else:
            self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        y = xf * torch.rsqrt(var + self.eps)
        if self.num_heads is not None:
            # 逐 head affine：weight (nh, d) → 广播到 (..., nh, d)
            shape = (1,) * (x.dim() - 2) + (self.num_heads, self.head_dim)
            w = self.weight.float().reshape(shape)
            y = y * w
        else:
            y = y * self.weight.float()
        return y.to(dtype)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, eps={self.eps}" + (
            f", num_heads={self.num_heads}" if self.num_heads is not None else "")


class GatedRMSNorm(nn.Module):
    """带 gate 的 RMSNorm，形如 y = RMSNorm(x) ⊙ silu(g)。

    Gated DeltaNet / Mamba-2 的输出都依赖这个结构。它把"要输出多少"的决策权
    交给一个单独学习的门，而不是让 layer norm 的 affine 参数硬扛。

    注意：此处 RMS 只在 x 上做，gate 不参与归一化，这与 fla 的
    `FusedRMSNormGated` 语义一致。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        y = xf * torch.rsqrt(var + self.eps) * self.weight.float()
        y = y * F.silu(gate.float())
        return y.to(dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 归一化到单位球面上。

        x̂ = x / max(‖x‖₂, eps)

    Gated DeltaNet 要求 ‖k‖=‖q‖=1，因为 delta rule 的状态更新
    S ← S(I - β k kᵀ) 只有在 k 为单位向量时才是**严格 Householder 反射**，
    从而保证所有状态转移矩阵的特征值模 ≤ 1（动力系统不发散）。
    """
    return x / x.norm(2, dim=dim, keepdim=True).clamp_min(eps)
