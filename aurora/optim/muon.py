# -*- coding: utf-8 -*-
"""
Muon / MuonClip 优化器（零依赖，纯 PyTorch）。

来源
----
* Muon        Keller Jordan et al., 2024-12 ——
              https://github.com/KellerJordan/modded-nanogpt
              对 2D 隐藏层权重矩阵做「矩量正交化」：SGD+Nesterov 动量后，把动量矩阵
              M 替换成其在 Frobenius 范数意义下最近的正交矩阵 UVᵀ（极分解因子），
              即把所有奇异值推向 1，让每个方向拿到同等步长。相比 AdamW 提速 ~1.35×
              （NanoGPT），compute-optimal 下 ~2× FLOPs 收益（Moonlight 5.7T token）。
* MuonClip    Moonshot AI《Kimi K2: Open Agentic Intelligence》, 2025-07。
              在 Muon 上叠加：解耦权重衰减、一致的 RMS 缩放（单参数更新 RMS 对齐
              Adam，可直接迁移 Adam 超参）、以及 QK-Clip（logit 超限就回缩 Q/K 权重）。
              Kimi K2（1T 参数，15.5T token）全程零 loss spike。

数学（Kimi K2 Alg.1，对每个 2D 权重 W ∈ ℝ^{n×m}）
------------------------------------------------
    M_t   = μ · M_{t-1} + G_t                          ▷ 动量（M_0 = 0；G_t 为梯度）
    O_t   = NewtonSchulz(M_t) · 0.2 · √(max(n, m))     ▷ 正交化 + 一致 RMS 缩放
    W_t   = W_{t-1} − η · (O_t + λ · W_{t-1})          ▷ 学习率 η，解耦权重衰减 λ
随后对每个注意力层的每个 head 做 QK-Clip（见各 mixer 模块的 qk_clip_step）。

Newton-Schulz 五阶迭代
----------------------
    系数 a=3.4445, b=−4.7750, c=2.0315（刻意调成在 0 附近斜率最大，bF16 下也稳定）
    X = G / (‖G‖_F + ε)
    A = X Xᵀ
    repeat 5×:   A = a·I + b·A + c·A²
    return G · A        ▷ ≈ G (GᵀG)^{−½} = 极分解正交因子 UVᵀ

RMS 缩放两种口径
----------------
* 'kimi'   : 0.2 · √(max(n, m))                         —— Kimi K2 默认，单参数更新 RMS 对齐 Adam
* 'jordan' : √(max(1, n/m))                             —— Keller Jordan 原始口径

哪些参数交给 Muon
-----------------
Muon 只对 **2D 隐藏层权重矩阵** 有意义；1D 参数（bias、norm scale）、embedding、
以及 LM head（输出层）交给 AdamW（角色/经验理由，Muon 作者原话）。
`muon_clip_split_params(model)` 自动按此切分。

作者: 晨星
许可: Apache-2.0
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


def newton_schulz_5(G: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """把矩阵推到「谱范数最近的正交矩阵」（极分解正交因子 U Vᵀ），奇异值全部 → 1。

    这是 Muon 的核心：对动量矩阵做正交化，让每个方向拿到同等步长。

    Parameters
    ----------
    G : (n, m) 或更高维（会自动压扁成 2D）
    steps : 仅作历史兼容保留（本实现用 SVD 精确求极分解正交因子，
            不依赖迭代步数；Newton–Schulz 迭代正是此操作的近似）。
    eps   : 数值稳定项（谱范数归一化用，避免除零）。

    Returns
    -------
    (n, m) 正交化后的更新方向，奇异值全部 = 1，与输入同形。

    Method（精确极分解正交因子）
    ---------------------------
    Muon 的数学目标是把动量矩阵 M 替换成其 **Frobenius 范数最近的正交矩阵**，
    即极分解的**正交因子** U（使 M ≈ U·P，P 半正定）——等价于把 M 的全部奇异值
    推向 1。Keller Jordan 的 Newton–Schulz 五阶迭代是这一操作的高效近似，但存在
    **根本性收敛缺陷**：Newton–Schulz 对极小奇异值方向的收敛是**线性的**
    （每步误差约乘 (1−√λ_min)），当动量矩阵病态（随机矩阵常如此）时，float32 下
    小奇异值方向长时间停在 0.25 量级无法收敛到 1，导致「奇异值→1」不变量失效。

    本实现直接通过 **SVD 精确计算极分解正交因子**，无任何迭代近似：

        G = U Σ Vᵀ   ⟹   正交因子 = U Vᵀ ∈ ℝ^{n×m}，其奇异值全部严格 = 1

    这是 Muon 理论目标的**精确**实现：对任意随机 / 病态矩阵都无条件成立，奇异值
    全部 → 1（浮点精度内），正好满足本项目「奇异值→1」硬不变量。在 CPU 端的
    Aurora 体量与训练规模下，SVD 的开销远小于 Muon 整体预算，正确性优先于
    那一点迭代加速；大规模 GPU 训练时可换成 Newton–Schulz 近似以省算力。
    """
    flat = G.reshape(-1, G.shape[-1]).double()        # 压扁成 2D，float64 保证精度
    sigma = torch.linalg.matrix_norm(flat, ord=2)      # 谱范数（仅用于稳定，可选）
    U, S, Vh = torch.linalg.svd(flat / (sigma + eps), full_matrices=False)
    orth = U @ Vh                                      # 正交因子，奇异值全部 = 1
    return orth.to(G.dtype).reshape(G.shape)


def _rms_scale(shape: Tuple[int, int], mode: str, muon_scale: float) -> float:
    n, m = shape
    if mode == "kimi":
        return muon_scale * math.sqrt(max(n, m))
    if mode == "jordan":
        return math.sqrt(max(1.0, n / m))
    raise ValueError(f"未知 RMS 缩放口径: {mode!r}（可选 'kimi' / 'jordan'）")


class Muon(Optimizer):
    """Muon：2D 隐藏层权重矩阵的正交化优化器。

    仅用于 2D 参数；其余参数请用 AdamW（见 `muon_clip_split_params`）。

    Parameters
    ----------
    params     : 2D 权重张量的迭代器
    lr         : 学习率 η（默认 0.02，远大于 AdamW 的 3e-4，因为正交化更新幅度固定）
    momentum   : 动量系数 μ（默认 0.95）
    nesterov   : 是否 Nesterov（默认 True，送正交化的是 G + μ·M）
    weight_decay : 解耦权重衰减 λ（默认 0.0；Kimi K2 默认 ~0.01）
    ns_steps   : Newton-Schulz 迭代次数（默认 5）
    rms_match  : RMS 缩放口径 'kimi'（默认）或 'jordan'
    muon_scale : Kimi 口径下的常数（默认 0.2）
    """

    def __init__(self, params: Iterable[torch.Tensor], lr: float = 0.02,
                 momentum: float = 0.95, nesterov: bool = True,
                 weight_decay: float = 0.0, ns_steps: int = 5,
                 rms_match: str = "kimi", muon_scale: float = 0.2):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_steps=ns_steps,
                        rms_match=rms_match, muon_scale=muon_scale)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            rms_match = group["rms_match"]
            muon_scale = group["muon_scale"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError(
                        "Muon 只接受 2D 参数；请把 1D 参数（bias/norm）与 "
                        "embedding / LM head 交给 AdamW。")
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                # 动量：M_t = μ·M + G
                buf.mul_(mom).add_(g)
                # Nesterov 混合：送正交化的是 G + μ·M
                u = g.add(buf, alpha=mom) if nesterov else buf
                # 正交化 + 一致 RMS 缩放
                u = newton_schulz_5(u, steps=ns_steps)
                scale = _rms_scale((u.shape[0], u.shape[1]), rms_match, muon_scale)
                u = u * scale
                # 解耦权重衰减 + 更新：W ← W·(1 − ηλ) − η·O
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(u, alpha=-lr)
        return loss


class MuonClip(Muon):
    """MuonClip：在 Muon 之上封装 QK-Clip 的"事后校正"调用。

    QK-Clip 不是 optimizer 的内部状态，而是「每个训练步、Muon 更新之后」对注意力
    模块权重做的裁剪。这里提供一个便捷入口：`step()` 照常更新权重，调用方随后用
    `apply_qk_clip(modules, threshold)` 或本类方法 `qk_clip(modules, threshold)`
    触发裁剪。两种用法等价，后者更紧凑。

    典型训练循环：

        for x in loader:
            logits = model(x)
            loss = ...
            loss.backward()
            muon_clip.step()                       # 更新全部参数
            apply_qk_clip(attn_modules, tau=100)  # 仅裁剪 Q/K 投影
    """

    @staticmethod
    @torch.no_grad()
    def qk_clip(attn_modules: Iterable, threshold: float = 100.0) -> List[torch.Tensor]:
        """对每个注意力模块调用其 ``qk_clip_step``（MLA/SWA 均实现）。

        Returns
        -------
        list[Tensor]：各模块实际施加的逐 head/group γ（用于日志）。
        """
        return apply_qk_clip(attn_modules, threshold=threshold)


@torch.no_grad()
def apply_qk_clip(attn_modules: Iterable, threshold: float = 100.0) -> List[torch.Tensor]:
    """对所有注意力模块触发 QK-Clip（不改变当前步前向/反向，纯事后校正）。

    Returns
    -------
    list[Tensor]：各模块返回的逐 head/group γ。
    """
    out = []
    for m in attn_modules:
        fn = getattr(m, "qk_clip_step", None)
        if callable(fn):
            out.append(fn(threshold))
    return out


def muon_clip_split_params(model: torch.nn.Module) -> Tuple[List[dict], List[torch.Tensor]]:
    """按 MuonClip 的角色划分规则，把模型参数切成两组。

    规则（Muon 作者 / Kimi K2）：
    * 2D 隐藏层权重矩阵（Linear 的 weight，非 embedding / 非 LM head） → MuonClip
    * 其余（1D bias、RMSNorm 的 weight、embedding、LM head、MTP 的共享 embed/head
      等"角色特殊"的 2D 参数）                           → AdamW

    Returns
    -------
    (muon_groups, adam_params)
        muon_groups : 可直接喂给 ``MuonClip(params=...)`` 的参数组列表（含单组）
        adam_params  : 可直接喂给 ``torch.optim.AdamW(params=...)`` 的张量列表
    """
    muon_params: List[torch.Tensor] = []
    adam_params: List[torch.Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim != 2:
            adam_params.append(p)
            continue
        low = name.lower()
        if any(k in low for k in ("embed", "lm_head", "tok_embeddings",
                                  "wte", "out_proj_lm", "head")):
            adam_params.append(p)
            continue
        muon_params.append(p)
    muon_groups = [{"params": muon_params}] if muon_params else []
    return muon_groups, adam_params
