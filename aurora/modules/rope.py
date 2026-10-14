# -*- coding: utf-8 -*-
"""
旋转位置编码 RoPE 及其长上下文扩展。

RoPE 的核心洞察是把位置信息编码成**旋转**，而不是加法：

    <f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m - n)

即注意力分数只依赖**相对位置**。这是 2024 年后所有旗舰模型的默认姿式。

数学定义（注意 torch 惯例复数定义顺序）

    对相邻两半 (x⁽¹⁾, x⁽²⁾)，旋转角为 mθ_i：
        x'⁽¹⁾ = x⁽¹⁾·cos(mθ_i) - x⁽²⁾·sin(mθ_i)
        x'⁽²⁾ = x⁽²⁾·cos(mθ_i) + x⁽¹⁾·sin(mθ_i)

    写成 W_m x，其中 W_m 是分块旋转矩阵，满足 W_mᵀ W_n = W_{n-m}（旋转群同态）。

    ⟹ qᵀ_m k_n = (W_m q)ᵀ (W_n k) = qᵀ W_mᵀ W_n k = qᵀ W_{n-m} k     ✓ 相对位置

Invariant（可验证不变量）
------------------------
1. 正交性:     Wᵀ W = I             —— rotate 后 ‖x‖ 不变
2. 相对性:     score(m,n) 只依赖 n-m —— 平移不变
3. 逆等变性:   RoPE(x, m+n) = RoPE(RoPE(x,m), n)

长上下文扩展三种解析方法：
    linear   —— θ' = θ / s，粗暴缩放，短距离分辨率受损
    ntk      —— 频域 NTK 插值补偿高频
    yarn     —— YaRN：分段 ramp + attention temperature 修正（Kimi K2 用它到 128K/256K）
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


__all__ = [
    "RotaryEmbedding", "YarnRotaryEmbedding",
    "rotate_half", "rotate_every_two", "apply_rotary_pos_emb",
    "yarn_find_correction_range", "yarn_get_mscale", "yarn_linear_ramp_mask",
]


# ---------------------------------------------------------------------------
# 基础旋转
# ---------------------------------------------------------------------------
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """交错式旋转 (x1,x2,x3,x4) → (-x2,x1,-x4,x3)。"""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """半分式旋转：前半 = -后半，后半 = 原前半。

    这是主流实现（LLaMA/DeepSeek）使用的布局。
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
            interleaved: bool = False) -> torch.Tensor:
    rot = rotate_every_two if interleaved else rotate_half
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return (x * cos) + (rot(x) * sin)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None, interleaved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把预计算的 cos/sin 施加到 q,k 上。

    Parameters
    ----------
    q, k : Tensor, shape (..., seq, head_dim)
        必须 head_dim 为偶数。
    cos, sin : Tensor, shape (..., seq, head_dim) 或 (seq, head_dim)
        由 RotaryEmbedding 预先取好，可含 batch/head 广播维。
    position_ids : 可选，用于从 cos/sin 的第一维 gather（便于推理时只算用到的位置）。
    """
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(-2) if cos.dim() >= 2 else cos.squeeze(0)[position_ids]
        sin = sin[position_ids].unsqueeze(-2) if sin.dim() >= 2 else sin.squeeze(0)[position_ids]
    return _rotate(q, cos, sin, interleaved), _rotate(k, cos, sin, interleaved)


def _build_inv_freq(dim: int, base: float, device=None,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """θ_i = base^(-2i/dim), i = 0 .. dim/2-1。

    这是一条几何级数，从 1 衰减到 base^{-(dim-2)/dim}。
    高频（小 i）编码近邻，低频（大 i）编码远距离 —— NTK/YaRN 的干预正是针对这条谱。
    """
    half = dim // 2
    if dim % 2 != 0:
        raise ValueError(f"RoPE 要求 dim 为偶数，收到 {dim}")
    exponent = torch.arange(0, half, dtype=torch.float64, device=device)
    inv_freq = (1.0 / (base ** (exponent / half)))
    return inv_freq.to(dtype)


def _compute_cos_sin(inv_freq: torch.Tensor, seq_len: int, device=None,
                     dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """外积得到 (seq_len, dim) 的角度表，再 cat 成完整 head_dim。"""
    inv_freq = inv_freq.to(device=device)
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)                     # (S, D/2)
    emb = torch.cat((freqs, freqs), dim=-1)              # (S, D)  半分式布局
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


# ---------------------------------------------------------------------------
# 标准 RoPE
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """标准（可扩展 NTK-aware）旋转位置编码。

    Parameters
    ----------
    dim : int
        单个注意力头的 RoPE 部分维度（DeepSeek 记为 d_r）。
    base : float
        θ_base，通常 1e4（LLaMA/Qwen）或 5e5 / 1e6（长上下文模型）。
    scaling_type : str
        none | linear | dynamic-ntk | yarn
    scaling_factor : float
        上下文扩展倍数 s。

    备注：为保持与训练/推理一致的确定性，本类缓存 cos/sin 表，并在超过缓存长度
    时自动扩容（而非每次重建——后者在 batch>1 时会造成可观的 CPU/GPU 开销）。
    """

    def __init__(self, dim: int, base: float = 10000.0, max_position: int = 8192,
                 scaling_type: str = "none", scaling_factor: float = 1.0,
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim = int(dim)
        self.base = float(base)
        self.max_position = int(max_position)
        self.scaling_type = scaling_type
        self.scaling_factor = float(scaling_factor)

        inv_freq = self._prep_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = _compute_cos_sin(self.inv_freq, max_position, device, dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    # ------------------------------------------------------------------
    def _apply_scaling(self, device) -> torch.Tensor:
        """按 scaling_type 修正 inv_freq（NTK-aware / linear）。"""
        base = self.base
        s = self.scaling_factor
        if self.scaling_type in (None, "none") or s <= 1.0:
            return _build_inv_freq(self.dim, base, device)

        if self.scaling_type == "linear":
            # 位置压缩：等价于 rope base 缩放
            return _build_inv_freq(self.dim, base * s, device)

        if self.scaling_type == "dynamic-ntk":
            # NTK-aware：base' = base * s ^ (dim/(dim-2))
            d = self.dim
            eff = base * (s ** (d / max(1, d - 2)))
            return _build_inv_freq(d, eff, device)

        raise ValueError(
            f"RoPE scaling_type={self.scaling_type!r} 不支持；"
            f"请在 none/linear/dynamic-ntk 中选，或用 YarnRotaryEmbedding。"
        )

    def _prep_inv_freq(self, device) -> torch.Tensor:
        return self._apply_scaling(device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extend_cache(self, seq_len: int, device) -> None:
        if seq_len <= self.cos_cached.shape[0]:
            return
        new_len = max(seq_len, int(self.cos_cached.shape[0] * 1.5) + 64)
        inv_freq = self.inv_freq.to(device=device)
        cos, sin = _compute_cos_sin(
            inv_freq, new_len, device, self.cos_cached.dtype)
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(self, seq_len: int, device=None,
                dtype: torch.dtype = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 shape (seq_len, dim) 的 (cos, sin)。"""
        dev = device if device is not None else self.cos_cached.device
        self._extend_cache(seq_len, dev)
        cos = self.cos_cached[:seq_len].to(device=dev)
        sin = self.sin_cached[:seq_len].to(device=dev)
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, base={self.base}, max_position={self.max_position}, "
                f"scaling={self.scaling_type}x{self.scaling_factor}")


# ---------------------------------------------------------------------------
# YaRN
# ---------------------------------------------------------------------------
def yarn_find_correction_dim(num_rotations: float, dim: int, base: float = 10000.0,
                             max_position: int = 2048) -> float:
    """求解使波长等于 max_position / num_rotations 的那个 2i 索引。

    推导：波长 λ_i = 2π / θ_i = 2π · base^(2i/dim)。
    令 λ_i = L / r  ⟹ base^(2i/dim) = L / (2π r)
    ⟹ 2i = dim · log(L / (2π r)) / log(base)
    """
    return (dim * math.log(max_position / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot: float, high_rot: float, dim: int,
                               base: float = 10000.0,
                               max_position: int = 2048) -> Tuple[int, int]:
    """返回需要进行 NTK 插值的 inv_freq 索引区间 [low, high]。"""
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position))
    return max(low, 0), min(high, dim // 2 - 1)


def yarn_linear_ramp_mask(low: float, high: float, dim: int,
                          dtype: torch.dtype = torch.float32,
                          device=None) -> torch.Tensor:
    """从 low 到 high 的线性斜坡 mask（作用在 2i 轴上）。

    YaRN 的关键之一：不要在做插值的区域和非插值区域之间造突变，否则会在长上下文
    上产生尖锐的位置伪影。
    """
    if low == high:
        high += 0.001
    if dim % 2 != 0:
        raise ValueError(f"YaRN 要求偶数 dim，收到 {dim}")
    half = dim // 2
    idx = torch.arange(half, dtype=dtype, device=device)
    linear_func = (idx - low) / (high - low)
    return linear_func.clamp_(0.0, 1.0)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """attention temperature 修正系数。

    YaRN 论文的最后一个部件：上下文变长后 softmax 的分布会变平，
    用 mscale 补回来：

        attn_scale = (0.1 · ln(s) + 1)²     当 mscale == 1

    这是一个经验修正，但在 128K+ 上确实起作用（Kimi K2 在退火阶段用它）。
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YarnRotaryEmbedding(nn.Module):
    """Yet another RoPE extensioN。

    相比粗暴的 linear interpolation，YaRN 做三件事：
      1. **NTK-by-parts**：对每个频率波长度身定制 —— 高频（近邻）不插值保精度，
         低频（远程）做插值获得外推能力，中间用斜坡过渡。
      2. **attention scaling**：logits 乘 mscale 补偿 softmax 变平。
      3. 可选的注意力因子按层微调（correction_range_attn_factors）。

    这就是 Kimi K2 把上下文从 32K 退火到 128K 再推到 256K 所用的方法。
    """

    def __init__(
        self,
        dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        correction_range_attn_factors=(1.0, 1.0),
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = int(dim)
        self.max_position = int(max_position)
        self.base = float(base)
        self.scaling_factor = float(scaling_factor)
        self.original_max_position = int(original_max_position)
        self.beta_fast = int(beta_fast)
        self.beta_slow = int(beta_slow)
        self.mscale = float(mscale)
        self.mscale_all_dim = float(mscale_all_dim)
        self.cr_low, self.cr_high = correction_range_attn_factors

        inv_freq, mscale_tensor = self._compute_inv_freq(device, dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("mscale", mscale_tensor, persistent=False)

        cos, sin = _compute_cos_sin(self.inv_freq, max_position, device, dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    # ------------------------------------------------------------------
    def _compute_inv_freq(self, device, dtype):
        s = self.scaling_factor
        base = self.base
        dim = self.dim
        inv_freq_extrapolation = _build_inv_freq(dim, base, device, torch.float64)
        inv_freq_interpolation = 1.0 / (s * (base ** (torch.arange(
            0, dim, 2, dtype=torch.float64, device=device) / dim)))

        # beta_fast/beta_slow 定义了"哪些波长算快/算慢"
        low, high = yarn_find_correction_range(
            self.beta_fast, self.beta_slow, dim, base, self.original_max_position)
        low, high = int(low * self.cr_low), int(high * self.cr_high)

        ramp = yarn_linear_ramp_mask(low, high, dim, torch.float64, device)
        inv_freq = inv_freq_interpolation * (1 - ramp) + inv_freq_extrapolation * ramp

        # mscale: 分 Partial / All-dim 两种用法
        mscale = yarn_get_mscale(s, self.mscale)
        if self.mscale_all_dim > 0:
            mscale_all = yarn_get_mscale(s, self.mscale_all_dim)
            scale_ratio = base / (s ** (dim / (dim - 2))) / base  # 保守形式
            mscale = float(mscale * mscale_all) / max(1e-8, abs(scale_ratio))

        return inv_freq.to(torch.float32), torch.tensor(float(mscale), dtype=torch.float32)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extend_cache(self, seq_len: int, device) -> None:
        if seq_len <= self.cos_cached.shape[0]:
            return
        new_len = max(seq_len, int(self.cos_cached.shape[0] * 1.5) + 64)
        cos, sin = _compute_cos_sin(
            self.inv_freq, new_len, device, self.cos_cached.dtype)
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(self, seq_len: int, device=None,
                dtype: torch.dtype = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = device if device is not None else self.cos_cached.device
        self._extend_cache(seq_len, dev)
        cos = self.cos_cached[:seq_len].to(device=dev)
        sin = self.sin_cached[:seq_len].to(device=dev)
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin

    def get_mscale(self) -> float:
        return float(self.mscale.item())

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, factor={self.scaling_factor}, "
                f"orig_max={self.original_max_position}, base={self.base}, "
                f"beta=[{self.beta_fast},{self.beta_slow}]")
