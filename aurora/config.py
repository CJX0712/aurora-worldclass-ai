# -*- coding: utf-8 -*-
"""
Aurora 配置。

设计哲学：一个配置对象应当足以**重建并且专利权人可以一目了然地读懂**整个模型的
结构。因此这里没有魔法字符串，没有隐式推断，所有维度显式可查。

每一层的 token mixer 由 `mixer_pattern` 决定，允许任意交错。用一个字符串就能表达
当今所有主流的混合配比：

    "MMMM"        纯 MLA 全局注意力        —— DeepSeek-V3, Kimi K2
    "DDDD"        纯 Gated DeltaNet        —— 纯线性 recurrent
    "DSDSDSDSM"   Gemma 3/4 式 5:1 局部:全局
    "DDDDDDDM"    MiniMax-M1 式 7:1 lightning
    "DDDSDDDM"    Qwen3.5 式 ~75% GDN + 25% softmax
    "DSMDSMDSM"   混合 softmax / 复发 / 全局

这不是"又一堆开关"，这是把"混合注意力配比"这一 2026 年的核心设计维度，提升为
一等公民（first-class hyper-parameter）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Token mixer 类型表
# ---------------------------------------------------------------------------
#   M = MLA       — Multi-head Latent Attention，全局，KV 低秩压缩
#   D = GDN       — Gated DeltaNet，线性注意力 / recurrent，O(N) 状态
#   S = SWA       — Sliding-Window Attention with GQA，局部 softmax
MIXER_KINDS: Dict[str, str] = {
    "M": "mla",
    "D": "gdn",
    "S": "swa",
}

_KIND_TO_CODE: Dict[str, str] = {v: k for k, v in MIXER_KINDS.items()}


def build_mixer_plan(pattern: str, num_layers: int) -> List[str]:
    """把重复的模式串展开为逐层的 mixer 类型列表。

    支持两种写法：
      - 重复模式:  "DSM" + num_layers=12  -> ["D","S","M", "D","S","M", ...]
      - 逐层显式:  长度恰好等于 num_layers 时直接使用

    Raises
    ------
    ValueError
        模式非法或长度不匹配。
    """
    pattern = pattern.strip().upper()
    unknown = set(pattern) - set(MIXER_KINDS)
    if unknown:
        raise ValueError(
            f"未知的 mixer 类型 {sorted(unknown)}；可选: {list(MIXER_KINDS)}"
        )

    if len(pattern) == num_layers:
        return list(pattern)
    if num_layers % len(pattern) != 0:
        raise ValueError(
            f"num_layers={num_layers} 不能被 pattern 长度 {len(pattern)} 整除；"
            f"请给出长度恰为 {num_layers} 的逐层显式 pattern。"
        )
    repeat = num_layers // len(pattern)
    return list(pattern * repeat)


# ---------------------------------------------------------------------------
# 主配置
# ---------------------------------------------------------------------------
@dataclass
class AuroraConfig:
    """Aurora 模型的全部超参数。

    命名刻意贴近 DeepSeek-V3 / Kimi K2 / Qwen3.5 的配置项，便于对照阅读。
    """

    # ---- 主干 ----
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    dtype: str = "float32"

    # ---- Token mixer 编排（Aurora 的核心设计维度）----
    mixer_pattern: str = "DSM"
    sliding_window: int = 512

    # ---- MLA（Multi-head Latent Attention）----
    num_attention_heads: int = 12
    q_lora_rank: int = 384          # 0 表示不做 query 低秩压缩
    kv_lora_rank: int = 256         # d_c：KV 联合压缩维度
    qk_nope_head_dim: int = 128     # d_n：非 RoPE 部分
    qk_rope_head_dim: int = 64      # d_r：解耦 RoPE 部分
    v_head_dim: int = 128           # d_v
    attention_dropout: float = 0.0

    # ---- SWA / GQA ----
    num_key_value_heads: int = 4    # SWA 层的 KV 头数（GQA 分组）
    swa_head_dim: int = 128

    # ---- Gated DeltaNet（线性注意力）----
    gdn_num_heads: int = 12
    gdn_key_dim: int = 128          # 每个头的 k/q 维度
    gdn_value_dim: int = 128        # 每个头的 v 维度
    gdn_conv_size: int = 4          # 短卷积核宽度（务必 ≥2，质量影响显著）
    gdn_use_gate: bool = True       # 输出门控（Mamba 式）
    gdn_use_beta: bool = True
    gdn_allow_neg_eigval: bool = False
    gdn_norm_eps: float = 1e-6
    gdn_chunk_size: int = 64        # chunkwise 并行块大小

    # ---- Feed-Forward / MoE ----
    intermediate_size: int = 1536
    hidden_act: str = "silu"
    moe_intermediate_size: int = 512
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    moe_layer_freq: int = 1         # 每 N 层一个 MoE 层；1 = 全 MoE
    first_k_dense_replace: int = 0  # 前 k 层强制稠密 FFN
    routed_scaling_factor: float = 1.0
    topk_method: str = "greedy"     # greedy | group_limited_greedy | noaux_tc
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    scoring_func: str = "softmax"   # softmax | sigmoid
    # 负载均衡
    use_quantile_balancing: bool = True   # Kimi K3：按 router score 分位数分配
    quantile_lambda: float = 1.0          # bias 更新速率 / 也可作 aux loss 权重
    balance_tau: float = 0.01             # 目标不均衡度容忍区间
    use_aux_loss: bool = False            # 经典 GShard 式辅助损失（默认关闭）
    aux_loss_alpha: float = 0.0001

    # ---- Multi-Token Prediction（自带投机解码草稿器）----
    num_nextn_predict_layers: int = 1
    mtp_loss_factor: float = 0.3

    # ---- RoPE ----
    rope_theta: float = 10000.0
    rope_scaling_type: str = "none"  # none | linear | yarn | dynamic-ntk
    rope_scaling_factor: float = 1.0
    yarn_original_max_position: int = 4096
    yarn_beta_fast: int = 32
    yarn_beta_slow: int = 1
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 1.0
    yarn_correction_range_attn_factors: Tuple[float, float] = (1.0, 1.0)

    # ---- MuonClip ----
    use_muon_for_matrices: bool = True
    qk_clip_threshold: float = 100.0

    # ---- 推理 ----
    page_size: int = 16

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------
    @property
    def mixer_plan(self) -> List[str]:
        return build_mixer_plan(self.mixer_pattern, self.num_layers)

    @property
    def head_dim(self) -> int:
        """MLA 单个头的总维度 d_n + d_r。"""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def num_layers_by_kind(self) -> Dict[str, int]:
        plan = self.mixer_plan
        return {k: sum(1 for p in plan if p == k) for k in MIXER_KINDS}

    def is_moe_layer(self, layer_idx: int) -> bool:
        """该层的 FFN 是否为 MoE。"""
        if layer_idx < self.first_k_dense_replace:
            return False
        if self.n_routed_experts <= 0:
            return False
        return (layer_idx - self.first_k_dense_replace) % max(1, self.moe_layer_freq) == 0

    @property
    def torch_dtype(self):
        import torch
        return getattr(torch, self.dtype)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AuroraConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}  # type: ignore
        return cls(**known)

    def evolve(self, **kwargs) -> "AuroraConfig":
        return replace(self, **kwargs)

    # ------------------------------------------------------------------
    # 参数量估计（不需要实例化即可算，用于快速配方扫描）
    # ------------------------------------------------------------------
    def estimate_num_parameters(self) -> int:
        """解析地估计非嵌入、嵌入与 MoE 的参数量。用于配方扫描。"""
        h = self.hidden_size
        V = self.vocab_size
        L = self.num_layers
        per_head = self.num_attention_heads
        cnt = self.num_layers_by_kind

        p = 0

        # ---- MLA ----
        n_mla = cnt.get("M", 0)
        if n_mla:
            q_lora = self.q_lora_rank
            # down-proj: h -> q_lora ; up: q_lora -> H*(d_n + d_r)
            if q_lora:
                p += n_mla * (h * q_lora + q_lora * per_head * self.qk_head_dim)
            else:
                p += n_mla * (h * per_head * self.qk_head_dim)
            # kv down  (h -> d_c + d_r)  +  kv up (d_c -> H*(d_n + d_v))
            p += n_mla * (h * (self.kv_lora_rank + self.qk_rope_head_dim))
            p += n_mla * (self.kv_lora_rank * per_head *
                          (self.qk_nope_head_dim + self.v_head_dim))
            # out proj: H*d_v -> h
            p += n_mla * (per_head * self.v_head_dim * h)
            if q_lora:
                p += n_mla * self.qk_nope_head_dim  # q layernorm

        # ---- SWA ----
        n_swa = cnt.get("S", 0)
        if n_swa:
            hd = self.swa_head_dim
            p += n_swa * (h * per_head * hd)                      # q
            p += n_swa * (h * self.num_key_value_heads * hd)       # k
            p += n_swa * (h * self.num_key_value_heads * hd)       # v
            p += n_swa * (per_head * hd * h)                       # o

        # ---- GDN ----
        n_gdn = cnt.get("D", 0)
        if n_gdn:
            nh, kd, vd = self.gdn_num_heads, self.gdn_key_dim, self.gdn_value_dim
            p += n_gdn * h * nh * kd            # q
            p += n_gdn * h * nh * kd            # k
            p += n_gdn * h * nh * vd            # v
            p += n_gdn * h * nh * vd            # beta (scalar per head -> nh*1, 近似 vd)
            p += n_gdn * nh * vd * h            # out
            if self.gdn_use_gate:
                p += n_gdn * h * nh * vd        # gate
                p += n_gdn * h                  # 简化：alpha/A_log 相关投影
            # short conv depthwise
            cs = self.gdn_conv_size
            p += n_gdn * cs * (nh * kd * 2 + nh * vd)

        # ---- FFN / MoE（按平均每层的 expert 计算量）----
        for li in range(L):
            if self.is_moe_layer(li):
                p += self.n_routed_experts * 3 * h * self.moe_intermediate_size
                p += self.n_shared_experts * 3 * h * self.moe_intermediate_size
                p += h * self.n_routed_experts  # router
            else:
                p += 3 * h * self.intermediate_size

        # ---- norms ----
        p += 2 * h * L  # input_layernorm + post_attention_layernorm
        p += h          # final norm

        # ---- embedding / head ----
        embed = V * h
        p += embed
        if not self.tie_word_embeddings:
            p += embed

        return p


# ---------------------------------------------------------------------------
# 预设配方：直接对应真实旗舰模型的思路，但缩小到 CPU 可跑的规模
# ---------------------------------------------------------------------------
def tiny(name: str = "tiny", **overrides) -> AuroraConfig:
    """一组已被验证可用的测试/教学规模配方。"""
    presets: Dict[str, AuroraConfig] = {
        # ~ 用于单元测试：极快
        "nano": AuroraConfig(
            vocab_size=1024, hidden_size=128, num_layers=4,
            num_attention_heads=4, q_lora_rank=64, kv_lora_rank=64,
            qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
            num_key_value_heads=2, swa_head_dim=32, sliding_window=32,
            gdn_num_heads=4, gdn_key_dim=32, gdn_value_dim=32,
            intermediate_size=256, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
            mixer_pattern="DDSM", max_position_embeddings=512,
            num_nextn_predict_layers=1,
        ),
        # ~ 60M：CPU 上几分钟可看到明显收敛
        "tiny": AuroraConfig(
            vocab_size=8192, hidden_size=512, num_layers=8,
            num_attention_heads=8, q_lora_rank=192, kv_lora_rank=160,
            qk_nope_head_dim=96, qk_rope_head_dim=32, v_head_dim=96,
            num_key_value_heads=2, swa_head_dim=96, sliding_window=256,
            gdn_num_heads=8, gdn_key_dim=96, gdn_value_dim=96,
            intermediate_size=1024, moe_intermediate_size=256,
            n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
            mixer_pattern="DDSMDDDM", max_position_embeddings=2048,
            rope_theta=10000.0, num_nextn_predict_layers=1,
        ),
        # ~ 对标的 DeepSeek-V3 结构比例（Kimi K2 风格 2 层 1 全局），仅放大维度用
        "k2_like": AuroraConfig(
            vocab_size=32768, hidden_size=768, num_layers=12,
            num_attention_heads=12, q_lora_rank=384, kv_lora_rank=256,
            qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
            num_key_value_heads=4, swa_head_dim=128, sliding_window=512,
            gdn_num_heads=12, gdn_key_dim=128, gdn_value_dim=128,
            intermediate_size=2048, moe_intermediate_size=512,
            n_routed_experts=16, n_shared_experts=1, num_experts_per_tok=4,
            mixer_pattern="DDM", max_position_embeddings=4096,
            num_nextn_predict_layers=2,
        ),
    }
    if name not in presets:
        raise ValueError(f"未知预设 {name}; 可选: {list(presets)}")
    return presets[name].evolve(**overrides)
