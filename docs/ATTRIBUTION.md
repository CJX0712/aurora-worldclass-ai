# Aurora 技术归因表（集大成 · 各技术来源）

> Aurora 严格遵循「**零重写 · 集大成**」原则：每一项核心实现都源自世界已有的最优解，并在代码中显式标注出处。
> 本表逐子系统列出技术来源、原始文献/模型与在本仓库中的落点，便于核对与致谢。

| # | 子系统 | 原始来源 | 关键文献 / 模型 | 本仓库落点 |
|---|--------|----------|----------------|-----------|
| 1 | **Multi-head Latent Attention (MLA)** | DeepSeek-AI | *DeepSeek-V2* (2024-05)、*DeepSeek-V3 Technical Report* (2024-12) | `aurora/modules/mla.py` |
| 2 | **KV 低秩压缩（d_c + d_r 缓存）** | DeepSeek-V3 | 同上；KV 成本从 O(n_h) 降到 O(1) | `aurora/modules/mla.py` `MLACache` |
| 3 | **解耦 RoPE（decoupled RoPE）** | DeepSeek-V2/V3 | 共享 kᴿ 经 RoPE，与头数无关 | `aurora/modules/mla.py` `k_pe` |
| 4 | **QK-Clip（逐组件裁剪）** | Kimi K2 | *Kimi K2: Open Agentic Intelligence* (2025-07) | `aurora/modules/mla.py` `qk_clip_step` |
| 5 | **Gated DeltaNet（线性注意力）** | Yang Song 等 | *Gated Delta Networks: Improving Mamba2 with the Delta Rule* (2024-12) | `aurora/modules/gdn.py` |
| 6 | **Delta Rule 递推** | Schlag 等 | *Parallelizing Linear Transformers with the Delta Rule* (2021/2023) | `aurora/modules/gdn.py` `recurrent_gated_delta_rule` |
| 7 | **Chunkwise 并行（WY 表示）** | 同上 | (I + strictLower(diag β·KKᵀ))⁻¹ 解三角系统 | `aurora/modules/gdn.py` `chunk_gated_delta_rule` |
| 8 | **因果短卷积** | Gated DeltaNet 论文 | depthwise conv 提供局部上下文 | `aurora/modules/gdn.py` `causal_short_conv` |
| 9 | **Sliding-Window Attention (GQA)** | Gemma / MiniMax-M1 | 局部 softmax + 分组查询注意力 | `aurora/modules/attention.py` |
| 10 | **DeepSeekMoE（细粒度专家）** | DeepSeek-AI | *DeepSeek-V2* MoE 架构 | `aurora/modules/moe.py` `DeepSeekMoE` |
| 11 | **共享专家 + 路由专家** | DeepSeek-V2 | 共享专家承担通用知识 | `aurora/modules/moe.py` `shared_experts` |
| 12 | **Quantile Balancing 路由** | Kimi K3 / Qwen3.5 | 按 router score 分位数动态提权，零辅助损失 | `aurora/modules/moe.py` `QuantileRouter` |
| 13 | **Muon 优化器** | Keller, Blumenthal 等 | *Muon: MomentUm Orthogonalized by Newton-schulz* (2024) | `aurora/optim/muon.py` |
| 14 | **极分解正交因子（精确 SVD）** | Muon 数学目标 | 谱范数最近正交矩阵 = U·Vᵀ | `aurora/optim/muon.py` `newton_schulz_5` |
| 15 | **MuonClip（Muon + QK-Clip 融合）** | Kimi K2 + Muon | 2D 权重走 Muon，每步后 QK-Clip | `aurora/optim/__init__.py` `MuonClip` |
| 16 | **Multi-Token Prediction (MTP)** | DeepSeek-V3 | *DeepSeek-V3* §3.2；MTP 头即自投机草稿器 | `aurora/modules/mtp.py` |
| 17 | **MTP 自投机解码（self-speculative）** | DeepSeek-V3 / vLLM | 草稿经主模型单次前向严格验证 | `aurora/inference/engine.py` `_spec_step` |
| 18 | **连续批处理（continuous batching）** | vLLM / TensorRT-LLM | 请求动态进出批，算力不空转 | `aurora/inference/engine.py` `generate` |
| 19 | **Paged / 异构 KV 缓存** | vLLM PagedAttention | 缓存沿 batch 维拼接/切分 | `aurora/inference/engine.py` `_stack_mixer` |
| 20 | **Byte-level BPE** | Sennrich / GPT-2 | *NMT of Rare Words*；`bytes_to_unicode` | `aurora/data/tokenizer.py` |
| 21 | **RoPE（旋转位置编码）** | Su 等 | *RoFormer* (2021)；相对位置等变 | `aurora/modules/rope.py` |
| 22 | **YaRN（长上下文扩展）** | Kimi K2 | *YaRN: Efficient LLM Scaling*；NTK-by-parts | `aurora/modules/rope.py` `YarnRotaryEmbedding` |
| 23 | **Pre-norm 残差（RMSNorm 内残差）** | DeepSeek / LLaMA | 训练稳定性标准形态 | `aurora/modules/block.py` `AuroraBlock` |
| 24 | **GRPO（组相对策略优化）** | DeepSeekMath / R1 | *DeepSeekMath*；组内相对优势 + KL 正则 | `aurora/train/grpo.py` |
| 25 | **SwiGLU FFN** | PaLM / LLaMA | 门控线性单元 | `aurora/modules/moe.py` `SwiGLUExpert` |

## 致谢与署名

- 作者 / 维护者：**晨星**（GitHub `@CJX0712`）
- Aurora 站在上述世界级工作的肩膀上，按「零重写 · 集大成」原则整合，并在每一处实现中保留其原始出处与可验证不变量。
- 所有核心算法均配有独立可验证的不变量测试（见 `tests/`），确保「整合」而非「臆造」。
