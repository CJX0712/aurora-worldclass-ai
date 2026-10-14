# Aurora · 晨星锻造的世界顶级 AI 系统

> **Aurora** 是一个从数学基础到前沿大模型**全栈自洽**的 AI 系统实现：把 2024–2026 年最具代表性的架构创新（DeepSeek-V3 的 MLA、Kimi K2 的 QK-Clip、Gated DeltaNet 的线性注意力、DeepSeekMoE 的细粒度专家、MTP 自投机解码、MuonClip 优化器）收敛进**一套可插拔、可验证、CPU 可跑**的代码底座。
>
> 作者：**晨星** · 目标：锻造架构最先进、性能最强、代码最优、文档最全的世界级 AI，完整开源。

---

## 0. 设计哲学：零重写 · 集大成 · 闭环交付

Aurora 不是「又一个玩具 Transformer」，而是严格遵循以下三条原则的工程产物：

| 原则 | 含义 |
|------|------|
| **零重写** | 不重复造轮子。先用 `mixer_pattern` 字符串复用世界已有最优结构，再在其上做超越式整合。 |
| **集大成** | 单一代码库同时覆盖 MLA / GDN / SWA 三种 token mixer，DeepSeekMoE，MTP，MuonClip——一个配置即可复现当今所有主流配比。 |
| **闭环交付** | 成果必须可运行、可复现、可验证，并完整开源至 GitHub（署名 **晨星**）。 |

Aurora 把「混合注意力配比」这一 2026 年的**核心设计维度**提升为一等公民超参：一行 `mixer_pattern` 即可表达——

```
"MMMM"        纯 MLA 全局注意力        —— DeepSeek-V3 / Kimi K2
"DDDD"        纯 Gated DeltaNet        —— 纯线性 recurrent
"DSDSDSDSM"   Gemma 3/4 式 5:1 局部:全局
"DDDDDDDM"    MiniMax-M1 式 7:1 lightning
"DDDSDDDM"    Qwen3.5 式 ~75% GDN + 25% softmax
```

---

## 1. 架构总览

```
输入 token
   │
   ▼
Embedding (权重绑定 lm_head)
   │
   ▼
┌───────────────────── AuroraBlock × N ─────────────────────┐
│  pre-norm 残差:  h = h + mixer( RMSNorm(h) )              │
│  pre-norm 残差:  h = h + ffn(   RMSNorm(h) )              │
│                                                          │
│  mixer ∈ { MLA (M) | GatedDeltaNet (D) | SWA (S) }       │
│  ffn   ∈ { DeepSeekMoE | SwiGLU }                         │
└──────────────────────────────────────────────────────────┘
   │
   ▼
Final RMSNorm → lm_head → logits
   │
   ▼
MTP 头（Multi-Token Prediction）—— 同时是训练监督源 + 推理自投机草稿器
```

### 1.1 三种 Token Mixer（可插拔 slot）

| Mixer | 类别 | KV 缓存 | 代表模型 | 关键特性 |
|-------|------|----------|----------|----------|
| **MLA** (M) | 全局注意力 · KV 低秩压缩 | `d_c + d_r`/token（与头数无关） | DeepSeek-V3 / Kimi K2 | 解耦 RoPE + 潜变量缓存，KV 成本从 O(n_h) 降到 O(1) |
| **GDN** (D) | 线性注意力 · recurrent | **常数** O(H·Dv·Dk) | Gated DeltaNet / Kimi KDA | gated delta rule + 短卷积，S-NIAH 检索 92 分 |
| **SWA** (S) | 滑动窗口 softmax · GQA | 随长度线性 | Gemma / MiniMax | 局部注意力，长上下文窗口外截断 |

### 1.2 关键子系统

- **MuonClip 优化器**（`aurora/optim/muon.py`）：2D 权重走 Muon（极分解正交因子，精确 SVD），embedding / lm_head / 1D / norm 走 AdamW；每步 optimizer 后跑 **Kimi K2 QK-Clip** 把超额 attention logit 钉在阈值 τ 以下，杜绝 loss spike。
- **DeepSeekMoE + Quantile Balancing**（`aurora/modules/moe.py`）：细粒度路由专家 + 共享专家；训练期按 router score 分位数动态提权，零辅助损失漂移。
- **MTP 自投机解码**（`aurora/modules/mtp.py` + `aurora/inference/engine.py`）：MTP 头当免费草稿器，一次主模型前向严格验证 k 个候选，greedy 下输出与朴素贪婪**逐 token 一致**。
- **连续批处理推理引擎**：异构缓存（MLACache / DeltaNetCache / SWACache）沿 batch 维通用拼接/切分，封装成一次批量 `forward`，对外只暴露 `generate`。

---

## 2. 可验证不变量（Invariants）

每一处核心实现都有一条**可被独立实现交叉验证**的硬约束。Aurora 的测试套件（`tests/`）逐条锁定它们：

| 模块 | 不变量 | 验证方式 |
|------|--------|----------|
| Muon 极分解 | `max\|singular_value(U·Vᵀ) − 1\| < 1e-6` | 精确 SVD vs 数值迭代 |
| MLA QK-Clip | 裁剪后 `max attention logit = τ`（精确） | 裁剪前后 logit 对比 |
| GDN chunk vs recurrent | `‖chunk_out − recurrent_out‖ < 1e-5` | fp32 逐 token 比对 |
| MLA 增量 vs 全量 | `‖incremental − full‖ < 1e-5` | 缓存路径 vs 重算路径 |
| Byte-level BPE | `decode(encode(s)) == s` | 任意 UTF-8 无损往返 |
| **引擎 == 朴素贪婪** | `engine.generate(...) == greedy_reference(...)` | 逐 token 一致 |
| **MTP 投机 == 非投机** | `speculative(3) == greedy` | greedy 逐 token 一致 |
| GRPO | `loss finite ∧ grad 可回传` | 单步训练 |

> **关键工程修复**：MoE 的 Quantile Balancing 原本在 `forward` 中**无条件**原地修改 `routing_bias` 等持久 buffer。这在自回归解码中会跨步污染路由决策，导致引擎输出随步数漂移（与朴素贪婪不一致）。已将其严格门控在 `model.training` 之下——**推理/验证期 forward 是对参数的纯函数**，23 项测试全绿。

---

## 3. 安装

```bash
# 仅依赖 PyTorch（CPU 即可运行全部示例与不变量测试）
pip install torch
python -c "import aurora; print(aurora.__file__)"
```

Aurora 刻意保持**零非必要依赖**：分词器、数据集、训练、推理引擎均为纯 `torch` + 标准库实现。

---

## 4. 快速上手

```python
import torch
from aurora.config import tiny
from aurora.model import AuroraForCausalLM
from aurora.inference.engine import AuroraEngine

# 1) 一个配置复现一款「旗舰配比」：3 层 GDN + 1 层 MLA
cfg = tiny("nano", mixer_pattern="DDSM")
model = AuroraForCausalLM(cfg).eval()

# 2) 高吞吐推理（连续批处理 + 异构 KV 缓存）
engine = AuroraEngine(model, device="cpu")
out = engine.generate(["你好，世界"], max_new_tokens=32, return_ids=True)
print(out)

# 3) MTP 自投机解码（greedy 下结果与朴素贪婪逐 token 一致，零开销加速）
fast = engine.generate(["你好，世界"], max_new_tokens=32, speculative=3)
```

### 训练（MuonClip + QK-Clip）

```python
from aurora.train.trainer import Trainer
from aurora.data.dataset import build_pretrain_dataloader

trainer = Trainer(model, cfg, lr=1e-3, muon_lr=0.02,
                  grad_clip=1.0, qk_clip_threshold=100.0)
loader = build_pretrain_dataloader(ids, batch_size=4, seq_len=32, seed=123)
for _ in range(60):
    trainer.train_step(*next(loader))
trainer.save("aurora-nano.pt")
```

### RL 微调（GRPO）

```python
from aurora.train.grpo import GRPOTrainer
grpo = GRPOTrainer(model, ref_model, lr=1e-5, device="cpu")
loss, info = grpo.train_step(ids, attn_mask, rewards, groups)
```

---

## 5. 测试

```bash
PYTHONPATH=. python tests/test_stack.py      # 训练/推理栈 6 项不变量
PYTHONPATH=. python -m pytest tests/ -q      # 全部 23 项（含 17 项组件不变量）
```

全部通过（CPU, fp32）。不变量覆盖：Muon 正交性、QK-Clip、GDN chunk/recurrent 等价、MLA 增量/全量等价、BPE 无损、引擎==贪婪、MTP==贪婪、GRPO 可训练。

---

## 6. 归因表（集大成 · 各技术来源）

Aurora 的每一项创新都**有其世界级出处**，并在此处显式标注（详见 [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md)）：

| 子系统 | 源自 | 文献 / 模型 |
|--------|------|-------------|
| MLA（KV 低秩压缩） | DeepSeek-V2 / V3 | *DeepSeek-V3 Technical Report* (2024) |
| QK-Clip（逐组件裁剪） | Kimi K2 | *Kimi K2: Open Agentic Intelligence* (2025) |
| Gated DeltaNet | Gated Delta Networks | *Improving Mamba2 with the Delta Rule* (2024) |
| DeepSeekMoE + 负载均衡 | DeepSeek-V2 | *DeepSeek-V2 / MoE 负载均衡* |
| Muon 优化器 | Keller 等 | *Muon: MomentUm Orthogonalized by Newton-schulz* (2024) |
| Multi-Token Prediction | DeepSeek-V3 | *DeepSeek-V3* §3.2 |
| 连续批处理 / Paged KV | vLLM / TensorRT-LLM | 推理系统工程范式 |
| Byte-level BPE | GPT-2 / Sennrich | *Neural Machine Translation of Rare Words* |

---

## 7. 仓库结构

```
aurora/
├── aurora/
│   ├── config.py            # 全量超参 + mixer_pattern 编排
│   ├── model.py             # AuroraModel / AuroraForCausalLM / MTP
│   ├── modules/
│   │   ├── mla.py          # Multi-head Latent Attention + QK-Clip
│   │   ├── gdn.py          # Gated DeltaNet (chunkwise + recurrent)
│   │   ├── attention.py    # Sliding-Window Attention (GQA)
│   │   ├── moe.py          # DeepSeekMoE + Quantile Balancing
│   │   ├── block.py        # 可插拔 AuroraBlock（M/D/S 统一）
│   │   ├── mtp.py          # Multi-Token Prediction 头
│   │   ├── rope.py         # RoPE / YaRN
│   │   └── norms.py        # RMSNorm / GatedRMSNorm / QKRMSNorm
│   ├── optim/
│   │   └── muon.py         # MuonClip（精确 SVD 极分解 + QK-Clip）
│   ├── data/               # Byte-level BPE 分词器 + memmap 数据集
│   ├── train/              # Trainer (MuonClip) + GRPOTrainer
│   ├── inference/          # AuroraEngine（KV 缓存 + 连续批 + MTP 自投机）
│   └── utils/              # 参数量 / 设备 / 随机种子
├── tests/
│   ├── test_invariants.py  # 17 项组件不变量
│   └── test_stack.py       # 6 项训练/推理栈不变量
└── docs/
    └── ATTRIBUTION.md      # 详细技术归因表
```

---

## 8. 许可证与署名

- 作者 / 维护者：**晨星**
- 开源协议：MIT
- 完整代码与历史提交均署名 **晨星**（GitHub `@CJX0712`）。

> 「架构最先进、性能最强、代码最优、文档最全」不是口号，而是 Aurora 对每个不变量逐条验证的工程纪律。
