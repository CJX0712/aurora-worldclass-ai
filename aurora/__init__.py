# -*- coding: utf-8 -*-
"""
Aurora — 启明星。一个正面战线（frontier-class）的开放语言模型栈。

Aurora 不是一个玩具实现。它把 2024-2026 年间被万亿参数级旗舰模型逐一验证过的
全部关键结构，收敛进一套统一、可组合、可验证的代码库：

    MLA（低秩键值联合压缩）      DeepSeek-V2/V3, Kimi K2
    Gated DeltaNet（线性注意力）  Kimi K3 KDA, Qwen3.5/3.6 GDN
    Sliding-Window / GQA         Gemma 3/4, Mistral
    DeepSeekMoE + 共享专家        DeepSeek-V3, Kimi K2
    Quantile Balancing           Kimi K3（无辅助损失负载均衡）
    MuonClip                     Kimi K2（1T 参数零 loss spike）
    MTP 多 token 预测             DeepSeek-V3, MiMo
    PagedAttention                vLLM / Orca
    GRPO                          DeepSeekMath / DeepSeek-R1

作者: 晨星
许可: Apache-2.0
"""

__version__ = "0.1.0"
__author__ = "晨星 (Chenxing)"

from aurora.config import AuroraConfig, MIXER_KINDS, build_mixer_plan
from aurora.model import AuroraForCausalLM, AuroraModel
from aurora.data.tokenizer import ByteLevelBPETokenizer
from aurora.data.dataset import TokenizedDataset, TextCorpusDataset, build_pretrain_dataloader
from aurora.train.trainer import Trainer
from aurora.train.grpo import GRPOTrainer
from aurora.inference.engine import AuroraEngine
from aurora.optim import MuonClip, apply_qk_clip, muon_clip_split_params

__all__ = [
    "AuroraConfig",
    "AuroraForCausalLM",
    "AuroraModel",
    "AuroraEngine",
    "Trainer",
    "GRPOTrainer",
    "MuonClip",
    "apply_qk_clip",
    "muon_clip_split_params",
    "ByteLevelBPETokenizer",
    "TokenizedDataset",
    "TextCorpusDataset",
    "build_pretrain_dataloader",
    "MIXER_KINDS",
    "build_mixer_plan",
    "__version__",
    "__author__",
]
