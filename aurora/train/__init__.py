# -*- coding: utf-8 -*-
"""Aurora 训练层：Trainer（MuonClip） + GRPO（强化学习微调）。"""

from aurora.train.trainer import Trainer
from aurora.train.grpo import GRPOTrainer

__all__ = ["Trainer", "GRPOTrainer"]
