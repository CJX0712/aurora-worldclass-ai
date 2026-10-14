# -*- coding: utf-8 -*-
"""通用工具：参数量统计、设备解析、种子设置。"""

from __future__ import annotations

import torch


def count_parameters(module) -> int:
    """统计（默认全部）参数量。"""
    return sum(p.numel() for p in module.parameters())


def count_trainable_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def resolve_device(prefer: str = "cpu") -> str:
    """返回可用设备；本工程面向 CPU 友好部署，默认回退 cpu。"""
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer == "mps" and getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
