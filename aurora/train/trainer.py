# -*- coding: utf-8 -*-
"""
Aurora Trainer —— 把 MuonClip 优化器 + QK-Clip 接进标准训练循环。

来源 / 集大成
------------
* 优化器用 **MuonClip**（Kimi K2, 2025）：2D 隐藏权重走 Muon（Nesterov 动量 +
  极分解正交化），embedding / LM head / 1D / norm 走 AdamW；每步 optimizer 更新后
  跑一次 **QK-Clip**（把超额 attention logit 回缩），全程零 loss spike。
* 训练循环是语言建模的标准形态：forward(ids, labels) → loss.backward() → grad clip
  → step → qk_clip。

为什么这样切分参数（详见 `aurora.optim.muon.muon_clip_split_params`）：
Muon 只对「2D 隐藏层权重矩阵」有意义——它把更新方向推成正交矩阵，让每个特征方向
拿到同等步长；而对 embedding、LM head、norm 的 1D 缩放参数，AdamW 才是正确工具。

用法
----
    trainer = Trainer(model, config, lr=..., muon_lr=...)
    trainer.fit(dataloader, steps=2000, log_every=50)
    trainer.save("ckpt.pt")
"""

from __future__ import annotations

import os
import time
from typing import Iterable, Iterator, List, Optional

import torch

from aurora.optim import (
    MuonClip,
    apply_qk_clip,
    muon_clip_split_params,
)


def _collect_attn_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    """收集所有实现了 qk_clip_step 的注意力模块（MLA / SWA）。"""
    out: List[torch.nn.Module] = []
    for m in model.modules():
        if hasattr(m, "qk_clip_step") and callable(getattr(m, "qk_clip_step")):
            out.append(m)
    return out


class Trainer:
    """基于 MuonClip 的预训练 Trainer。"""

    def __init__(
        self,
        model: torch.nn.Module,
        config=None,
        lr: float = 3e-4,
        muon_lr: float = 0.02,
        weight_decay: float = 0.0,
        muon_weight_decay: float = 0.01,
        adam_betas: tuple = (0.9, 0.95),
        muon_momentum: float = 0.95,
        grad_clip: float = 1.0,
        qk_clip_threshold: float = 100.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.qk_clip_threshold = qk_clip_threshold

        # ---- 按 MuonClip 角色切分参数 ----
        muon_groups, adam_params = muon_clip_split_params(model)
        self.muon_opt = MuonClip(
            muon_groups, lr=muon_lr, momentum=muon_momentum,
            weight_decay=muon_weight_decay, nesterov=True,
            rms_match="kimi", muon_scale=0.2,
        )
        self.adam_opt = torch.optim.AdamW(
            adam_params, lr=lr, betas=adam_betas, weight_decay=weight_decay,
        )
        self.attn_modules = _collect_attn_modules(model)
        self.step_count = 0
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------
    def _zero(self):
        self.muon_opt.zero_grad(set_to_none=True)
        self.adam_opt.zero_grad(set_to_none=True)

    def _clip_and_step(self):
        # 梯度裁剪（作用于所有参数）
        params = [p for pg in self.muon_opt.param_groups for p in pg["params"]]
        params += list(self.adam_opt.param_groups[0]["params"])
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
        self.muon_opt.step()
        self.adam_opt.step()
        # ---- QK-Clip：每步 optimizer 之后，把超额 attention logit 钉回阈值 ----
        if self.attn_modules:
            apply_qk_clip(self.attn_modules, threshold=self.qk_clip_threshold)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _attn_max_logit(self) -> float:
        mx = 0.0
        for m in self.attn_modules:
            a = getattr(m, "attn_max_logits", None)
            if a is not None:
                mx = max(mx, float(a.max().item()))
        return mx

    # ------------------------------------------------------------------
    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
        """单步训练。返回该步 loss。"""
        self.model.train()
        self._zero()
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        out = self.model(input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        self._clip_and_step()
        self.step_count += 1
        self.loss_history.append(loss.item())
        return loss.item()

    # ------------------------------------------------------------------
    def fit(self, dataloader: Iterator, steps: int = 1000, log_every: int = 50,
            validate_every: int = 0, val_fn=None):
        """标准训练循环。dataloader 每次产出 (input_ids, labels)。"""
        t0 = time.time()
        for step in range(1, steps + 1):
            try:
                ids, lbls = next(dataloader)
            except StopIteration:
                dataloader = iter(dataloader)
                ids, lbls = next(dataloader)
            loss = self.train_step(ids, lbls)
            if log_every and step % log_every == 0:
                dt = time.time() - t0
                amax = self._attn_max_logit()
                print(f"step {step:>6}/{steps}  loss={loss:.4f}  "
                      f"attn_max_logit={amax:.3f}  ({dt:.1f}s)")
                t0 = time.time()
            if validate_every and val_fn is not None and step % validate_every == 0:
                self.model.eval()
                with torch.no_grad():
                    val = val_fn(self.model)
                print(f"  [val] {val}")
                self.model.train()

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "muon": self.muon_opt.state_dict(),
            "adam": self.adam_opt.state_dict(),
            "step": self.step_count,
        }, path)
        print(f"[trainer] saved checkpoint -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.muon_opt.load_state_dict(ckpt["muon"])
        self.adam_opt.load_state_dict(ckpt["adam"])
        self.step_count = ckpt.get("step", 0)
        print(f"[trainer] loaded checkpoint from {path}")
