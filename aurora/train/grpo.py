# -*- coding: utf-8 -*-
"""
GRPO —— Group-Relative Policy Optimization（强化学习微调）。

来源 / 集大成
------------
* DeepSeekMath (Shao et al., 2024) 与 DeepSeek-R1 (2025) 提出的稳定化 RL 算法：
  对每个 prompt **采样一组 G 个 completions**，用组内相对奖励作为优势
  A_i = (r_i − mean(r)) / std(r)，**不需要额外的价值网络（critic）**——
  这正是它相比 PPO 省掉一半参数的关键，也是 R1 能大规模 RL 的基石。
* 策略梯度部分沿用 PPO 的 clipped surrogate，再加一个对参考模型（SFT 初始化）的
  KL 正则，防止策略跑飞。

数学（单组 G 个样本，序列长度 T）
--------------------------------
    A_i           = (r_i − μ) / (σ + ε)                      ▷ 组内相对优势
    ρ_t           = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)       ▷ 重要性比
    L_policy      = − E[ min(ρ·A, clip(ρ,1−ε,1+ε)·A) ]
    L_KL          = β · E[ log π_θ − log π_ref ]
    L             = L_policy + L_KL

本实现是**参考级、可验证**的：给定「已采样的 (prompt+completion, 旧策略 logp,
奖励, 组号)」，返回可直接反向传播的 GRPO loss。采样（rollout）由调用方用推理引擎
完成（见 `aurora.inference`），本类只负责「从样本到梯度」这一段，职责单一、易测。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from aurora.optim import MuonClip, muon_clip_split_params


class GRPOTrainer:
    """GRPO 微调器：把「采样样本」变成策略梯度更新。"""

    def __init__(self, model: torch.nn.Module, ref_model: torch.nn.Module,
                 lr: float = 1e-6, kl_beta: float = 0.04, clip_eps: float = 0.2,
                 mu_clip: float = 1.0, device: str = "cpu"):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device).eval()
        self.kl_beta = kl_beta
        self.clip_eps = clip_eps
        self.device = device

        muon_groups, adam_params = muon_clip_split_params(model)
        self.muon_opt = MuonClip(muon_groups, lr=lr, weight_decay=0.0)
        self.adam_opt = torch.optim.AdamW(adam_params, lr=lr)
        self.mu_clip = mu_clip

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _old_and_ref_logp(self, input_ids, attn_mask):
        """返回当前模型（作为 old，graph 截断）与参考模型的逐 token log-prob。"""
        # old：用当前模型参数但不建图，作为重要性比分母
        out = self.model(input_ids, attention_mask=attn_mask)
        logits = out.logits.float()
        logp_old = F.log_softmax(logits[:, :-1], dim=-1)
        logp_old = torch.gather(logp_old, -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        ref = self.ref_model(input_ids, attention_mask=attn_mask)
        logp_ref = F.log_softmax(ref.logits.float()[:, :-1], dim=-1)
        logp_ref = torch.gather(logp_ref, -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return logp_old.detach(), logp_ref.detach()

    # ------------------------------------------------------------------
    def grpo_loss(self, input_ids, attn_mask, logp_old, logp_ref, rewards, group_ids,
                  loss_mask=None):
        """计算 GRPO loss。

        Parameters
        ----------
        input_ids : (B, T)       每个序列 = prompt + completion
        attn_mask  : (B, T)      1=有效 token
        logp_old   : (B, T-1)    旧策略在当前模型参数下（图断开）的逐 token logp
        logp_ref   : (B, T-1)    参考模型逐 token logp
        rewards    : (B,)         每个序列的标量奖励
        group_ids  : (B,)         每个序列所属 prompt 的组号（同组一起归一化优势）
        loss_mask  : (B, T-1)    可选，标记 completion 部分（排除 prompt 的 teacher-forcing）
        """
        B, T = input_ids.shape
        out = self.model(input_ids, attention_mask=attn_mask)
        logits = out.logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        logp_cur = torch.gather(logp, -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        if loss_mask is None:
            loss_mask = attn_mask[:, 1:].float()

        # ---- 组内相对优势 ----
        rewards = rewards.to(self.device)
        group_ids = group_ids.to(self.device)
        adv = torch.zeros(B, device=self.device)
        for g in torch.unique(group_ids):
            idx = (group_ids == g)
            r = rewards[idx]
            if r.numel() > 1:
                adv[idx] = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
            else:
                adv[idx] = 0.0
        adv = adv.unsqueeze(-1)  # (B,1) broadcast to (B,T-1)

        # ---- clipped policy surrogate ----
        ratio = torch.exp(logp_cur - logp_old)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss = -torch.min(unclipped, clipped)
        policy_loss = (policy_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

        # ---- KL 正则（向参考模型）----
        kl = (logp_cur - logp_ref) * loss_mask
        kl_loss = (kl.sum() / loss_mask.sum().clamp_min(1.0))

        loss = policy_loss + self.kl_beta * kl_loss
        return loss, dict(policy=policy_loss.item(), kl=kl_loss.item(),
                          ratio=ratio.mean().item())

    # ------------------------------------------------------------------
    def train_step(self, input_ids, attn_mask, rewards, group_ids, loss_mask=None):
        # 先算 old / ref 的 logp（图断开），再做一次带图的前向得到当前策略
        with torch.no_grad():
            logp_old, logp_ref = self._old_and_ref_logp(input_ids, attn_mask)
        self.model.train()
        self.muon_opt.zero_grad(set_to_none=True)
        self.adam_opt.zero_grad(set_to_none=True)
        loss, info = self.grpo_loss(
            input_ids, attn_mask, logp_old, logp_ref, rewards, group_ids, loss_mask)
        loss.backward()
        params = [p for pg in self.muon_opt.param_groups for p in pg["params"]]
        params += list(self.adam_opt.param_groups[0]["params"])
        if self.mu_clip > 0:
            clip_grad_norm_(params, self.mu_clip)
        self.muon_opt.step()
        self.adam_opt.step()
        return loss.item(), info
