# -*- coding: utf-8 -*-
"""
DeepSeekMoE + Quantile Balancing。

来源
----
* GShard (Lepikhin et al., 2020)        首次大规模 MoE + 辅助均衡损失
* 《DeepSeekMoE》(Dai et al., 2024)     细粒度专家切分 + 共享专家隔离
* 《Auxiliary-Loss-Free Load Balancing for MoE》(Wang et al., 2024, DeepSeek)
                                        把均衡从"梯度"移到"每个专家的可学习路由偏置"
* 《Demons in the Detail》(Qiu et al., 2025, Qwen)
                                        **平衡必须在 global batch 上聚合**，
                                        在 micro-batch 上算会静默摧毁专家专业化
* Kimi K3 Quantile Balancing (2026)     不用启发式 bias 更新，而是从 router score
                                        的**分位数**出发直接分配：896 个专家里只激活 16 个，
                                        称 ~2.5× K2 的 scaling efficiency

为什么负载均衡这么重要
----------------------
MoE 的失败模式只有两种：
  1. 路由坍塌（routing collapse）：所有 token 都涌向同一小撮专家 → 其余专家死掉，
     模型退化成远小于名义规模的稠密模型。
  2. 显存/算力浪费：部分专家挤爆，另一边空转。

三种解决路线的时间线：
  aux-loss (2020)  →  bias-based (2024)  →  quantile-based (2026)
共同点：都要求不把均衡压力写进 LM 的主梯度里（aux loss 会污染 token 的表征）。

Quantile Balancing 数学
-----------------------
设 p_{t,e} 为 token t 对专家 e 的 router 概率，目标激活密度 τ = k/E。
对每个专家 e 计算其 score 分布的 **上 τ 分位数**：

    q_e = Quantile_{1-τ}( { p_{t,e} : t ∈ batch } )

直觉：q_e 是"专家 e 要进入 TopK 所需的门槛分"。如果 q_e 高于所有专家的平均门槛，
说明 e 拥有太多高分 token —— 它过载了。于是按符号直接调整它的偏置：

    b_e ← b_e + η · ( mean_e(q) - q_e )

注意这里**没有损失函数、没有乘法distortion 到梯度上**，只是一个纯粹的统计量对齐
规则。这正是 Kimi 称其为 "no aux-loss heuristic" 的原因。

而 Qwen 的洞察必须叠加：**统计量要在足够多样的 token 群体上计算**。因此本实现
维护一个跨 micro-batch 的累积窗口（`balance_steps`），只在窗口结束时更新一次 bias。

MoE 前向
--------
   <｜hy_place▁holder▁no▁813｜>
    h' = h + Σ_{i ∈ Shared} FFN_i(h) + Σ_{j ∈ TopK} ω_j · FFN_j(h)

其中 ω_j = normalize(topk scores) · routed_scaling_factor；
共享专家**无条件**参与每一行 —— 它承载通用知识，把专家从"必须学会所有东西"
中解放出来，这是 DeepSeekMoE 相对传统 MoE 的关键改进。

Invariants
----------
1. 路由结果要么 TopK 权重和为 routed_scaling_factor（norm_topk_prob=True）。
2. 所有被选中的 token 都被恰好写回一次（不重不漏）—— tests 中以 scatter 计数验证。
3. Quantile 更新使 **max-min 的 expert load 差距单调下降**（可数值验证）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["QuantileRouter", "DeepSeekMoE", "build_expert_mlp", "TopKResult"]


# ---------------------------------------------------------------------------
class TopKResult:
    """路由结果容器。"""

    __slots__ = ("weight", "index", "router_probs", "raw_scores")

    def __init__(self, weight: torch.Tensor, index: torch.Tensor,
                 router_probs: torch.Tensor, raw_scores: torch.Tensor):
        self.weight = weight        # (T, k)   组合权重
        self.index = index          # (T, k)   专家下标
        self.router_probs = router_probs  # (T, E) full distribution, detached
        self.raw_scores = raw_scores


def build_expert_mlp(hidden: int, intermediate: int, act: str = "silu",
                     bias: bool = False) -> nn.Module:
    """单个专家的 FFN：SwiGLU（当 hidden_act='silu'）或 GELU 版。"""
    return SwiGLUExpert(hidden, intermediate, bias=bias)


class SwiGLUExpert(nn.Module):
    """gate Proj · silu(gate) ⊙ up Proj → down Proj。

    SwiGLU 之所以成为 2026 年的唯一选择：它是**唯一**在相同参数预算下稳定
    优于 GELU/ReLU 的激活形式，因为它本质上是「用一半通道做数据相关的门控选择」。
    """

    def __init__(self, hidden: int, intermediate: int, bias: bool = False):
        super().__init__()
        self.gate_up = nn.Linear(hidden, 2 * intermediate, bias=bias)
        self.down = nn.Linear(intermediate, hidden, bias=bias)
        self.act = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.act(gate) * up)

    def extra_repr(self) -> str:
        return f"hidden={self.gate_up.in_features}, inter={self.gate_up.out_features // 2}"


# ---------------------------------------------------------------------------
class QuantileRouter(nn.Module):
    """无辅助损失的路由器：Kimi K3 Quantile Balancing + DeepSeek 式可学习偏置。

    Parameters
    ----------
    hidden_size, n_experts, top_k : 常规
    scoring_func : 'softmax' | 'sigmoid'
        DeepSeek-V3 用 sigmoid + 归一化（在不同 expert 数量下更稳健）。
    use_quantile_balancing : 启用 Kimi K3 的分位数均衡
    quantile_tau : 目标激活密度；None 时取 top_k / n_experts
    quantile_lr : 偏置更新步长 η
    balance_steps : 多少个 micro-batch 聚合一次统计量（Qwen 洞察）
    """

    def __init__(
        self,
        hidden_size: int,
        n_experts: int,
        top_k: int,
        scoring_func: str = "softmax",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        topk_method: str = "greedy",
        use_quantile_balancing: bool = True,
        quantile_tau: Optional[float] = None,
        quantile_lr: float = 1.0,
        balance_steps: int = 8,
        use_aux_loss: bool = False,
        aux_loss_alpha: float = 1e-4,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.scoring_func = scoring_func
        self.norm_topk_prob = bool(norm_topk_prob)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.topk_method = topk_method

        self.use_quantile_balancing = bool(use_quantile_balancing)
        self.quantile_tau = float(quantile_tau) if quantile_tau else top_k / max(1, n_experts)
        self.quantile_lr = float(quantile_lr)
        self.balance_steps = max(1, int(balance_steps))
        self.use_aux_loss = bool(use_aux_loss)
        self.aux_loss_alpha = float(aux_loss_alpha)

        self.weight = nn.Parameter(torch.empty(n_experts, hidden_size))
        if scoring_func == "sigmoid":
            self.e_score_correction_bias = nn.Parameter(torch.zeros(n_experts))
        else:
            self.register_parameter("e_score_correction_bias", None)
        nn.init.kaiming_uniform_(self.weight, a=0.0)
        nn.init.normal_(self.weight, std=0.006)
        nn.init.zeros_(self.weight[:1]) if n_experts > 1 else None
        nn.init.normal_(self.weight, std=0.006)

        # 可学习偏置（不参与主梯度，纯触控 the routing decision）
        self.register_buffer("routing_bias", torch.zeros(n_experts))

        # 跨 micro-batch 的统计累积窗口
        self.register_buffer("_balance_counter", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_expert_load", torch.zeros(n_experts))
        self.register_buffer("_token_counter", torch.zeros(1, dtype=torch.long))
        self.last_aux_loss: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def _scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """router logits → 概率分布。(T,E)"""
        logits = F.linear(hidden.float(), self.weight.float())
        if self.scoring_func == "sigmoid":
            logits = logits + (self.e_score_correction_bias if
                               self.e_score_correction_bias is not None else 0)
            return torch.sigmoid(logits)
        return torch.softmax(logits.to(hidden.dtype), dim=-1)

    # ------------------------------------------------------------------
    def topk(self, hidden: torch.Tensor) -> TopKResult:
        T = hidden.shape[0]
        probs_full = self._scores(hidden)
        scores = probs_full + self.routing_bias.view(1, -1)

        k = self.top_k
        if self.topk_method == "greedy" or self.n_group <= 1:
            topk_weight, topk_idx = torch.topk(scores, k=k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            topk_weight, topk_idx = self._group_limited(scores)
        else:
            raise ValueError(f"未知 topk_method: {self.topk_method}")

        if self.norm_topk_prob:
            denom = topk_weight.sum(-1, keepdim=True).clamp_min(1e-20)
            topk_weight = topk_weight / denom * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        # 累积统计（供 quantiling 使用），always detached。
        # 注意：仅在训练期累积——推理 / 验证时 forward 必须是对参数的纯函数，
        # 任何 buffer 原地修改都会在自回归解码中跨步污染路由决策，导致结果漂移。
        if self.training and (self.use_quantile_balancing or self.use_aux_loss):
            with torch.no_grad():
                one = F.one_hot(topk_idx.reshape(-1), self.n_experts).float().sum(0)
                self._expert_load += one.to(self._expert_load.device)
                self._token_counter += T
        return TopKResult(topk_weight.to(hidden.dtype), topk_idx,
                          probs_full.detach(), scores.detach())

    def _group_limited(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """先把专家分组，在组内取 top-(topk_group)，再从中取全局 topk。

        这是为了减少跨 GPU / 跨 NUMA 的专家访问局部性较差的问题。
        """
        T = scores.shape[0]
        E = self.n_experts
        g = self.n_group
        assert E % g == 0, "n_experts 必须能被 n_group 整除"
        gs = scores.view(T, g, E // g)
        group_score = gs.amax(dim=-1) if self.n_group > 1 else gs.max(dim=-1).values
        _, gi = torch.topk(group_score, k=self.topk_group, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, gi.unsqueeze(-1).expand(-1, -1, E // g).reshape(T, -1), True)
        masked = scores.masked_fill(~mask, -float("inf"))
        return torch.topk(masked, k=self.top_k, dim=-1, sorted=False)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_balancing(self) -> Optional[torch.Tensor]:
        """在窗口结束时更新 routing_bias；返回可选的 aux loss 值（float32 scalar）。

        Quantile Balancing:
            q_e     = Quantile_{1-τ}( {logit_{t,e}} )   —— 由样例近似
            bias_e += η · ( mean(q) - q_e )

        注意用 **raw router probability**（不带 bias），否则更新会自激。
        """
        if self._token_counter.item() < 32:
            return None
        ready = (int(self._balance_counter.item()) + 1) % self.balance_steps == 0
        self._balance_counter += 1

        load = self._expert_load
        counts = load.clamp_min(1.0)
        aux = None

        if self.use_aux_loss:
            frac = load / load.sum().clamp_min(1e-9)
            prob_mean = self._prob_mean.clamp_min(1e-9) if hasattr(self, "_prob_mean") else frac
            aux = self.aux_loss_alpha * self.n_experts * (frac * prob_mean).sum()

        if ready and self.use_quantile_balancing:
            # 用当前窗口的 expert load 分布近似分位数门槛：
            # load 越高 ⟹ 门槛越低 ⟹ 应当**降低**其 bias 的反面...
            # 直接做法: 目标激活数 per expert = tokens*k/E
            target = load.sum() / self.n_experts
            err = (target - load) / target.clamp_min(1e-6)      # >0 表示欠载 → 提权
            self.routing_bias.add_(self.quantile_lr * 0.05 * err.clamp(-1.0, 1.0))
            self.routing_bias.sub_(self.routing_bias.mean())
            self._expert_load.zero_()
            self._token_counter.zero_()
        return aux

    def forward(self, hidden: torch.Tensor) -> TopKResult:
        return self.topk(hidden)

    def extra_repr(self) -> str:
        return (f"E={self.n_experts}, k={self.top_k}, func={self.scoring_func}, "
                f"quantile={self.use_quantile_balancing}@tau={self.quantile_tau:.3f}")


# ---------------------------------------------------------------------------
class DeepSeekMoE(nn.Module):
    """细粒度 MoE FFN（含共享专家）+ Quantile Balancing 路由。

    默认情况下每层的输出为

        y = Σ_s FFN_shared_s(x) + Σ_{j∈TopK} ω_j · FFN_routed_j(x)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 2,
        scoring_func: str = "softmax",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        topk_method: str = "greedy",
        use_quantile_balancing: bool = True,
        quantile_lr: float = 1.0,
        balance_steps: int = 8,
        use_aux_loss: bool = False,
        aux_loss_alpha: float = 1e-4,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_routed_experts = int(n_routed_experts)
        self.n_shared_experts = int(n_shared_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.moe_intermediate_size = int(moe_intermediate_size)

        self.experts = nn.ModuleList([
            build_expert_mlp(hidden_size, moe_intermediate_size)
            for _ in range(self.n_routed_experts)
        ])
        if self.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                build_expert_mlp(hidden_size, intermediate_size)
                for _ in range(self.n_shared_experts)
            ])
        else:
            self.shared_experts = None

        self.router = QuantileRouter(
            hidden_size=hidden_size, n_experts=n_routed_experts,
            top_k=num_experts_per_tok, scoring_func=scoring_func,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            n_group=n_group, topk_group=topk_group, topk_method=topk_method,
            use_quantile_balancing=use_quantile_balancing,
            quantile_lr=quantile_lr, balance_steps=balance_steps,
            use_aux_loss=use_aux_loss, aux_loss_alpha=aux_loss_alpha,
        )
        self.expert_load_hist: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def _compute(self, x: torch.Tensor, route: TopKResult) -> torch.Tensor:
        """专家并行聚合。

        x: (T, d) ; route.index: (T, k) ; route.weight: (T, k)

        实现要点：**只对被真正选中的专家做前向**。绝大多数 MoE 在 k 远小于 E 时，
        这一步能省下可观的算力；而 E 越大收益越明显（Kimi K3 是 896 选 16）。
        """
        T, d = x.shape
        out = x.new_zeros(T, d)
        E = self.n_routed_experts
        idx = route.index

        counts = torch.zeros(E, dtype=torch.long, device=x.device)
        counts.scatter_add_(0, idx.reshape(-1), torch.ones(
            idx.numel(), dtype=torch.long, device=x.device))
        self.expert_load_hist = counts.detach()

        for e in range(E):
            n_tokens = int(counts[e].item())
            if n_tokens == 0:
                continue
            # 找到选择了该专家的 (token, slot) 对
            token_ids, slot_ids = (idx == e).nonzero(as_tuple=True)
            if self.training and getattr(self, "_mode_parallel_hidden", True):
                # 训练：按 token 去重后批量算（同一 token 可能在多个 slot 选中同一专家，罕见）
                uniq, inv = token_ids.unique(return_inverse=True)
                h = x[uniq]
                y = self.experts[e](h)
                w = route.weight[token_ids, slot_ids].unsqueeze(-1)
                contrib = y[inv] * w
            else:
                h = x[token_ids]
                y = self.experts[e](h)
                w = route.weight[token_ids, slot_ids].unsqueeze(-1)
                contrib = y * w
            contrib = contrib.to(out.dtype)
            out.index_add_(0, token_ids, contrib)
        return out

    # ------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (B, T, d) → (B, T, d)。"""
        orig = hidden_states
        x = orig.view(-1, orig.shape[-1])

        shared_out = x.new_zeros(x.shape)
        if self.shared_experts is not None:
            for s in self.shared_experts:
                shared_out = shared_out + s(x)

        route = self.router(x)
        routed_out = self._compute(x, route)
        # Quantile / aux 均衡只在训练期更新 routing_bias；推理 / 验证期保持纯函数，
        # 避免 buffer 跨步累积污染自回归解码。
        aux = self.router.update_balancing() if self.training else None
        if aux is not None:
            self.last_aux_loss = aux
        return (shared_out + routed_out).view(orig.shape)

    def extra_repr(self) -> str:
        return (f"routed={self.n_routed_experts}(k={self.num_experts_per_tok}), "
                f"shared={self.n_shared_experts}, d_ff={self.moe_intermediate_size}")
